/* SPDX-License-Identifier: LGPL-2.1-or-later */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcudakernellanczos.h"
#include <math.h>
#include <string.h>
#include <vector>
#include <gst/cuda/gstcudanvrtc.h>

/* Offsets are relative to the start of the parameter block. Keep the
 * host/device layout identical by compiling the same declarations. */
#define LANCZOS_LAYOUT \
  struct LanczosAxis { \
    unsigned int offsets; \
    unsigned int weights; \
    unsigned int taps; \
  }; \
  struct LanczosParameters { \
    unsigned int direction; \
    LanczosAxis horizontal[2]; \
    LanczosAxis vertical[2]; \
  };

LANCZOS_LAYOUT

static const char lanczos_source[] = G_STRINGIFY (LANCZOS_LAYOUT) R"CUDA(
namespace {
constexpr int kBlockWidth = 32;
constexpr int kBlockRows = 8;              // threads per block in y
constexpr int kRowsPerThread = 4;
constexpr int kOutRows = kBlockRows * kRowsPerThread;
constexpr int kTileRows = 128;
// Weight tables up to this many taps are staged in shared memory; wider
// filters (extreme reductions) read their weights from the parameter block.
constexpr int kMaxSharedTaps = 48;

__device__ void store_pixel(unsigned char* dst, unsigned long long pitch,
                            unsigned int x, unsigned int y,
                            unsigned int width, unsigned int height,
                            unsigned int channels, unsigned int direction,
                            unsigned char value) {
  const unsigned int pixel = x / channels;
  unsigned int dx = pixel;
  unsigned int dy = y;
  // GstVideoOrientationMethod's eight fixed orientations. Rotate pixel
  // coordinates, keeping each interleaved UV pair in U,V order.
  switch (direction) {
    case 1: dx = height - 1 - y; dy = pixel; break;
    case 2: dx = width - 1 - pixel; dy = height - 1 - y; break;
    case 3: dx = y; dy = width - 1 - pixel; break;
    case 4: dx = width - 1 - pixel; break;
    case 5: dy = height - 1 - y; break;
    case 6: dx = y; dy = pixel; break;
    case 7: dx = height - 1 - y; dy = width - 1 - pixel; break;
  }
  dst[dy * pitch + dx * channels + x % channels] = value;
}
}  // namespace

// Each block writes kBlockWidth bytes x kOutRows rows. Horizontally filtered
// rows for the block's source span go through an FP32 shared-memory tile in
// kTileRows chunks, so there is no full-frame scratch buffer. Per-output
// accumulation order is taps ascending, then rows ascending.
extern "C" __global__ void __launch_bounds__(256) gst_cuda_kernel_v1(
    const unsigned char* __restrict__ src, unsigned long long src_pitch,
    unsigned char* __restrict__ dst, unsigned long long dst_pitch,
    unsigned int src_width, unsigned int src_height, unsigned int dst_width,
    unsigned int dst_height, unsigned int channels, unsigned int plane,
    const unsigned char* __restrict__ parameters,
    unsigned int parameters_size) {
  const unsigned int x = blockIdx.x * kBlockWidth + threadIdx.x;
  const unsigned int y0 = blockIdx.y * kOutRows;
  const bool valid_x = x < dst_width * channels;

  if (parameters_size < sizeof(LanczosParameters)) {
    return;
  }
  const auto geometry =
      *reinterpret_cast<const LanczosParameters*>(parameters);

  if (src_width == dst_width && src_height == dst_height) {
    for (int r = 0; r < kRowsPerThread; ++r) {
      const unsigned int y = y0 + threadIdx.y + r * kBlockRows;
      if (valid_x && y < dst_height)
        store_pixel(dst, dst_pitch, x, y, dst_width, dst_height, channels,
                    geometry.direction, src[y * src_pitch + x]);
    }
    return;
  }

  const auto horizontal = geometry.horizontal[plane];
  const auto vertical = geometry.vertical[plane];
  const auto* __restrict__ x_offset =
      reinterpret_cast<const unsigned int*>(parameters + horizontal.offsets);
  const auto* __restrict__ y_offset =
      reinterpret_cast<const unsigned int*>(parameters + vertical.offsets);
  const auto* __restrict__ x_weights =
      reinterpret_cast<const float*>(parameters + horizontal.weights);
  const auto* __restrict__ y_weights =
      reinterpret_cast<const float*>(parameters + vertical.weights);
  const unsigned int htaps = horizontal.taps;
  const unsigned int vtaps = vertical.taps;
  const bool h_shared = htaps <= kMaxSharedTaps;
  const bool v_shared = vtaps <= kMaxSharedTaps;

  __shared__ float tile[kBlockWidth * kTileRows];
  __shared__ float hw[kMaxSharedTaps * kBlockWidth];
  __shared__ float vw[kMaxSharedTaps * kOutRows];

  const unsigned int tid = threadIdx.y * kBlockWidth + threadIdx.x;
  const unsigned int pixel = x / channels;
  const unsigned int channel = x % channels;
  const unsigned int y_last = min(dst_height, y0 + kOutRows) - 1;
  if (y0 > y_last) return;  // uniform across the block

  // Stage weights: hw[tap][column], vw[tap][row in block].
  if (h_shared) {
    for (unsigned int k = threadIdx.y; k < htaps; k += kBlockRows)
      hw[k * kBlockWidth + threadIdx.x] =
          valid_x ? x_weights[pixel * htaps + k] : 0.0f;
  }
  if (v_shared) {
    const unsigned int n = (y_last - y0 + 1) * vtaps;
    for (unsigned int i = tid; i < n; i += kBlockWidth * kBlockRows) {
      const unsigned int r = i / vtaps, k = i % vtaps;
      vw[k * kOutRows + r] = y_weights[(y0 + r) * vtaps + k];
    }
  }

  // Window offsets are non-decreasing (checked on the host), so the block's
  // source rows are [first, last).
  const unsigned int first = y_offset[y0];
  const unsigned int last = y_offset[y_last] + vtaps;
  const unsigned int xo = valid_x ? x_offset[pixel] : 0;
  const unsigned char* __restrict__ col = src + xo * channels + channel;

  unsigned int my_off[kRowsPerThread];
  float sum[kRowsPerThread];
#pragma unroll
  for (int r = 0; r < kRowsPerThread; ++r) {
    const unsigned int y = min(y0 + threadIdx.y + r * kBlockRows, y_last);
    my_off[r] = y_offset[y];
    sum[r] = 0.0f;
  }
  __syncthreads();

  for (unsigned int chunk = first; chunk < last; chunk += kTileRows) {
    const unsigned int end = min(chunk + kTileRows, last);
    if (valid_x) {
      for (unsigned int row = threadIdx.y; row < end - chunk;
           row += kBlockRows) {
        const unsigned char* __restrict__ p = col + (chunk + row) * src_pitch;
        float filtered = 0.0f;
        if (h_shared) {
          for (unsigned int k = 0; k < htaps; ++k)
            filtered += __ldg(p + k * channels) * hw[k * kBlockWidth + threadIdx.x];
        } else {
          for (unsigned int k = 0; k < htaps; ++k)
            filtered += __ldg(p + k * channels) * x_weights[pixel * htaps + k];
        }
        tile[row * kBlockWidth + threadIdx.x] = filtered;
      }
    }
    __syncthreads();

    if (valid_x) {
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        const unsigned int ry = threadIdx.y + r * kBlockRows;
        const unsigned int y = y0 + ry;
        if (y <= y_last) {
          const unsigned int begin_row = max(chunk, my_off[r]);
          const unsigned int end_row = min(end, my_off[r] + vtaps);
          float s = sum[r];
          if (v_shared) {
            for (unsigned int row = begin_row; row < end_row; ++row)
              s += tile[(row - chunk) * kBlockWidth + threadIdx.x] *
                   vw[(row - my_off[r]) * kOutRows + ry];
          } else {
            for (unsigned int row = begin_row; row < end_row; ++row)
              s += tile[(row - chunk) * kBlockWidth + threadIdx.x] *
                   y_weights[y * vtaps + row - my_off[r]];
          }
          sum[r] = s;
        }
      }
    }
    if (end < last) {
      __syncthreads();
    }
  }
  if (valid_x) {
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const unsigned int y = y0 + threadIdx.y + r * kBlockRows;
      if (y <= y_last)
        store_pixel(dst, dst_pitch, x, y, dst_width, dst_height, channels,
                    geometry.direction,
                    static_cast<unsigned char>(min(255, max(0, __float2int_rn(sum[r])))));
    }
  }
}

)CUDA";

void
gst_cuda_kernel_lanczos_launch_dims (guint dst_bytes, guint dst_height,
    guint * grid_x, guint * grid_y, guint * block_x, guint * block_y)
{
  /* Must match kBlockWidth, kBlockRows and kRowsPerThread above. */
  const guint rows = 8 * 4;

  *block_x = 32;
  *block_y = 8;
  *grid_x = (dst_bytes + 31) / 32;
  *grid_y = (dst_height + rows - 1) / rows;
}

G_LOCK_DEFINE_STATIC (lanczos_cache);

const gchar *
gst_cuda_kernel_lanczos_program (GstCudaContext * context)
{
  static GHashTable *programs;
  guint device;
  gpointer program = NULL;

  g_object_get (context, "cuda-device-id", &device, NULL);

  /* The image is immutable and can be loaded into any context on this device. */
  G_LOCK (lanczos_cache);
  if (!programs)
    programs = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL, g_free);

  if (!g_hash_table_lookup_extended (programs, GUINT_TO_POINTER (device),
          NULL, &program)) {
    program = gst_cuda_nvrtc_compile_cubin (lanczos_source, device);
    if (!program)
      program = gst_cuda_nvrtc_compile (lanczos_source);
    if (program)
      g_hash_table_insert (programs, GUINT_TO_POINTER (device), program);
  }
  G_UNLOCK (lanczos_cache);

  return static_cast<const gchar *> (program);
}

static double
sinc (double x)
{
  if (x == 0)
    return 1.0;
  x *= G_PI;
  return sin (x) / x;
}

/* Coefficients for one axis, matching CV-CUDA 0.18.0 HQResize with
 * NVCV_INTERP_LANCZOS and antialiasing. Luma uses Lanczos-3, sampled from a
 * table of 32 points per unit with linear interpolation; when downscaling,
 * the kernel is stretched by the scale factor so it also acts as the
 * anti-alias filter. Chroma uses a triangular filter of radius max(1, scale)
 * when downscaling and bilinear interpolation when upscaling. Each window
 * starts at ceil(centre - support / 2), weights are normalised to sum to 1,
 * and taps outside the plane are folded onto the edge sample
 * (clamp-to-edge). Float arithmetic follows CV-CUDA so the tables agree to
 * within one output code value. */
static gboolean
append_axis (guint in, guint out, gboolean luma, GByteArray * bytes,
    LanczosAxis * axis)
{
  const double a = 3.0;
  const double s = (double) in / out;
  const gboolean down = out < in;
  const gboolean bilinear = !luma && !down;
  std::vector < float >lut;
  float fscale, anchor;

  if (luma) {
    const int per_unit = 32;
    const int n = (int) (2 * a * per_unit + 1);
    lut.resize (n);
    for (int i = 0; i < n; i++) {
      float x = (float) (2 * a * (i - (n - 1) * 0.5f) / (n - 1));
      lut[i] = fabsf (x) >= a ? 0.0f : (float) (sinc (x) * sinc (x / a));
    }
    float radius = down ? (float) (a * s) : (float) a;
    float support = 2.0f * MAX ((float) a, radius);
    fscale = (n - 1) / support;
    anchor = support * 0.5f;
  } else {
    lut = { 0.0f, 1.0f, 0.0f };
    float radius = down ? (float) s : 1.0f;
    float support = MAX (1.0f, 2 * radius);
    fscale = (lut.size () - 1) / support;
    anchor = support * 0.5f;
  }
  const int n_coeffs = (int) lut.size ();

  int taps = (int) ceilf ((n_coeffs - 1) / fscale);
  if (bilinear)
    taps = 2;
  taps = CLAMP (taps, 1, (int) in);

  gsize offset_size = static_cast < gsize > (out) * sizeof (guint);
  gsize tap_count = static_cast < gsize > (out) * taps;
  gsize end = bytes->len + offset_size + tap_count * sizeof (gfloat);
  if (end > G_MAXUINT)
    return FALSE;

  axis->offsets = bytes->len;
  axis->weights = bytes->len + offset_size;
  axis->taps = taps;
  g_byte_array_set_size (bytes, end);
  guint *offsets = reinterpret_cast < guint * >(bytes->data + axis->offsets);
  gfloat *weights = reinterpret_cast < gfloat * >(bytes->data + axis->weights);

  std::vector < int >index (taps);
  std::vector < float >coeff (taps);
  std::vector < float >folded (taps);
  for (guint j = 0; j < out; j++) {
    if (bilinear) {
      float centre = (float) j * (float) s + 0.5f * (float) s - 0.5f;
      int s0 = (int) floorf (centre);
      float d = centre - s0;
      index[0] = s0;
      coeff[0] = 1.0f - d;
      if (taps > 1) {
        index[1] = s0 + 1;
        coeff[1] = d;
      } else {
        index[0] = 0;
        coeff[0] = 1.0f;
      }
    } else {
      float start_f = (float) j * (float) s + (0.5f * (float) s - 0.5f - anchor);
      int start = (int) ceilf (start_f);
      float f = (start - start_f) * fscale;
      for (int k = 0; k < taps; k++) {
        float x = f + k * fscale;
        float v = 0.0f;
        if (x > -1 && x < n_coeffs) {
          int x0 = (int) floorf (x);
          int x1 = x0 + 1;
          float f0 = x0 < 0 ? 0.0f : lut[x0];
          float f1 = x1 >= n_coeffs ? 0.0f : lut[x1];
          v = f0 + (x - x0) * (f1 - f0);
        }
        index[k] = start + k;
        coeff[k] = v;
      }
    }

    float total = 0;
    for (int k = 0; k < taps; k++)
      total += coeff[k];
    float norm = bilinear ? 1.0f : 1.0f / total;

    /* Place the window inside the plane and fold clamped taps onto it. */
    int first = CLAMP (index[0], 0, (int) in - taps);
    offsets[j] = first;
    std::fill (folded.begin (), folded.end (), 0.0f);
    for (int k = 0; k < taps; k++)
      folded[CLAMP (index[k], 0, (int) in - 1) - first] += coeff[k] * norm;
    for (int k = 0; k < taps; k++)
      weights[static_cast < gsize > (j) * taps + k] = folded[k];
  }
  return TRUE;
}

GBytes *
gst_cuda_kernel_lanczos_parameters (const GstVideoInfo * input,
    const GstVideoInfo * output, GstVideoOrientationMethod direction)
{
  LanczosParameters geometry = {};
  guint width = output->width;
  guint height = output->height;

  if (gst_cuda_kernel_lanczos_swaps_axes (direction)) {
    width = output->height;
    height = output->width;
  }
  geometry.direction = direction;
  if (input->width == width && input->height == height)
    return g_bytes_new (&geometry, sizeof (geometry));

  g_autoptr (GByteArray) bytes = g_byte_array_new ();
  g_byte_array_set_size (bytes, sizeof (geometry));

  for (guint plane = 0; plane < 2; plane++) {
    if (!append_axis (input->width >> plane, width >> plane, plane == 0,
            bytes, &geometry.horizontal[plane]) ||
        !append_axis (input->height >> plane, height >> plane, plane == 0,
            bytes, &geometry.vertical[plane]))
      return NULL;
  }

  /* The kernel relies on non-decreasing window offsets that keep every
   * window inside the plane. Refuse the geometry rather than mis-filter. */
  for (guint plane = 0; plane < 2; plane++) {
    const LanczosAxis *axes[2] =
        { &geometry.horizontal[plane], &geometry.vertical[plane] };
    const guint outs[2] = { width >> plane, height >> plane };
    const guint ins[2] = { (guint) input->width >> plane,
      (guint) input->height >> plane
    };
    for (guint a = 0; a < 2; a++) {
      const guint *off =
          reinterpret_cast < const guint * >(bytes->data + axes[a]->offsets);
      for (guint j = 0; j < outs[a]; j++) {
        if (off[j] + axes[a]->taps > ins[a] || (j && off[j] < off[j - 1]))
          return NULL;
      }
    }
  }

  memcpy (bytes->data, &geometry, sizeof (geometry));
  return g_byte_array_free_to_bytes (g_steal_pointer (&bytes));
}

gboolean
gst_cuda_kernel_lanczos_swaps_axes (GstVideoOrientationMethod direction)
{
  return direction == GST_VIDEO_ORIENTATION_90R ||
      direction == GST_VIDEO_ORIENTATION_90L ||
      direction == GST_VIDEO_ORIENTATION_UL_LR ||
      direction == GST_VIDEO_ORIENTATION_UR_LL;
}
