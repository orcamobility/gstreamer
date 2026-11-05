/* SPDX-License-Identifier: LGPL-2.1-or-later */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcudakernellanczos.h"
#include <string.h>
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
    LanczosAxis horizontal[2]; \
    LanczosAxis vertical[2]; \
  };

LANCZOS_LAYOUT

static const char lanczos_source[] = G_STRINGIFY (LANCZOS_LAYOUT) R"CUDA(
namespace {
constexpr int kBlockWidth = 32;
constexpr int kTileRows = 128;
}  // namespace

extern "C" __global__ void gst_cuda_kernel_v1(const unsigned char* src,
                                              unsigned long long src_pitch,
                                              unsigned char* dst,
                                              unsigned long long dst_pitch,
                                              unsigned int src_width,
                                              unsigned int src_height,
                                              unsigned int dst_width,
                                              unsigned int dst_height,
                                              unsigned int channels,
                                              unsigned int plane,
                                              const unsigned char* parameters,
                                              unsigned int parameters_size) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y0 = blockIdx.y * blockDim.y;
  const unsigned int y = y0 + threadIdx.y;
  const bool valid_x = x < dst_width * channels;

  if (src_width == dst_width && src_height == dst_height) {
    if (valid_x && y < dst_height) {
      dst[y * dst_pitch + x] = src[y * src_pitch + x];
    }
    return;
  }

  if (parameters_size < sizeof(LanczosParameters)) {
    return;
  }
  const LanczosParameters& geometry =
      *reinterpret_cast<const LanczosParameters*>(parameters);
  const auto horizontal = geometry.horizontal[plane];
  const auto vertical = geometry.vertical[plane];
  const auto* x_offset =
      reinterpret_cast<const unsigned int*>(parameters + horizontal.offsets);
  const auto* y_offset =
      reinterpret_cast<const unsigned int*>(parameters + vertical.offsets);
  const auto* x_weights =
      reinterpret_cast<const float*>(parameters + horizontal.weights);
  const auto* y_weights =
      reinterpret_cast<const float*>(parameters + vertical.weights);

  // Horizontal intermediates live in shared memory and are reused by the
  // output rows in this block. All threads reach the barriers at tile edges.
  __shared__ float tile[kBlockWidth * kTileRows];
  unsigned int first = y_offset[y0];
  unsigned int last = first + vertical.taps;
  for (unsigned int row = y0 + 1; row < min(dst_height, y0 + blockDim.y);
       ++row) {
    first = min(first, y_offset[row]);
    last = max(last, y_offset[row] + vertical.taps);
  }

  const unsigned int pixel = x / channels;
  const unsigned int channel = x % channels;
  float sum = 0.0f;
  for (unsigned int chunk = first; chunk < last; chunk += kTileRows) {
    const unsigned int end = min(chunk + kTileRows, last);
    for (unsigned int row = threadIdx.y; row < end - chunk; row += blockDim.y) {
      if (valid_x) {
        float filtered = 0.0f;
        for (unsigned int k = 0; k < horizontal.taps; ++k) {
          filtered += src[(chunk + row) * src_pitch +
                          (x_offset[pixel] + k) * channels + channel] *
                      x_weights[pixel * horizontal.taps + k];
        }
        tile[row * kBlockWidth + threadIdx.x] = filtered;
      }
    }
    __syncthreads();

    if (valid_x && y < dst_height) {
      const unsigned int begin_row = max(chunk, y_offset[y]);
      const unsigned int end_row = min(end, y_offset[y] + vertical.taps);
      for (unsigned int row = begin_row; row < end_row; ++row) {
        sum += tile[(row - chunk) * kBlockWidth + threadIdx.x] *
               y_weights[y * vertical.taps + row - y_offset[y]];
      }
    }
    if (end < last) {
      __syncthreads();
    }
  }
  if (valid_x && y < dst_height) {
    dst[y * dst_pitch + x] =
        static_cast<unsigned char>(min(255, max(0, __float2int_rn(sum))));
  }
}

)CUDA";

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
    g_hash_table_insert (programs, GUINT_TO_POINTER (device), program);
  }
  G_UNLOCK (lanczos_cache);

  return static_cast<const gchar *> (program);
}

static gboolean
append_axis (guint input, guint output, GstVideoResamplerMethod method,
    GByteArray * bytes, LanczosAxis * axis)
{
  GstVideoResampler resampler = {};
  if (!gst_video_resampler_init (&resampler, method,
          GST_VIDEO_RESAMPLER_FLAG_NONE, output, 0, 0.0, input, output, NULL))
    return FALSE;

  gsize offset_size = static_cast<gsize> (output) * sizeof (guint);
  gsize tap_count = static_cast<gsize> (output) * resampler.max_taps;
  gsize end = bytes->len + offset_size + tap_count * sizeof (gfloat);
  if (end > G_MAXUINT) {
    gst_video_resampler_clear (&resampler);
    return FALSE;
  }

  axis->offsets = bytes->len;
  axis->weights = bytes->len + offset_size;
  axis->taps = resampler.max_taps;
  g_byte_array_set_size (bytes, end);
  memcpy (bytes->data + axis->offsets, resampler.offset, offset_size);
  for (gsize i = 0; i < tap_count; i++) {
    gfloat weight = static_cast<gfloat> (resampler.taps[i]);
    memcpy (bytes->data + axis->weights + i * sizeof (weight), &weight,
        sizeof (weight));
  }
  gst_video_resampler_clear (&resampler);
  return TRUE;
}

GBytes *
gst_cuda_kernel_lanczos_parameters (const GstVideoInfo * input,
    const GstVideoInfo * output)
{
  LanczosParameters geometry = {};

  if (input->width == output->width && input->height == output->height)
    return g_bytes_new (NULL, 0);

  g_autoptr (GByteArray) bytes = g_byte_array_new ();
  g_byte_array_set_size (bytes, sizeof (geometry));

  for (guint plane = 0; plane < 2; plane++) {
    GstVideoResamplerMethod method = plane == 0
        ? GST_VIDEO_RESAMPLER_METHOD_LANCZOS : GST_VIDEO_RESAMPLER_METHOD_LINEAR;
    if (!append_axis (input->width >> plane, output->width >> plane, method,
            bytes, &geometry.horizontal[plane]) ||
        !append_axis (input->height >> plane, output->height >> plane, method,
            bytes, &geometry.vertical[plane]))
      return NULL;
  }

  memcpy (bytes->data, &geometry, sizeof (geometry));
  return g_byte_array_free_to_bytes (g_steal_pointer (&bytes));
}
