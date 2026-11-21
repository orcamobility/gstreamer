/* GStreamer
 * Copyright (C) 2025 Seungha Yang <seungha@centricular.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifdef __NVCC__
extern "C" {
__global__ void
GstNvJpegEncConvertMain (const unsigned char * in_uv, int in_stride,
    unsigned char * out_u, unsigned char * out_v,
    int width, int height, int out_stride)
{
  int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_pos >= width || y_pos >= height)
    return;
  const unsigned char * src = in_uv + y_pos * in_stride + x_pos * 2;
  unsigned int pos = x_pos + (y_pos * out_stride);
  out_u[pos] = src[0];
  out_v[pos] = src[1];
}
}
#else
static const gchar *GstNvJpegEncConvertMain_str = R"(
extern "C" {
__global__ void
GstNvJpegEncConvertMain (const unsigned char * in_uv, int in_stride,
    unsigned char * out_u, unsigned char * out_v,
    int width, int height, int out_stride)
{
  int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int y_pos = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_pos >= width || y_pos >= height)
    return;
  const unsigned char * src = in_uv + y_pos * in_stride + x_pos * 2;
  unsigned int pos = x_pos + (y_pos * out_stride);
  out_u[pos] = src[0];
  out_v[pos] = src[1];
}
}
)";
#endif
