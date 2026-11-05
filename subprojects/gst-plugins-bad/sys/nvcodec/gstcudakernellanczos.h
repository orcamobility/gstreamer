/* SPDX-License-Identifier: LGPL-2.1-or-later */

#ifndef __GST_CUDA_KERNEL_LANCZOS_H__
#define __GST_CUDA_KERNEL_LANCZOS_H__

#include <gst/cuda/gstcuda.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

const gchar *gst_cuda_kernel_lanczos_program (GstCudaContext * context);
GBytes *gst_cuda_kernel_lanczos_parameters (const GstVideoInfo * input,
    const GstVideoInfo * output);

G_END_DECLS

#endif /* __GST_CUDA_KERNEL_LANCZOS_H__ */
