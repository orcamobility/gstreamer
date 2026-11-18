/* SPDX-License-Identifier: LGPL-2.1-or-later */

#ifndef __GST_CUDA_KERNEL_H__
#define __GST_CUDA_KERNEL_H__

#include "gstcudabasetransform.h"

G_BEGIN_DECLS

#define GST_TYPE_CUDA_KERNEL (gst_cuda_kernel_get_type ())
G_DECLARE_FINAL_TYPE (GstCudaKernel, gst_cuda_kernel, GST, CUDA_KERNEL,
    GstCudaBaseTransform)

G_END_DECLS

#endif /* __GST_CUDA_KERNEL_H__ */
