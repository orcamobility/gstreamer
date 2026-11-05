/* SPDX-License-Identifier: LGPL-2.1-or-later */

/**
 * SECTION:element-cudakernel
 * @title: cudakernel
 * @short_description: Runs a CUDA module on video frames
 *
 * Runs a trusted PTX, cubin or fatbin module on progressive NV12 CUDA
 * buffers. By default the output has the input dimensions. A downstream
 * caps filter can request different dimensions.
 *
 * The module must export this versioned CUDA entry point:
 *
 * |[<!-- language="cpp" -->
 * extern "C" __global__ void gst_cuda_kernel_v1 (
 *     const unsigned char *src, unsigned long long src_pitch,
 *     unsigned char *dst, unsigned long long dst_pitch,
 *     unsigned int src_width, unsigned int src_height,
 *     unsigned int dst_width, unsigned int dst_height,
 *     unsigned int channels, unsigned int plane,
 *     const unsigned char *parameters, unsigned int parameters_size);
 * ]|
 *
 * The element launches it once for each plane, using 32 by 8 threads per
 * block and a grid covering dst_width * channels bytes by dst_height rows.
 * The module uses 64-bit device addresses. Dimensions are in samples and
 * pitches are in bytes. Plane 0 is Y with one channel. Plane 1 is interleaved
 * UV with two channels and half the width and height. The kernel must guard
 * edge threads and write the whole visible output plane. All buffers are
 * valid until the launch completes.
 *
 * Applications may provide a read-only parameter block as GBytes. The
 * device pointer and size are zero when the block is empty. Module,
 * parameters and dynamic shared memory settings are configurable in READY.
 * A module runs in the pipeline's CUDA context with access to its memory.
 *
 * ## Example launch line
 * |[
 * gst-launch-1.0 videotestsrc ! video/x-raw,format=NV12 ! cudaupload \
 *     ! cudakernel module-location=effect.ptx ! cudadownload ! autovideosink
 * ]|
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcudakernel.h"

GST_DEBUG_CATEGORY_STATIC (gst_cuda_kernel_debug);
#define GST_CAT_DEFAULT gst_cuda_kernel_debug

#define KERNEL_ENTRY "gst_cuda_kernel_v1"
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 8
#define KERNEL_CAPS "video/x-raw(memory:CUDAMemory), format=(string)NV12, " \
    "interlace-mode=(string)progressive, " \
    "width=(int)[2,2147483646,2], height=(int)[2,2147483646,2]"

struct _GstCudaKernel
{
  GstCudaBaseTransform parent;
  gchar *module_location;
  GBytes *parameters;
  guint shared_memory;

  GstCudaContext *module_context;
  CUmodule module;
  CUfunction function;
  CUdeviceptr device_parameters;
  guint32 parameters_size;
};

enum
{
  PROP_0,
  PROP_MODULE_LOCATION,
  PROP_PARAMETERS,
  PROP_SHARED_MEMORY,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS (KERNEL_CAPS));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS (KERNEL_CAPS));

#define gst_cuda_kernel_parent_class parent_class
G_DEFINE_TYPE (GstCudaKernel, gst_cuda_kernel, GST_TYPE_CUDA_BASE_TRANSFORM);

static void
gst_cuda_kernel_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCudaKernel *self = GST_CUDA_KERNEL (object);
  GstElement *element = GST_ELEMENT (object);
  GBytes *old_parameters = NULL;

  GST_OBJECT_LOCK (element);
  if (GST_STATE (element) > GST_STATE_READY ||
      GST_STATE_PENDING (element) > GST_STATE_READY) {
    GST_OBJECT_UNLOCK (element);
    GST_WARNING_OBJECT (self, "Set %s in NULL or READY", pspec->name);
    return;
  }

  switch (prop_id) {
    case PROP_MODULE_LOCATION:
      g_free (self->module_location);
      self->module_location = g_value_dup_string (value);
      break;
    case PROP_PARAMETERS:
      old_parameters = self->parameters;
      self->parameters = g_value_dup_boxed (value);
      break;
    case PROP_SHARED_MEMORY:
      self->shared_memory = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (element);
  g_clear_pointer (&old_parameters, g_bytes_unref);
}

static void
gst_cuda_kernel_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCudaKernel *self = GST_CUDA_KERNEL (object);

  GST_OBJECT_LOCK (self);
  switch (prop_id) {
    case PROP_MODULE_LOCATION:
      g_value_set_string (value, self->module_location);
      break;
    case PROP_PARAMETERS:
      g_value_set_boxed (value, self->parameters);
      break;
    case PROP_SHARED_MEMORY:
      g_value_set_uint (value, self->shared_memory);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (self);
}

static void
gst_cuda_kernel_clear_module (GstCudaKernel * self)
{
  if (!self->module_context)
    return;

  if (gst_cuda_context_push (self->module_context)) {
    if (self->device_parameters)
      CuMemFree (self->device_parameters);
    CuModuleUnload (self->module);
    gst_cuda_context_pop (NULL);
  } else {
    GST_WARNING_OBJECT (self, "Could not activate CUDA context to unload module");
  }

  self->module = NULL;
  self->function = NULL;
  self->device_parameters = 0;
  self->parameters_size = 0;
  gst_clear_object (&self->module_context);
}

static gboolean
gst_cuda_kernel_stop (GstBaseTransform * transform)
{
  gst_cuda_kernel_clear_module (GST_CUDA_KERNEL (transform));
  return GST_BASE_TRANSFORM_CLASS (parent_class)->stop (transform);
}

static void
gst_cuda_kernel_finalize (GObject * object)
{
  GstCudaKernel *self = GST_CUDA_KERNEL (object);

  gst_cuda_kernel_clear_module (self);
  g_free (self->module_location);
  g_clear_pointer (&self->parameters, g_bytes_unref);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_cuda_kernel_load_module (GstCudaKernel * self, GstCudaContext * context)
{
  g_autofree gchar *image = NULL;
  g_autoptr (GError) error = NULL;
  gsize image_size;
  gsize size = self->parameters ? g_bytes_get_size (self->parameters) : 0;
  CUmodule module = NULL;
  CUfunction function;
  CUdeviceptr device_parameters = 0;
  CUresult result;
  const gchar *operation = "cuModuleLoadData";

  if (self->module_context == context)
    return TRUE;

  gst_cuda_kernel_clear_module (self);
  if (!self->module_location || !*self->module_location) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("A CUDA module-location is required"), (NULL));
    return FALSE;
  }

  if (size > G_MAXUINT32) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("CUDA kernel parameter block is too large"), (NULL));
    return FALSE;
  }

  /* g_file_get_contents() adds the terminator required by PTX modules. */
  if (!g_file_get_contents (self->module_location, &image, &image_size, &error)
      || image_size == 0) {
    GST_ELEMENT_ERROR (self, RESOURCE, READ, ("Could not read CUDA module"),
        ("%s: %s", self->module_location,
            error ? error->message : "file is empty"));
    return FALSE;
  }

  if (!gst_cuda_context_push (context)) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("Could not activate CUDA context"), (NULL));
    return FALSE;
  }

  result = CuModuleLoadData (&module, image);
  if (result == CUDA_SUCCESS) {
    operation = "cuModuleGetFunction";
    result = CuModuleGetFunction (&function, module, KERNEL_ENTRY);
  }

  if (result == CUDA_SUCCESS && size) {
    operation = "cuMemAlloc";
    result = CuMemAlloc (&device_parameters, size);
    if (result == CUDA_SUCCESS) {
      CUDA_MEMCPY2D copy = { 0, };
      copy.srcMemoryType = CU_MEMORYTYPE_HOST;
      copy.srcHost = g_bytes_get_data (self->parameters, NULL);
      copy.srcPitch = size;
      copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copy.dstDevice = device_parameters;
      copy.dstPitch = size;
      copy.WidthInBytes = size;
      copy.Height = 1;
      operation = "cuMemcpy2D";
      result = CuMemcpy2D (&copy);
    }
  }

  if (result != CUDA_SUCCESS) {
    if (device_parameters)
      CuMemFree (device_parameters);
    if (module)
      CuModuleUnload (module);
    gst_cuda_context_pop (NULL);
    GST_ELEMENT_ERROR (self, LIBRARY, INIT, ("Could not initialize CUDA kernel"),
        ("%s in %s (%s) returned CUDA error %d", operation,
            self->module_location, KERNEL_ENTRY, result));
    return FALSE;
  }

  gst_cuda_context_pop (NULL);
  self->module_context = gst_object_ref (context);
  self->module = module;
  self->function = function;
  self->device_parameters = device_parameters;
  self->parameters_size = (guint32) size;
  return TRUE;
}

static GstCaps *
gst_cuda_kernel_transform_caps (GstBaseTransform * transform,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *result = gst_caps_copy (caps);
  guint i;

  for (i = 0; i < gst_caps_get_size (result); i++) {
    GstStructure *structure = gst_caps_get_structure (result, i);
    gst_structure_remove_fields (structure, "width", "height", NULL);
  }

  if (filter) {
    GstCaps *intersection = gst_caps_intersect_full (filter, result,
        GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (result);
    result = intersection;
  }
  return result;
}

static GstCaps *
gst_cuda_kernel_fixate_caps (GstBaseTransform * transform,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstStructure *target;
  GstVideoInfo info;

  othercaps = gst_caps_make_writable (gst_caps_truncate (othercaps));
  if (gst_caps_is_empty (othercaps) || !gst_video_info_from_caps (&info, caps))
    return gst_caps_fixate (othercaps);

  target = gst_caps_get_structure (othercaps, 0);
  gst_structure_fixate_field_nearest_int (target, "width", info.width);
  gst_structure_fixate_field_nearest_int (target, "height", info.height);

  /* Preserve the input interpretation when downstream leaves it unspecified. */
  if (direction == GST_PAD_SINK) {
    if (!gst_structure_has_field (target, "colorimetry")) {
      g_autofree gchar *color =
          gst_video_colorimetry_to_string (&info.colorimetry);
      if (color)
        gst_structure_set (target, "colorimetry", G_TYPE_STRING, color, NULL);
    }
    if (!gst_structure_has_field (target, "chroma-site") &&
        info.chroma_site != GST_VIDEO_CHROMA_SITE_UNKNOWN) {
      g_autofree gchar *site = gst_video_chroma_site_to_string (info.chroma_site);
      if (site)
        gst_structure_set (target, "chroma-site", G_TYPE_STRING, site, NULL);
    }
  }

  return gst_caps_fixate (othercaps);
}

static gboolean
gst_cuda_kernel_set_info (GstCudaBaseTransform * transform,
    GstCaps * incaps, GstVideoInfo * input, GstCaps * outcaps,
    GstVideoInfo * output)
{
  if (GST_VIDEO_INFO_FORMAT (input) != GST_VIDEO_FORMAT_NV12 ||
      GST_VIDEO_INFO_FORMAT (output) != GST_VIDEO_FORMAT_NV12 ||
      input->interlace_mode != GST_VIDEO_INTERLACE_MODE_PROGRESSIVE ||
      output->interlace_mode != GST_VIDEO_INTERLACE_MODE_PROGRESSIVE ||
      ((input->width | input->height | output->width | output->height) & 1) ||
      !gst_video_colorimetry_is_equal (&input->colorimetry, &output->colorimetry)
      || input->chroma_site != output->chroma_site) {
    GST_ERROR_OBJECT (transform, "Expected even progressive NV12 dimensions "
        "and matching colorimetry and chroma site, got %" GST_PTR_FORMAT
        " -> %" GST_PTR_FORMAT, incaps, outcaps);
    return FALSE;
  }

  return gst_cuda_kernel_load_module (GST_CUDA_KERNEL (transform),
      transform->context);
}

static gboolean
gst_cuda_kernel_decide_allocation (GstBaseTransform * transform,
    GstQuery * query)
{
  GstCudaBaseTransform *cuda = GST_CUDA_BASE_TRANSFORM (transform);
  GstCaps *caps;
  GstBufferPool *pool;
  GstStructure *config;
  guint size, min = 0, max = 0;
  gboolean have_pool = gst_query_get_n_allocation_pools (query) > 0;

  gst_query_parse_allocation (query, &caps, NULL);
  if (!caps || cuda->out_info.size > G_MAXUINT)
    return FALSE;

  if (have_pool)
    gst_query_parse_nth_allocation_pool (query, 0, NULL, NULL, &min, &max);

  pool = gst_cuda_buffer_pool_new (cuda->context);
  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_add_option (config, GST_BUFFER_POOL_OPTION_VIDEO_META);
  gst_buffer_pool_config_set_params (config, caps, cuda->out_info.size, min, max);
  gst_buffer_pool_config_set_cuda_stream (config, cuda->stream);
  if (!gst_buffer_pool_set_config (pool, config)) {
    gst_object_unref (pool);
    return FALSE;
  }

  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_get_params (config, NULL, &size, NULL, NULL);
  gst_structure_free (config);
  if (have_pool)
    gst_query_set_nth_allocation_pool (query, 0, pool, size, min, max);
  else
    gst_query_add_allocation_pool (query, pool, size, min, max);
  gst_object_unref (pool);

  return GST_BASE_TRANSFORM_CLASS (parent_class)->decide_allocation (transform,
      query);
}

static gboolean
gst_cuda_kernel_transform_meta (GstBaseTransform * transform,
    GstBuffer * outbuf, GstMeta * meta, GstBuffer * inbuf)
{
  GType api = meta->info->api;

  /* A kernel can move image content even when its input and output caps match. */
  if (gst_meta_api_type_has_tag (api,
          g_quark_from_static_string (GST_META_TAG_VIDEO_SIZE_STR)) ||
      gst_meta_api_type_has_tag (api,
          g_quark_from_static_string (GST_META_TAG_VIDEO_ORIENTATION_STR)))
    return FALSE;

  return GST_BASE_TRANSFORM_CLASS (parent_class)->transform_meta (transform,
      outbuf, meta, inbuf);
}

static GstFlowReturn
gst_cuda_kernel_transform (GstBaseTransform * transform,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstCudaKernel *self = GST_CUDA_KERNEL (transform);
  GstCudaBaseTransform *cuda = GST_CUDA_BASE_TRANSFORM (transform);
  GstMemory *input, *output;
  const GstVideoInfo *input_info = &cuda->in_info;
  GstVideoFrame in_frame, out_frame;
  CUresult result = CUDA_SUCCESS, sync_result;
  CUstream stream;
  guint32 plane;

  if (!self->function || self->module_context != cuda->context)
    return GST_FLOW_NOT_NEGOTIATED;

  if (gst_buffer_n_memory (inbuf) != 1 || gst_buffer_n_memory (outbuf) != 1)
    goto invalid_memory;
  input = gst_buffer_peek_memory (inbuf, 0);
  output = gst_buffer_peek_memory (outbuf, 0);
  if (!gst_is_cuda_memory (input) || !gst_is_cuda_memory (output) ||
      GST_CUDA_MEMORY_CAST (input)->context != cuda->context ||
      GST_CUDA_MEMORY_CAST (output)->context != cuda->context)
    goto invalid_memory;

  /* A directly allocated CUDA buffer may carry its pitch only in memory. */
  if (!gst_buffer_get_video_meta (inbuf)) {
    input_info = &GST_CUDA_MEMORY_CAST (input)->info;
    if (GST_VIDEO_INFO_FORMAT (input_info) != GST_VIDEO_FORMAT_NV12 ||
        input_info->width != cuda->in_info.width ||
        input_info->height != cuda->in_info.height)
      goto invalid_memory;
  }

  if (!gst_video_frame_map (&in_frame, input_info, inbuf,
          GST_MAP_READ | GST_MAP_CUDA))
    goto invalid_memory;

  /* Mapping can enqueue an upload on the input memory's stream. */
  if (gst_cuda_memory_get_stream (GST_CUDA_MEMORY_CAST (input)) != cuda->stream)
    gst_cuda_memory_sync (GST_CUDA_MEMORY_CAST (input));

  if (!gst_video_frame_map (&out_frame, &cuda->out_info, outbuf,
          GST_MAP_WRITE | GST_MAP_CUDA)) {
    gst_video_frame_unmap (&in_frame);
    goto invalid_memory;
  }

  for (plane = 0; plane < 2; plane++) {
    if (!GST_VIDEO_FRAME_PLANE_DATA (&in_frame, plane) ||
        !GST_VIDEO_FRAME_PLANE_DATA (&out_frame, plane) ||
        GST_VIDEO_FRAME_PLANE_STRIDE (&in_frame, plane) < cuda->in_info.width ||
        GST_VIDEO_FRAME_PLANE_STRIDE (&out_frame, plane) < cuda->out_info.width) {
      gst_video_frame_unmap (&out_frame);
      gst_video_frame_unmap (&in_frame);
      goto invalid_memory;
    }
  }

  if (!gst_cuda_context_push (cuda->context)) {
    gst_video_frame_unmap (&out_frame);
    gst_video_frame_unmap (&in_frame);
    GST_ELEMENT_ERROR (self, LIBRARY, FAILED,
        ("Could not activate CUDA context"), (NULL));
    return GST_FLOW_ERROR;
  }

  stream = gst_cuda_stream_get_handle (cuda->stream);
  for (plane = 0; plane < 2; plane++) {
    guint64 src = (guintptr) GST_VIDEO_FRAME_PLANE_DATA (&in_frame, plane);
    guint64 dst = (guintptr) GST_VIDEO_FRAME_PLANE_DATA (&out_frame, plane);
    guint64 src_pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&in_frame, plane);
    guint64 dst_pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&out_frame, plane);
    guint32 src_width = cuda->in_info.width >> plane;
    guint32 src_height = cuda->in_info.height >> plane;
    guint32 dst_width = cuda->out_info.width >> plane;
    guint32 dst_height = cuda->out_info.height >> plane;
    guint32 channels = plane + 1;
    guint64 parameters = self->device_parameters;
    void *args[] = { &src, &src_pitch, &dst, &dst_pitch, &src_width,
      &src_height, &dst_width, &dst_height, &channels, &plane, &parameters,
      &self->parameters_size };

    result = CuLaunchKernel (self->function,
        (dst_width * channels + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (dst_height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT, 1, BLOCK_WIDTH,
        BLOCK_HEIGHT, 1, self->shared_memory, stream, args, NULL);
    if (result != CUDA_SUCCESS)
      break;
  }

  /* Even on a partial launch failure, keep imported input alive until all
   * queued reads finish. Downstream can then consume output on any stream. */
  sync_result = CuStreamSynchronize (stream);
  if (sync_result == CUDA_SUCCESS)
    GST_MEMORY_FLAG_UNSET (output, GST_CUDA_MEMORY_TRANSFER_NEED_SYNC);
  gst_cuda_context_pop (NULL);
  gst_video_frame_unmap (&out_frame);
  gst_video_frame_unmap (&in_frame);

  if (result == CUDA_SUCCESS && sync_result == CUDA_SUCCESS)
    return GST_FLOW_OK;

  GST_ELEMENT_ERROR (self, LIBRARY, FAILED, ("CUDA kernel failed"),
      ("Launch error %d, stream completion error %d", result, sync_result));
  return GST_FLOW_ERROR;

invalid_memory:
  GST_ELEMENT_ERROR (self, STREAM, FORMAT,
      ("Expected mappable NV12 CUDA buffers in the negotiated context"), (NULL));
  return GST_FLOW_ERROR;
}

static void
gst_cuda_kernel_class_init (GstCudaKernelClass * klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstCudaBaseTransformClass *cuda_class = GST_CUDA_BASE_TRANSFORM_CLASS (klass);
  GParamFlags flags = G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
      GST_PARAM_MUTABLE_READY;

  GST_DEBUG_CATEGORY_INIT (gst_cuda_kernel_debug, "cudakernel", 0,
      "CUDA kernel video transform");
  object_class->set_property = gst_cuda_kernel_set_property;
  object_class->get_property = gst_cuda_kernel_get_property;
  object_class->finalize = gst_cuda_kernel_finalize;
  g_object_class_install_property (object_class, PROP_MODULE_LOCATION,
      g_param_spec_string ("module-location", "Module location",
          "Path to a trusted PTX, cubin or fatbin module", NULL, flags));
  g_object_class_install_property (object_class, PROP_PARAMETERS,
      g_param_spec_boxed ("parameters", "Parameters",
          "Read-only parameter bytes copied to CUDA memory", G_TYPE_BYTES,
          flags));
  g_object_class_install_property (object_class, PROP_SHARED_MEMORY,
      g_param_spec_uint ("shared-memory", "Shared memory",
          "Dynamic shared memory bytes per CUDA thread block", 0, G_MAXUINT, 0,
          flags));

  gst_element_class_add_static_pad_template (element_class, &sink_template);
  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_set_static_metadata (element_class, "CUDA kernel filter",
      "Filter/Video", "Apply a CUDA kernel to NV12 video", "Coram");

  transform_class->passthrough_on_same_caps = FALSE;
  transform_class->stop = gst_cuda_kernel_stop;
  transform_class->transform_caps = gst_cuda_kernel_transform_caps;
  transform_class->fixate_caps = gst_cuda_kernel_fixate_caps;
  transform_class->decide_allocation = gst_cuda_kernel_decide_allocation;
  transform_class->transform_meta = gst_cuda_kernel_transform_meta;
  transform_class->transform = gst_cuda_kernel_transform;
  cuda_class->set_info = gst_cuda_kernel_set_info;
}

static void
gst_cuda_kernel_init (GstCudaKernel * self)
{
}
