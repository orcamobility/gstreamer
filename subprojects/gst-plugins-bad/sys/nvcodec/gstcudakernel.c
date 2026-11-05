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
 * device pointer and size are zero when the block is empty. A handler for
 * #GstCudaKernel::prepare-parameters can supply a new block when the frame
 * geometry changes. Module, parameters and dynamic shared memory settings
 * are configurable in READY.
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
  GBytes *module_data;
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
  PROP_MODULE_DATA,
  PROP_PARAMETERS,
  PROP_SHARED_MEMORY,
};

static guint prepare_parameters_signal;

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
  GBytes *old_bytes = NULL;

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
    case PROP_MODULE_DATA:
      old_bytes = self->module_data;
      self->module_data = g_value_dup_boxed (value);
      break;
    case PROP_PARAMETERS:
      old_bytes = self->parameters;
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
  g_clear_pointer (&old_bytes, g_bytes_unref);
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
    case PROP_MODULE_DATA:
      g_value_set_boxed (value, self->module_data);
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
  g_clear_pointer (&self->module_data, g_bytes_unref);
  g_clear_pointer (&self->parameters, g_bytes_unref);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_cuda_kernel_load_module (GstCudaKernel * self, GstCudaContext * context)
{
  g_autofree gchar *image = NULL;
  g_autoptr (GError) error = NULL;
  gsize image_size;
  gconstpointer module_data;
  const gchar *location = self->module_location;
  CUmodule module = NULL;
  CUfunction function;
  CUresult result;
  const gchar *operation = "cuModuleLoadData";

  if (self->module_context == context)
    return TRUE;

  gst_cuda_kernel_clear_module (self);
  if (self->module_data && g_bytes_get_size (self->module_data)) {
    module_data = g_bytes_get_data (self->module_data, NULL);
    location = "(module-data)";
  } else if (!location || !*location) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("A CUDA module-location or module-data is required"), (NULL));
    return FALSE;
  } else {
    /* g_file_get_contents() adds the terminator required by PTX modules. */
    if (!g_file_get_contents (location, &image, &image_size, &error) ||
        image_size == 0) {
      GST_ELEMENT_ERROR (self, RESOURCE, READ, ("Could not read CUDA module"),
          ("%s: %s", location, error ? error->message : "file is empty"));
      return FALSE;
    }
    module_data = image;
  }

  if (!gst_cuda_context_push (context)) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("Could not activate CUDA context"), (NULL));
    return FALSE;
  }

  result = CuModuleLoadData (&module, module_data);
  if (result == CUDA_SUCCESS) {
    operation = "cuModuleGetFunction";
    result = CuModuleGetFunction (&function, module, KERNEL_ENTRY);
  }

  if (result != CUDA_SUCCESS) {
    if (module)
      CuModuleUnload (module);
    gst_cuda_context_pop (NULL);
    GST_ELEMENT_ERROR (self, LIBRARY, INIT, ("Could not initialize CUDA kernel"),
        ("%s in %s (%s) returned CUDA error %d", operation,
            location, KERNEL_ENTRY, result));
    return FALSE;
  }

  gst_cuda_context_pop (NULL);
  self->module_context = gst_object_ref (context);
  self->module = module;
  self->function = function;
  return TRUE;
}

static gboolean
gst_cuda_kernel_upload_parameters (GstCudaKernel * self, GBytes * bytes)
{
  gsize size = bytes ? g_bytes_get_size (bytes) : 0;
  CUdeviceptr device_parameters = 0;
  CUresult result = CUDA_SUCCESS;
  const gchar *operation = NULL;

  if (size > G_MAXUINT32) {
    GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
        ("CUDA kernel parameter block is too large"), (NULL));
    return FALSE;
  }
  if (!gst_cuda_context_push (self->module_context)) {
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("Could not activate CUDA context"), (NULL));
    return FALSE;
  }
  if (size) {
    operation = "cuMemAlloc";
    result = CuMemAlloc (&device_parameters, size);
    if (result == CUDA_SUCCESS) {
      CUDA_MEMCPY2D copy = { 0, };
      copy.srcMemoryType = CU_MEMORYTYPE_HOST;
      copy.srcHost = g_bytes_get_data (bytes, NULL);
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
    gst_cuda_context_pop (NULL);
    GST_ELEMENT_ERROR (self, LIBRARY, INIT,
        ("Could not upload CUDA kernel parameters"),
        ("%s returned CUDA error %d", operation, result));
    return FALSE;
  }

  if (self->device_parameters)
    CuMemFree (self->device_parameters);
  gst_cuda_context_pop (NULL);
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
  GstCudaKernel *self = GST_CUDA_KERNEL (transform);
  gboolean new_context = self->module_context != transform->context;
  gboolean geometry_changed = input->width != transform->in_info.width ||
      input->height != transform->in_info.height ||
      output->width != transform->out_info.width ||
      output->height != transform->out_info.height;
  g_autoptr (GBytes) parameters = NULL;

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

  if (!gst_cuda_kernel_load_module (self, transform->context))
    return FALSE;

  if ((new_context || geometry_changed) &&
      g_signal_has_handler_pending (self, prepare_parameters_signal, 0,
          FALSE)) {
    g_signal_emit (self, prepare_parameters_signal, 0,
        (guint) input->width, (guint) input->height,
        (guint) output->width, (guint) output->height, &parameters);
    if (!parameters) {
      GST_ELEMENT_ERROR (self, RESOURCE, SETTINGS,
          ("Could not prepare CUDA kernel parameters"), (NULL));
      goto error;
    }
    if (!gst_cuda_kernel_upload_parameters (self, parameters))
      goto error;
  } else if (new_context && self->parameters &&
      !gst_cuda_kernel_upload_parameters (self, self->parameters)) {
    goto error;
  }

  return TRUE;

error:
  gst_cuda_kernel_clear_module (self);
  return FALSE;
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
  g_object_class_install_property (object_class, PROP_MODULE_DATA,
      g_param_spec_boxed ("module-data", "Module data",
          "Trusted cubin or fatbin bytes, taking precedence over module-location",
          G_TYPE_BYTES, flags));
  g_object_class_install_property (object_class, PROP_PARAMETERS,
      g_param_spec_boxed ("parameters", "Parameters",
          "Read-only parameter bytes copied to CUDA memory", G_TYPE_BYTES,
          flags));
  g_object_class_install_property (object_class, PROP_SHARED_MEMORY,
      g_param_spec_uint ("shared-memory", "Shared memory",
          "Dynamic shared memory bytes per CUDA thread block", 0, G_MAXUINT, 0,
          flags));

  /**
   * GstCudaKernel::prepare-parameters:
   * @self: the CUDA kernel element
   * @input_width: width of the input frame
   * @input_height: height of the input frame
   * @output_width: width of the output frame
   * @output_height: height of the output frame
   *
   * Supplies parameters at caps negotiation when the geometry or CUDA
   * context changes. Connect one handler before starting the pipeline.
   * Its result takes precedence over the parameters property.
   *
   * Returns: (transfer full) (nullable): Parameter bytes or %NULL on failure
   */
  prepare_parameters_signal = g_signal_new ("prepare-parameters",
      G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST, 0,
      g_signal_accumulator_first_wins, NULL, NULL, G_TYPE_BYTES, 4,
      G_TYPE_UINT, G_TYPE_UINT, G_TYPE_UINT, G_TYPE_UINT);

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
