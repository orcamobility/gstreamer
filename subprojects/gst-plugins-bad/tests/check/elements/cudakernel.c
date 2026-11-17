/* SPDX-License-Identifier: LGPL-2.1-or-later */

#include <gst/check/gstcheck.h>
#include <gst/check/gstharness.h>
#include <gst/cuda/gstcudanvrtc.h>
#include <gst/video/video.h>
#include <string.h>

static GstHarness *
make_harness (guint width, guint height, guint direction, gboolean reference)
{
  gboolean swap = direction == GST_VIDEO_ORIENTATION_90R ||
      direction == GST_VIDEO_ORIENTATION_90L ||
      direction == GST_VIDEO_ORIENTATION_UL_LR ||
      direction == GST_VIDEO_ORIENTATION_UR_LL;
  gchar *pipeline = g_strdup_printf (
      "cudaupload ! cudakernel preset=lanczos video-direction=%u ! "
      "capsfilter name=size caps=\"video/x-raw(memory:CUDAMemory),"
      "width=%u,height=%u,pixel-aspect-ratio=1/1\" ! cudadownload %s",
      reference ? 0 : direction,
      !reference && swap ? height : width,
      !reference && swap ? width : height,
      reference ? "! videoflip name=flip" : "");
  if (width == 0) {
    g_free (pipeline);
    pipeline = g_strdup_printf (
        "cudaupload ! cudakernel preset=lanczos video-direction=%u ! "
        "cudadownload %s", reference ? 0 : direction,
        reference ? "! videoflip name=flip" : "");
  }
  GstHarness *h = gst_harness_new_parse (pipeline);
  g_free (pipeline);
  if (reference) {
    GstElement *flip = gst_bin_get_by_name (GST_BIN (h->element), "flip");
    g_object_set (flip, "video-direction", direction, NULL);
    gst_object_unref (flip);
  }
  gst_harness_set_src_caps_str (h,
      "video/x-raw,format=NV12,width=68,height=36,"
      "interlace-mode=progressive,framerate=6/1,pixel-aspect-ratio=1/1,"
      "colorimetry=bt709");
  return h;
}

static GstBuffer *
make_input_size (guint width, guint height, guint frame)
{
  GstVideoInfo info;
  GstVideoFrame map;
  GstBuffer *buffer;
  guint plane, x, y;

  gst_video_info_set_format (&info, GST_VIDEO_FORMAT_NV12, width, height);
  buffer = gst_buffer_new_allocate (NULL, info.size, NULL);
  fail_unless (gst_video_frame_map (&map, &info, buffer, GST_MAP_WRITE));
  for (plane = 0; plane < 2; plane++) {
    guint8 *data = GST_VIDEO_FRAME_PLANE_DATA (&map, plane);
    gint pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&map, plane);
    for (y = 0; y < (height >> plane); y++)
      for (x = 0; x < width; x++)
        data[y * pitch + x] = (x * 11 + y * 53 + plane * 127 + frame * 23) % 256;
  }
  gst_video_frame_unmap (&map);
  GST_BUFFER_PTS (buffer) = frame * GST_SECOND / 6;
  GST_BUFFER_DURATION (buffer) = GST_SECOND / 6;
  return buffer;
}

static GstBuffer *
make_input (guint frame)
{
  return make_input_size (68, 36, frame);
}

static void
compare_frames (GstHarness * actual, GstHarness * reference, guint frame)
{
  GstBuffer *input = make_input (frame);
  GstBuffer *a = gst_harness_push_and_pull (actual, gst_buffer_ref (input));
  GstBuffer *b = gst_harness_push_and_pull (reference, input);
  GstCaps *acaps = gst_pad_get_current_caps (actual->sinkpad);
  GstCaps *bcaps = gst_pad_get_current_caps (reference->sinkpad);
  GstVideoInfo ai, bi;
  GstVideoFrame am, bm;
  guint plane, y;

  fail_unless (a != NULL && b != NULL);
  fail_unless (gst_video_info_from_caps (&ai, acaps));
  fail_unless (gst_video_info_from_caps (&bi, bcaps));
  fail_unless_equals_int (ai.width, bi.width);
  fail_unless_equals_int (ai.height, bi.height);
  fail_unless_equals_int (ai.par_n, bi.par_n);
  fail_unless_equals_int (ai.par_d, bi.par_d);
  fail_unless_equals_uint64 (GST_BUFFER_PTS (a), GST_BUFFER_PTS (b));
  fail_unless_equals_uint64 (GST_BUFFER_DURATION (a), GST_BUFFER_DURATION (b));
  fail_unless (gst_video_frame_map (&am, &ai, a, GST_MAP_READ));
  fail_unless (gst_video_frame_map (&bm, &bi, b, GST_MAP_READ));
  for (plane = 0; plane < 2; plane++) {
    const guint8 *ap = GST_VIDEO_FRAME_PLANE_DATA (&am, plane);
    const guint8 *bp = GST_VIDEO_FRAME_PLANE_DATA (&bm, plane);
    for (y = 0; y < ((guint) ai.height >> plane); y++) {
      fail_unless (memcmp (ap, bp, ai.width) == 0,
          "frame %u plane %u row %u differs", frame, plane, y);
      ap += GST_VIDEO_FRAME_PLANE_STRIDE (&am, plane);
      bp += GST_VIDEO_FRAME_PLANE_STRIDE (&bm, plane);
    }
  }
  gst_video_frame_unmap (&am);
  gst_video_frame_unmap (&bm);
  gst_caps_unref (acaps);
  gst_caps_unref (bcaps);
  gst_buffer_unref (a);
  gst_buffer_unref (b);
}

GST_START_TEST (test_lanczos_orientation)
{
  const guint sizes[][2] = { {44, 26}, {68, 36}, {68, 26}, {30, 18}, {44, 26} };
  guint direction, size, frame;

  for (direction = 0; direction <= GST_VIDEO_ORIENTATION_UR_LL; direction++) {
    GstHarness *actual = make_harness (44, 26, direction, FALSE);
    GstHarness *reference = make_harness (44, 26, direction, TRUE);
    GstElement *actual_size =
        gst_bin_get_by_name (GST_BIN (actual->element), "size");
    GstElement *reference_size =
        gst_bin_get_by_name (GST_BIN (reference->element), "size");
    gboolean swap = direction == 1 || direction == 3 ||
        direction == 6 || direction == 7;

    for (size = 0; size < G_N_ELEMENTS (sizes); size++) {
      GstCaps *caps = gst_caps_new_simple ("video/x-raw",
          "width", G_TYPE_INT, sizes[size][0],
          "height", G_TYPE_INT, sizes[size][1],
          "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL);
      gst_caps_set_features (caps, 0,
          gst_caps_features_new ("memory:CUDAMemory", NULL));
      g_object_set (reference_size, "caps", caps, NULL);
      caps = gst_caps_make_writable (caps);
      if (swap)
        gst_structure_set (gst_caps_get_structure (caps, 0),
            "width", G_TYPE_INT, sizes[size][1],
            "height", G_TYPE_INT, sizes[size][0], NULL);
      g_object_set (actual_size, "caps", caps, NULL);
      gst_caps_unref (caps);
      for (frame = 0; frame < 3; frame++)
        compare_frames (actual, reference, size * 3 + frame);
    }
    gst_object_unref (actual_size);
    gst_object_unref (reference_size);
    gst_harness_teardown (actual);
    gst_harness_teardown (reference);
  }
}

GST_END_TEST;

GST_START_TEST (test_lanczos_ready_direction)
{
  GstHarness *actual = make_harness (44, 26, 0, FALSE);
  GstHarness *reference = make_harness (44, 26, 0, TRUE);
  GstElement *kernel = gst_harness_find_element (actual, "cudakernel");
  GstElement *size = gst_bin_get_by_name (GST_BIN (actual->element), "size");
  GstElement *flip = gst_bin_get_by_name (GST_BIN (reference->element), "flip");
  GstCaps *caps = gst_caps_from_string (
      "video/x-raw(memory:CUDAMemory),width=26,height=44,pixel-aspect-ratio=1/1");

  compare_frames (actual, reference, 0);
  fail_unless (gst_element_set_state (actual->element, GST_STATE_READY) !=
      GST_STATE_CHANGE_FAILURE);
  fail_unless (gst_element_set_state (reference->element, GST_STATE_READY) !=
      GST_STATE_CHANGE_FAILURE);
  g_object_set (kernel, "video-direction", GST_VIDEO_ORIENTATION_90R, NULL);
  g_object_set (flip, "video-direction", GST_VIDEO_ORIENTATION_90R, NULL);
  g_object_set (size, "caps", caps, NULL);
  gst_harness_play (actual);
  gst_harness_play (reference);
  compare_frames (actual, reference, 1);
  gst_caps_unref (caps);
  gst_object_unref (kernel);
  gst_object_unref (size);
  gst_object_unref (flip);
  gst_harness_teardown (actual);
  gst_harness_teardown (reference);
}

GST_END_TEST;

/* This oracle does not use the device parameters or the CUDA implementation.
 * It performs two separable passes in double precision with the public
 * resampler tables, quantizing only the coefficients to the GPU's FP32 input.
 * One code of tolerance allows FP32 accumulation and final rounding error. */
static void
check_numeric_output (GstBuffer * input, const GstVideoInfo * input_info,
    GstBuffer * output, const GstVideoInfo * output_info, gboolean chunked)
{
  GstVideoFrame in, out;
  guint plane;

  fail_unless (gst_video_frame_map (&in, input_info, input, GST_MAP_READ));
  fail_unless (gst_video_frame_map (&out, output_info, output, GST_MAP_READ));
  for (plane = 0; plane < 2; plane++) {
    GstVideoResampler horizontal = { 0 }, vertical = { 0 };
    GstVideoResamplerMethod method = plane == 0 ?
        GST_VIDEO_RESAMPLER_METHOD_LANCZOS : GST_VIDEO_RESAMPLER_METHOD_LINEAR;
    guint iw = input_info->width >> plane, ih = input_info->height >> plane;
    guint ow = output_info->width >> plane, oh = output_info->height >> plane;
    guint channels = plane + 1, x, y, c, k;
    const guint8 *src = GST_VIDEO_FRAME_PLANE_DATA (&in, plane);
    const guint8 *dst = GST_VIDEO_FRAME_PLANE_DATA (&out, plane);
    gint src_pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&in, plane);
    gint dst_pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&out, plane);
    gdouble *scratch = g_new (gdouble, ow * channels * ih);

    fail_unless (gst_video_resampler_init (&horizontal, method,
            GST_VIDEO_RESAMPLER_FLAG_NONE, ow, 0, 0.0, iw, ow, NULL));
    fail_unless (gst_video_resampler_init (&vertical, method,
            GST_VIDEO_RESAMPLER_FLAG_NONE, oh, 0, 0.0, ih, oh, NULL));
    if (chunked) {
      guint first = G_MAXUINT, last = 0;
      for (y = 0; y < MIN (oh, 8); y++) {
        first = MIN (first, vertical.offset[y]);
        last = MAX (last, vertical.offset[y] + vertical.max_taps);
      }
      fail_unless (last - first > 128);
    }
    for (y = 0; y < ih; y++) {
      for (x = 0; x < ow; x++) {
        for (c = 0; c < channels; c++) {
          gdouble sum = 0.0;
          for (k = 0; k < horizontal.max_taps; k++)
            sum += src[y * src_pitch +
                (horizontal.offset[x] + k) * channels + c] *
                (gdouble) (gfloat) horizontal.taps[x * horizontal.max_taps + k];
          scratch[(y * ow + x) * channels + c] = sum;
        }
      }
    }
    for (y = 0; y < oh; y++) {
      for (x = 0; x < ow; x++) {
        for (c = 0; c < channels; c++) {
          gdouble sum = 0.0;
          gint expected, actual = dst[y * dst_pitch + x * channels + c];
          for (k = 0; k < vertical.max_taps; k++)
            sum += scratch[((vertical.offset[y] + k) * ow + x) * channels + c] *
                (gdouble) (gfloat) vertical.taps[y * vertical.max_taps + k];
          expected = CLAMP ((gint) (sum + 0.5), 0, 255);
          fail_unless (ABS (actual - expected) <= 1,
              "plane %u pixel %u,%u channel %u: got %d, oracle %d",
              plane, x, y, c, actual, expected);
        }
      }
    }
    g_free (scratch);
    gst_video_resampler_clear (&horizontal);
    gst_video_resampler_clear (&vertical);
  }
  gst_video_frame_unmap (&in);
  gst_video_frame_unmap (&out);
}

GST_START_TEST (test_lanczos_numeric)
{
  const guint sizes[][4] = {
    {68, 36, 44, 26}, {68, 36, 30, 18}, {68, 36, 68, 26},
    {64, 1080, 64, 8}
  };
  guint size, frame;

  for (size = 0; size < G_N_ELEMENTS (sizes); size++) {
    guint iw = sizes[size][0], ih = sizes[size][1];
    guint ow = sizes[size][2], oh = sizes[size][3];
    gchar *pipeline = g_strdup_printf (
        "cudaupload ! cudakernel preset=lanczos ! "
        "video/x-raw(memory:CUDAMemory),width=%u,height=%u ! cudadownload", ow, oh);
    gchar *caps = g_strdup_printf (
        "video/x-raw,format=NV12,width=%u,height=%u,interlace-mode=progressive,"
        "framerate=6/1,colorimetry=bt709,pixel-aspect-ratio=1/1", iw, ih);
    GstHarness *h = gst_harness_new_parse (pipeline);
    GstVideoInfo input_info;

    gst_video_info_set_format (&input_info, GST_VIDEO_FORMAT_NV12, iw, ih);
    gst_harness_set_src_caps_str (h, caps);
    for (frame = 0; frame < 3; frame++) {
      GstBuffer *input = make_input_size (iw, ih, frame);
      GstBuffer *output = gst_harness_push_and_pull (h, gst_buffer_ref (input));
      GstCaps *output_caps = gst_pad_get_current_caps (h->sinkpad);
      GstVideoInfo output_info;

      fail_unless (output != NULL);
      fail_unless (gst_video_info_from_caps (&output_info, output_caps));
      fail_unless_equals_int (output_info.width, ow);
      fail_unless_equals_int (output_info.height, oh);
      check_numeric_output (input, &input_info, output, &output_info, ih == 1080);
      gst_caps_unref (output_caps);
      gst_buffer_unref (input);
      gst_buffer_unref (output);
    }
    gst_harness_teardown (h);
    g_free (pipeline);
    g_free (caps);
  }
}

GST_END_TEST;

GST_START_TEST (test_invalid_settings)
{
  const gchar *settings[] = {
    "preset=custom video-direction=1",
    "preset=lanczos video-direction=auto",
    "preset=lanczos video-direction=custom",
    "preset=custom shared-memory=49153"
  };
  const gchar *errors[] = {
    "video-direction requires preset=lanczos",
    "Only fixed video directions are supported",
    "Only fixed video directions are supported",
    "shared-memory must not exceed 48 KiB"
  };
  guint i;

  for (i = 0; i < G_N_ELEMENTS (settings); i++) {
    gchar *description = g_strdup_printf ("cudakernel %s ! fakesink", settings[i]);
    GError *parse_error = NULL, *error = NULL;
    GstElement *pipeline = gst_parse_launch (description, &parse_error);
    GstBus *bus;
    GstMessage *message;

    fail_unless (pipeline != NULL);
    fail_unless (parse_error == NULL);
    bus = gst_element_get_bus (pipeline);
    fail_unless_equals_int (gst_element_set_state (pipeline, GST_STATE_PAUSED),
        GST_STATE_CHANGE_FAILURE);
    message = gst_bus_timed_pop_filtered (bus, GST_SECOND, GST_MESSAGE_ERROR);
    fail_unless (message != NULL);
    gst_message_parse_error (message, &error, NULL);
    fail_unless (error->domain == GST_RESOURCE_ERROR);
    fail_unless_equals_int (error->code, GST_RESOURCE_ERROR_SETTINGS);
    fail_unless (strstr (error->message, errors[i]) != NULL);
    g_clear_error (&error);
    gst_message_unref (message);
    gst_object_unref (bus);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);
    g_free (description);
  }
}

GST_END_TEST;

GST_START_TEST (test_lanczos_default_geometry)
{
  guint direction;

  for (direction = 0; direction <= GST_VIDEO_ORIENTATION_UR_LL; direction++) {
    GstHarness *actual = make_harness (0, 0, direction, FALSE);
    GstHarness *reference = make_harness (0, 0, direction, TRUE);
    const gchar *caps = "video/x-raw,format=NV12,width=68,height=36,"
        "interlace-mode=progressive,framerate=6/1,pixel-aspect-ratio=2/3,"
        "colorimetry=bt709";
    gst_harness_set_src_caps_str (actual, caps);
    gst_harness_set_src_caps_str (reference, caps);
    compare_frames (actual, reference, 0);
    gst_harness_teardown (actual);
    gst_harness_teardown (reference);
  }
}

GST_END_TEST;

static Suite *
cudakernel_suite (void)
{
  Suite *s = suite_create ("cudakernel");
  TCase *tc = tcase_create ("orientation");
  GstElement *kernel = gst_element_factory_make ("cudakernel", NULL);

  g_setenv ("CK_FORK", "no", TRUE);
  suite_add_tcase (s, tc);
  tcase_set_timeout (tc, 120);
  if (!kernel) {
    gst_println ("Skip cudakernel test: CUDA element unavailable");
  } else {
    tcase_add_test (tc, test_invalid_settings);
    if (gst_cuda_nvrtc_load_library ()) {
      tcase_add_test (tc, test_lanczos_orientation);
      tcase_add_test (tc, test_lanczos_default_geometry);
      tcase_add_test (tc, test_lanczos_numeric);
      tcase_add_test (tc, test_lanczos_ready_direction);
    } else {
      gst_println ("Skip Lanczos tests: NVRTC unavailable");
    }
  }
  if (kernel)
    gst_object_unref (kernel);
  return s;
}

GST_CHECK_MAIN (cudakernel);
