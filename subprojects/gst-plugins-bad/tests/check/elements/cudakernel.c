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
make_input (guint frame)
{
  GstVideoInfo info;
  GstVideoFrame map;
  GstBuffer *buffer;
  guint plane, x, y;

  gst_video_info_set_format (&info, GST_VIDEO_FORMAT_NV12, 68, 36);
  buffer = gst_buffer_new_allocate (NULL, info.size, NULL);
  fail_unless (gst_video_frame_map (&map, &info, buffer, GST_MAP_WRITE));
  for (plane = 0; plane < 2; plane++) {
    guint8 *data = GST_VIDEO_FRAME_PLANE_DATA (&map, plane);
    gint pitch = GST_VIDEO_FRAME_PLANE_STRIDE (&map, plane);
    for (y = 0; y < (36u >> plane); y++)
      for (x = 0; x < 68; x++)
        data[y * pitch + x] = (x * 11 + y * 53 + plane * 127 + frame * 23) % 256;
  }
  gst_video_frame_unmap (&map);
  GST_BUFFER_PTS (buffer) = frame * GST_SECOND / 6;
  GST_BUFFER_DURATION (buffer) = GST_SECOND / 6;
  return buffer;
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
  } else if (!gst_cuda_nvrtc_load_library ()) {
    gst_println ("Skip Lanczos tests: NVRTC unavailable");
  } else {
    tcase_add_test (tc, test_lanczos_orientation);
    tcase_add_test (tc, test_lanczos_default_geometry);
  }
  if (kernel)
    gst_object_unref (kernel);
  return s;
}

GST_CHECK_MAIN (cudakernel);
