/* GStreamer CUDA IPC server lifecycle tests
 *
 * Copyright (C) 2026 Coram AI
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/check/gstcheck.h>
#include <glib/gstdio.h>
#include <gio/gunixsocketaddress.h>

#include "../../../sys/nvcodec/gstcudaipcserver_unix.h"

static GstCudaContext *context;

static guint
count_open_fds (void)
{
  GDir *dir = g_dir_open ("/proc/self/fd", 0, nullptr);
  guint count = 0;

  fail_unless (dir != nullptr);
  while (g_dir_read_name (dir))
    count++;
  g_dir_close (dir);

  return count;
}

static void
run_server (const gchar * path, gboolean connect_client)
{
  GstCudaIpcServer *server =
      gst_cuda_ipc_server_new (path, context, GST_CUDA_IPC_LEGACY);
  GSocketConnection *connection = nullptr;
  const gint64 deadline = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;

  fail_unless (server != nullptr);

  /* The listener is created asynchronously on the server's private context. */
  while (!g_file_test (path, G_FILE_TEST_EXISTS)
      && g_get_monotonic_time () < deadline)
    g_usleep (1000);
  fail_unless (g_file_test (path, G_FILE_TEST_EXISTS));

  if (connect_client) {
    GSocketClient *client = g_socket_client_new ();
    GSocketAddress *address = g_unix_socket_address_new (path);

    g_socket_client_set_timeout (client, 1);
    do {
      connection = g_socket_client_connect (client,
          G_SOCKET_CONNECTABLE (address), nullptr, nullptr);
      if (!connection)
        g_usleep (1000);
    } while (!connection && g_get_monotonic_time () < deadline);
    g_object_unref (address);
    g_object_unref (client);
    fail_unless (connection != nullptr);
  }

  /* With no frame data, stop must also close any accepted or queued client. */
  gst_cuda_ipc_server_stop (server);
  gst_object_unref (server);
  g_clear_object (&connection);
  fail_if (g_file_test (path, G_FILE_TEST_EXISTS));
}

GST_START_TEST (test_repeated_server_teardown)
{
  GError *error = nullptr;
  gchar *dir = g_dir_make_tmp ("gst-cudaipc-XXXXXX", &error);
  gchar *path;
  guint before;

  g_assert_no_error (error);
  fail_unless (dir != nullptr);
  path = g_build_filename (dir, "server.sock", nullptr);

  /* Prime GIO's one-time async infrastructure before measuring descriptors. */
  run_server (path, TRUE);
  before = count_open_fds ();
  for (guint i = 0; i < 100; i++)
    run_server (path, __i__ != 0);

  fail_unless_equals_int (count_open_fds (), before);
  fail_unless_equals_int (g_rmdir (dir), 0);
  g_free (path);
  g_free (dir);
}

GST_END_TEST;

static void
setup (void)
{
  context = gst_cuda_context_new (0);
  fail_unless (context != nullptr);
}

static void
teardown (void)
{
  gst_clear_object (&context);
}

static Suite *
cudaipc_suite (void)
{
  Suite *s = suite_create ("cudaipc");
  TCase *tc = tcase_create ("server teardown");
  GstCudaContext *probe;

  /* CUDA cannot be used in Check's forked test runner. */
  g_setenv ("CK_FORK", "no", TRUE);
  suite_add_tcase (s, tc);
  if (!gst_cuda_load_library ())
    return s;
  probe = gst_cuda_context_new (0);
  if (!probe)
    return s;
  gst_object_unref (probe);

  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_add_loop_test (tc, test_repeated_server_teardown, 0, 2);

  return s;
}

GST_CHECK_MAIN (cudaipc);
