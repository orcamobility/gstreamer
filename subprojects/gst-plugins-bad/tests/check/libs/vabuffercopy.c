/* GStreamer
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <gst/check/gstcheck.h>
#include "../../../sys/va/gstvabuffercopy.h"

#ifdef G_OS_UNIX
#include <sys/mman.h>
#include <unistd.h>
#endif

GST_START_TEST (test_copy_alignment)
{
  guint8 src[2048], dst[2048], expected[2048];
  const gsize sizes[] = { 0, 1, 15, 16, 17, 63, 64, 65, 127, 128,
    129, 191, 192, 193, 255, 256, 257, 1023, 1024, 1025
  };
  guint i, s, d, n;

  for (i = 0; i < sizeof (src); i++)
    src[i] = (i * 37 + i / 256) & 255;

  for (s = 0; s < 64; s++) {
    for (d = 0; d < 64; d++) {
      for (n = 0; n < G_N_ELEMENTS (sizes); n++) {
        gsize size = sizes[n];

        memset (dst, 0xa5, sizeof (dst));
        memset (expected, 0xa5, sizeof (expected));
        memcpy (expected + 64 + d, src + s, size);
        gst_va_buffer_copy (dst + 64 + d, src + s, size);
        fail_unless (memcmp (dst, expected, sizeof (dst)) == 0);
      }
    }
  }
}
GST_END_TEST;

#ifdef G_OS_UNIX
GST_START_TEST (test_copy_mapping_boundary)
{
  gsize page_size = sysconf (_SC_PAGESIZE);
  guint8 *src, *dst;
  gsize size;

  src = mmap (NULL, 3 * page_size, PROT_NONE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  dst = mmap (NULL, 3 * page_size, PROT_NONE,
      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  fail_unless (src != MAP_FAILED && dst != MAP_FAILED);
  fail_unless (mprotect (src + page_size, page_size,
          PROT_READ | PROT_WRITE) == 0);
  fail_unless (mprotect (dst + page_size, page_size,
          PROT_READ | PROT_WRITE) == 0);
  memset (src + page_size, 0x5a, page_size);
  fail_unless (mprotect (src + page_size, page_size, PROT_READ) == 0);

  for (size = 1; size <= 1024; size++) {
    guint8 *s = src + 2 * page_size - size;
    guint8 *d = dst + 2 * page_size - size;

    gst_va_buffer_copy (d, s, size);
    fail_unless (memcmp (d, s, size) == 0);
    gst_va_buffer_copy (dst + page_size, src + page_size, size);
    fail_unless (memcmp (dst + page_size, src + page_size, size) == 0);
  }

  munmap (src, 3 * page_size);
  munmap (dst, 3 * page_size);
}
GST_END_TEST;
#endif

static Suite *
va_buffer_copy_suite (void)
{
  Suite *s = suite_create ("VA buffer copy");
  TCase *tc = tcase_create ("copy");

  suite_add_tcase (s, tc);
  tcase_add_test (tc, test_copy_alignment);
#ifdef G_OS_UNIX
  tcase_add_test (tc, test_copy_mapping_boundary);
#endif
  return s;
}

GST_CHECK_MAIN (va_buffer_copy);
