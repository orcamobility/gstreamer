/* GStreamer
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GST_VA_BUFFER_COPY_H__
#define __GST_VA_BUFFER_COPY_H__

#include <glib.h>
#include <string.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <smmintrin.h>

/* Read complete cache lines with streaming loads. Ordinary loads from a
 * write-combined GPU mapping can be much slower than reads from system RAM.
 * The destination is consumed by the CPU, so keep ordinary cached stores. */
__attribute__((target("sse4.1")))
static void
gst_va_buffer_copy_sse41 (guint8 * dst, const guint8 * src, gsize size)
{
  gsize head = MIN (size, (64 - ((guintptr) src & 63)) & 63);

  if (head) {
    memcpy (dst, src, head);
    dst += head;
    src += head;
    size -= head;
  }

  _mm_mfence ();
  while (size >= 64) {
    __m128i a = _mm_stream_load_si128 ((__m128i *) src);
    __m128i b = _mm_stream_load_si128 ((__m128i *) (src + 16));
    __m128i c = _mm_stream_load_si128 ((__m128i *) (src + 32));
    __m128i d = _mm_stream_load_si128 ((__m128i *) (src + 48));

    _mm_storeu_si128 ((__m128i *) dst, a);
    _mm_storeu_si128 ((__m128i *) (dst + 16), b);
    _mm_storeu_si128 ((__m128i *) (dst + 32), c);
    _mm_storeu_si128 ((__m128i *) (dst + 48), d);
    src += 64;
    dst += 64;
    size -= 64;
  }

  if (size)
    memcpy (dst, src, size);
}
#endif

/* The caller must synchronize GPU writes before copying the mapped buffer. */
static inline void
gst_va_buffer_copy (gpointer dst, gconstpointer src, gsize size)
{
  if (!size)
    return;

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
  if (size >= 128 && __builtin_cpu_supports ("sse4.1")) {
    gst_va_buffer_copy_sse41 (dst, src, size);
    return;
  }
#endif

  memcpy (dst, src, size);
}

#endif /* __GST_VA_BUFFER_COPY_H__ */
