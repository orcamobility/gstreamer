# project('gst-nvunixfd_src', 'c',
#   version : '0.0.1',
#   meson_version : '>= 1.1',
#   default_options : [ 'warning_level=1',
#                       'buildtype=debugoptimized' ])

unixfd_sources = [
  'gst-nvunixfd/gstnvipcbufferpool.c',
  'gst-nvunixfd/gstunixfd.c',
  'gst-nvunixfd/gstunixfdsink.c',
  'gst-nvunixfd/gstunixfdsrc.c',
]

cc = meson.get_compiler('c')
add_project_link_arguments('-Wl,--no-undefined', language : 'c')

gst_dep = dependency('gstreamer-1.0')
gstbase_dep = dependency('gstreamer-base-1.0')
gstallocators_dep = dependency('gstreamer-allocators-1.0')
gstvideo_dep = dependency('gstreamer-video-1.0')
gio_dep = dependency('gio-2.0')
gio_unix_dep = dependency('gio-unix-2.0')
thread_dep = cc.find_library('pthread')

# Change this path if needed.
deepstream_lib_path = '/usr/lib/aarch64-linux-gnu/nvidia'
nvbufsurface_lib = cc.find_library('nvbufsurface', dirs: deepstream_lib_path)
nvbufsurftransform_lib = cc.find_library('nvbufsurftransform', dirs: deepstream_lib_path)

gstunixfd_plugin = library('gstnvunixfd',
  unixfd_sources,
  c_args: ['-fPIC', '-DHAVE_IPC_TARGET_NV'],
  include_directories: include_directories('.'),
  dependencies : [gst_dep, gstbase_dep, gstallocators_dep, gstvideo_dep,
                  gio_dep, gio_unix_dep, nvbufsurface_lib, nvbufsurftransform_lib],
)

