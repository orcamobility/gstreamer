sources = [
  'gstnvipcbufferpool.c',
  'gstunixfd.c',
  'gstunixfdsink.c',
  'gstunixfdsrc.c',
]

cc = meson.get_compiler('c')

gst_dep = dependency('gstreamer-1.0')
gstbase_dep = dependency('gstreamer-base-1.0')
gstallocators_dep = dependency('gstreamer-allocators-1.0')
gstvideo_dep = dependency('gstreamer-video-1.0')
gio_dep = dependency('gio-2.0')
gio_unix_dep = dependency('gio-unix-2.0')
thread_dep = cc.find_library('pthread')

# Change this path if needed.
deepstream_lib_path = '/usr/lib/aarch64-linux-gnu/nvidia'
#deepstream_lib_path = '/opt/nvidia/deepstream/deepstream/lib'

nvbufsurface_lib = cc.find_library('nvbufsurface', dirs: deepstream_lib_path)
nvbufsurftransform_lib = cc.find_library('nvbufsurftransform', dirs: deepstream_lib_path)

nvcustomunixfd_plugin = library('gstnvcustomunixfd',
  sources,
  c_args: gst_plugins_bad_args + ['-fPIC', '-DHAVE_IPC_TARGET_NV'],
  include_directories: [configinc],
  dependencies : [gstbase_dep, gstallocators_dep, gstvideo_dep,
                  gio_dep, gio_unix_dep, nvbufsurface_lib, nvbufsurftransform_lib],
)

plugins += [nvcustomunixfd_plugin]
