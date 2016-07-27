from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
GVolplugin_header_dir = '@GVolPLUGIN_HEADER_DIR@'
GVolplugin_library_dir = '@GVolPLUGIN_LIBRARY_DIR@'
GVol_dir = '@GVol_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_GVolplugin',
                      sources=['GVolPluginWrapper.cpp'],
                      libraries=['OpenMM', 'GVolPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), GVolplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), GVolplugin_library_dir, GVol_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='GVolplugin',
      version='1.0',
      py_modules=['GVolplugin'],
      ext_modules=[extension],
     )
