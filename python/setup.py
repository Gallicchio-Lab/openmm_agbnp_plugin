from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
AGBNPplugin_header_dir = '@AGBNPPLUGIN_HEADER_DIR@'
AGBNPplugin_library_dir = '@AGBNPPLUGIN_LIBRARY_DIR@'
AGBNP_dir = '@AGBNP_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_AGBNPplugin',
                      sources=['AGBNPPluginWrapper.cpp'],
                      libraries=['OpenMM', 'AGBNPPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), AGBNPplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), AGBNPplugin_library_dir, AGBNP_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='AGBNPplugin',
      version='1.0',
      py_modules=['AGBNPplugin'],
      ext_modules=[extension],
     )
