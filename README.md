# OpenMM AGBNP3 Plugin

A plugin to add the AGBNP3 Force to OpenMM

Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>
Last Modified: June 2015



## License

This software is released under the LGPL license. See LICENSE.

## Credits

This software is written and maintained by Emilio Gallicchio <egallicchio@brooklyn.cuny.edu> with support from a grant from the National Science Foundation (ACI 1440665).

The plugin interface is based on the [openmmexampleplugin](https://github.com/peastman/openmmexampleplugin) by Peter Eastman.

## Installation

Locate the OpenMM installation directory, otherwise it will default to `/usr/local/openmm`.

Download the package from github:

```
git clone --recursive https://github.com/egallicc/openmm_agbnp3_plugin.git
```
The `--recursive` option automatically downloads the AGBNP3 submodule.

Build and install the plugin with cmake. Assuming a unix system:

```
mkdir build_openmm_agbnp3_plugin
cd build_openmm_agbnp3_plugin
ccmake -i ../openmm_agbnp3_plugin
```

Hit `c` (configure) until all variables are correctly set, then `g` to generate the makefiles. `OPENMM_DIR` should point to an existing OpenMM installation. `CMAKE_INSTALL_PREFIX` normally is the same as `OPENMM_DIR`. The AGBNP3 plugin requires the python API. You need `python` and `swig` to install it.

Once the configuration is done do:

```
make
make install
make PythonInstall
```

The last two steps may need superuser access depending on the installation target. It is recommended to to build the plugin under a `virtualenv` environment to install the python modules without superuser access.

## Test

`cd` to the directory where you cloned the `openmm_agbnp3_plugin` sources. Then:

```
cd example
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<openmm_dir>/lib:<openmm_dir>lib/plugins
python test_agbnp3.py
```

`<openmm_dir>` is the OpenMM installation directory. Again, the last step is best accomplished under the same `virtualenv` environment used to build the python modules.

