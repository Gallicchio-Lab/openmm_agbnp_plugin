# OpenMM GaussVolSA plugin

A plugin that implements the Gaussian molecular volume and surface area model[1,2] in OpenMM.

Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>
Last Modified: July 2016


## License

This software is released under the LGPL license. See LICENSE.

## Credits

This software is written and maintained by Emilio Gallicchio <egallicchio@brooklyn.cuny.edu> with support from a grant from the National Science Foundation (ACI 1440665).

The plugin interface is based on the [openmmexampleplugin](https://github.com/peastman/openmmexampleplugin) by Peter Eastman.

## Installation

Locate the OpenMM installation directory, otherwise it will default to `/usr/local/openmm`.

Download the package from github:

```
git clone https://github.com/egallicc/openmm_gaussvol_plugin.git
```

Build and install the plugin with cmake. For exampple, assuming a unix system:
```
mkdir build_openmm_gaussvol_plugin
cd build_openmm_gaussvol_plugin
ccmake -i ../openmm_gaussvol_plugin
```

Hit `c` (configure) until all variables are correctly set, then `g` to generate the makefiles. `OPENMM_DIR` should point to an existing OpenMM installation. `CMAKE_INSTALL_PREFIX` normally is the same as `OPENMM_DIR`. The plugin requires the python API. You need `python` and `swig` to install it.

Once the configuration is done do:

```
make
make install
make PythonInstall
```

The last two steps may need superuser access depending on the installation target. It is recommended to to build the plugin under a `virtualenv` environment to install the python modules without superuser access.

## Test

`cd` to the directory where you cloned the `openmm_gaussvol_plugin` sources. Then:

```
cd example
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<openmm_dir>/lib:<openmm_dir>lib/plugins
python test_gaussvol.py
```

`<openmm_dir>` is the OpenMM installation directory. Again, the last step is best accomplished under the same `virtualenv` environment used to build the python modules.

## Relevant references:

1. Grant, J. A., and B. T. Pickup. "A Gaussian description of molecular shape." The Journal of Physical Chemistry 99.11 (1995): 3503-3510.
2. Gallicchio, Emilio, Kristina Paris, and Ronald M. Levy. "The AGBNP2 implicit solvation model." Journal of chemical theory and computation 5.9 (2009): 2544-2564.

