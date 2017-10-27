# OpenMM AGBNP plugin

A plugin that implements the AGBNP1 implicit solvent model [1] for OpenMM.

The plugin supports the OpenCL and Reference platforms. Implementation of the AGBNP2 model [2] is in progress.

Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>
Last Modified: October 2017


## License

This software is released under the LGPL license. See LICENSE.

## Credits

This software is written and maintained by Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>.

Support from the National Science Foundation (ACI 1440665) is acknowledged. 

The plugin interface is based on the [openmmexampleplugin](https://github.com/peastman/openmmexampleplugin) by Peter Eastman.

## Installation

These instructions assume Linux. Install OpenMM 7; the easiest is through `miniconda` using [these instructions](https://simtk.org/frs/download_start.php/file/4907/Conda%20installation%20instruction?group_id=161). Install `swig` through `conda` as well:

```
conda install -c omnia openmm swig
```

Locate the OpenMM installation directory, otherwise it will default to `/usr/local/openmm`. If OpenMM was installed via `conda` the OpenMM installation directory will be something like `$HOME/miniconda2/pkgs/openmm-7.0.1-py27_0`

Download this plugin package from github:

```
git clone https://github.com/egallicc/openmm_agbnp_plugin.git
```

Build and install the plugin with cmake. For example, assuming a unix system and a `conda` environment:
```
. ~/miniconda2/bin/activate
mkdir build_openmm_agbnp_plugin
cd build_openmm_agbnp_plugin
ccmake -i ../openmm_agbnp_plugin
```

Hit `c` (configure) until all variables are correctly set, then `g` to generate the makefiles. `OPENMM_DIR` should point to the OpenMM installation directory. `CMAKE_INSTALL_PREFIX` normally is the same as `OPENMM_DIR`. The plugin requires the python API. You need `python` and `swig` to install it.

Once the configuration is done do:

```
make
make install
make PythonInstall
```

The last two steps may need superuser access depending on the installation target, or use the recommended `conda` environment.

## Test

`cd` to the directory where you cloned the `openmm_agbnp_plugin` sources. Then:

```
cd example
export OPENMM_PLUGIN_DIR=<openmm_dir>/lib/plugins
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<openmm_dir>/lib:<openmm_dir>lib/plugins
python test_agbnp.py
```

where `<openmm_dir>` is the OpenMM installation directory.

## Relevant references:

1. Gallicchio E., and R.M. Levy. AGBNP, an analytic implicit solvent model suitable for molecular dynamics simulations and high-resolution modeling, J. Comp. Chem. 25, 479-499 (2004).
2. Gallicchio, Emilio, Kristina Paris, and Ronald M. Levy. "The AGBNP2 implicit solvation model." Journal of chemical theory and computation 5.9 (2009): 2544-2564.
3. Baofeng Zhang, Denise Kilburg, Peter Eastman, Vijay S. Pande, Emilio Gallicchio. Efficient Gaussian Density Formulation of Volume and Surface Areas of Macromolecules on Graphical Processing Units. J. Comp. Chem., 38, 740-752 (2017).
