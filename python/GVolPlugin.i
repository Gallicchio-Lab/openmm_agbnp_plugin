%module GVolplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"


/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include  "GVolForce.h"
#include "OpenMM.h" 
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}



/*
 * The code below strips all units before the wrapper
 * functions are called. This code also converts numpy
 * arrays to lists.
*/

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}


/* strip the units off of all input arguments */
%pythonprepend %{
try:
    args=mm.stripUnits(args)
except UnboundLocalError:
    pass
%}


/*
 * Add units to function outputs.
*/
%pythonappend GVolPlugin::GVolForce::getParticleParameters(int index, double& radius, double& gamma, 
							   bool& ishydrogen) const %{
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.kilojoule_per_mole / (unit.nanometer * unit.nanometer))
%}


namespace GVolPlugin {

class GVolForce : public OpenMM::Force {
public:
    GVolForce();

    int getNumParticles() const;

    void addParticle(double radius, double gamma, bool ishydrogen);

    void updateParametersInContext(OpenMM::Context& context);

    /*
     * The reference parameters to this function are output values.
     * Marking them as such will cause swig to return a tuple.
    */
    %apply double& OUTPUT {double& radius};
    %apply double& OUTPUT {double& gamma};
    %apply bool& OUTPUT {bool& ishydrogen};
    void getParticleParameters(int index, double& radius, double& gamma, bool& ishydrogen) const; 
};

}

