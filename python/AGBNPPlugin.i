%module AGBNPplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"


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
#include "AGBNPForce.h"
#include "OpenMM.h" 
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}



%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}



/*
 * Add units to function outputs.
*/
%pythonappend AGBNPPlugin::AGBNPForce::getParticleParameters(int index, double& radius, double& gamma, 
							   bool& ishydrogen) const %{
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.kilojoule_per_mole / (unit.nanometer * unit.nanometer))
%}


namespace AGBNPPlugin {

class AGBNPForce : public OpenMM::Force {
public:

    AGBNPForce();

    int getNumParticles() const;

    void addParticle(double radius, double gamma, bool ishydrogen);

    void updateParametersInContext(OpenMM::Context& context);

    void setCutoffDistance(double distance);

    enum NonbondedMethod {
      NoCutoff =  0,
      CutoffNonPeriodic =  1,
      CutoffPeriodic =  2
    };

    void setNonbondedMethod(NonbondedMethod method);

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

