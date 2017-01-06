/* -------------------------------------------------------------------------- *
 *                               OpenMM-AGBNP                                *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include "ReferenceAGBNPKernels.h"
#include "AGBNPForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "gaussvol.h"


using namespace AGBNPPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

// Initializes AGBNP library
void ReferenceCalcAGBNPForceKernel::initialize(const System& system, const AGBNPForce& force) {
    
    numParticles = force.getNumParticles();

    cout << "In initialize ..." << numParticles <<  endl;

    //input lists
    positions.resize(numParticles);
    radii.resize(numParticles);
    gammas.resize(numParticles);
    ishydrogen.resize(numParticles);

    //output lists
    free_volume.resize(numParticles);
    self_volume.resize(numParticles);
    surface_areas.resize(numParticles);
    surf_force.resize(numParticles);
    
    double ang2nm = 0.1;
    for (int i = 0; i < numParticles; i++){
      double r, g;
      bool h;
      force.getParticleParameters(i, r, g, h);
      radii[i] = r;
      gammas[i] = g;
      ishydrogen[i] = h;
    }

    cout << "Calling new GaussVol ..." << endl;

    //create and saves AGBNP instance
    gvol = new GaussVol(numParticles, radii, gammas, ishydrogen);
    //gvol = new GaussVol();

    cout << "Done" << endl;
}

double ReferenceCalcAGBNPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    bool verbose = true;
    int init = 0;

    //for (int i = 0; i < numParticles; i++){
    //  positions[i] = pos[i];
    //}

    // Compute the interactions.
    gvol->enerforc(init, pos, surf_energy, surf_force, free_volume, self_volume, surface_areas);

    //returns the gradients
    for(int i = 0; i < numParticles; i++){
      force[i] += surf_force[i];
    }

#ifdef NOTNOW
    vector<int> nov, nov_2body;
    gvol->getstat(nov, nov_2body);    
    int nn = 0, nn2 = 0;
    for(int i = 0; i < nov.size(); i++){
      nn += nov[i];
     nn2 += nov_2body[i];
    }
    cout << "Noverlaps: " << nn << " " << nn2 << endl;
#endif

    if(verbose){
      //to input to freesasa
      double nm2ang = 10.0;
      cout << "id x y z radius: (Ang)" << endl;
      for(int i = 0; i < numParticles; i++){
	double rad = radii[i];
	if(ishydrogen[i] == 1) rad = 0.0;
	cout << "xyzr: " << i << " " << nm2ang*pos[i][0] << " " << nm2ang*pos[i][1] << " " << nm2ang*pos[i][2] << " " << nm2ang*rad << endl;
      }
    }


    if(verbose){
      cout << "Surface areas:" << endl;
      double tot_surf_area = 0.0;
      for(int i = 0; i < numParticles; i++){
	cout << "SA " << i << " " << surface_areas[i] << endl;
	tot_surf_area += surface_areas[i];
      }
      cout << "Total surface area: (Ang^2) " << 100.0*tot_surf_area << endl;
      cout << "Energy: " << surf_energy << endl;
    }

    //returns energy
    return (double)surf_energy;
}

void ReferenceCalcAGBNPForceKernel::copyParametersToContext(ContextImpl& context, const AGBNPForce& force) {
  std::vector<int> neighbors;
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of AGBNP particles has changed");
    for (int i = 0; i < force.getNumParticles(); i++) {
      double r, g;
      bool h;
      force.getParticleParameters(i, r, g, h);
    }
}
