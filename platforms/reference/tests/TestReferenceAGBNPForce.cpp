/* -------------------------------------------------------------------------- *
 *                              OpenMM-AGBNP                                 *
 * -------------------------------------------------------------------------- */

/**
 * This tests the Reference implementation of AGBNPForce.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "AGBNPForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/reference/RealVec.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/NonbondedForce.h"

using namespace AGBNPPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerAGBNPReferenceKernelFactories();

void testForce() {
    bool verbose = true;
    bool veryverbose = false;
    System system;
    NonbondedForce *nb = new NonbondedForce();
    AGBNPForce* force = new AGBNPForce();
    force->setVersion(1);
    system.addForce(nb);
    system.addForce(force);
    //read from stdin
    int numParticles = 0;
    double id, x, y, z, radius, charge;
    double epsilon, sigma, bornr;
    double gamma;
    bool ishydrogen;
    vector<int> ihi;
    vector<Vec3> positions;
    std::cin >> numParticles;
    int ih;
    double ang2nm = 0.1;
    double kcalmol2kjmol = 4.184;


    double sigmaw = 3.15365*ang2nm; /* LJ sigma of TIP4P water oxygen */
    double epsilonw = 0.155*kcalmol2kjmol;        /* LJ epsilon of TIP4P water oxygen */
    double rho = 0.033428/pow(ang2nm,3);   /* water number density */
    double epsilon_LJ = 0.155*kcalmol2kjmol;
    double sigma_LJ;

    for(int i=0;i<numParticles;i++){
      std::cin >> id >> x >> y >> z >> radius >> charge >> gamma >> ih;
      system.addParticle(1.0);
      positions.push_back(Vec3(x, y, z)*ang2nm);
      ihi.push_back(ih);
      ishydrogen = (ih > 0);
      radius *= ang2nm;
      gamma *= kcalmol2kjmol/(ang2nm*ang2nm);
      sigma_LJ = 2.*radius;
      double sij = sqrt(sigmaw*sigma_LJ);
      double eij = sqrt(epsilonw*epsilon_LJ);
      double alpha = - 16.0 * M_PI * rho * eij * pow(sij,6) / 3.0;
      nb->addParticle(0.0,0.0,0.0);
      force->addParticle(radius, gamma, alpha, charge, ishydrogen);      
    }
    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("Reference");
    //It doesnt look like the reference platform supports changing precision
    //map<string, string> properties;
    //properties["Precision"] = "single";
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    double energy1 = state.getPotentialEnergy();
    std::cout << "Energy: " <<  energy1  << std::endl;

    if(veryverbose){
      //print out forces for debugging
      cout << "Forces: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "FW: " << i << " " << state.getForces()[i][0] << " " << state.getForces()[i][1] << " "  << state.getForces()[i][2] << " "<< endl;      
      }
    }
    
#ifdef NOTNOW
    // validate force by moving heavy atoms
    vector<RealVec> forces;
    for(int i = 0; i < numParticles; i++){
      forces.push_back(state.getForces()[i]);
    }
    double offset = 2.e-4;
    double max =  offset;
    double min = -offset;
    for(int i=0; i<numParticles; i++){
      if(ihi[i] > 0) continue;
      double dx = ((double) rand() / (RAND_MAX)) * (max-min) + min;
      double dy = ((double) rand() / (RAND_MAX)) * (max-min) + min;
      double dz = ((double) rand() / (RAND_MAX)) * (max-min) + min;
      RealVec displ = RealVec(dx,dy,dz);
      RealVec save_pos = positions[i];
      positions[i] += displ;
      context.setPositions(positions);
      double energy2 = context.getState(State::Energy).getPotentialEnergy();
      double de = -forces[i].dot(displ);
      std::cout << "P " << i << " " << energy2 - energy1 << " " << de << endl;
      positions[i] = save_pos;
    }
#endif
#ifdef NOTNOW
    double offset = 2.e-3;
    int pmove = 121;
    int direction = 1;
    positions[pmove][direction] += offset;
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double de = -state.getForces()[pmove][direction]*offset;
    std::cout << "Energy: " <<  energy2  << std::endl;
    std::cout << "Energy Change: " <<  energy2 - energy1  << std::endl;
    std::cout << "Energy Change from Gradient: " <<  de  << std::endl;
#endif
}

int main() {
  try {
    registerAGBNPReferenceKernelFactories();
    testForce();
	//        testChangingParameters();
  }
  catch(const std::exception& e) {
    std::cout << "exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
