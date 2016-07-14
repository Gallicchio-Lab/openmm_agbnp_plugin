/* -------------------------------------------------------------------------- *
 *                              OpenMM-GVol                                 *
 * -------------------------------------------------------------------------- */

/**
 * This tests the Reference implementation of GVolForce.
 */

#include "GVolForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace GVolPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerGVolReferenceKernelFactories();

void testForce() {
    System system;
    GVolForce* force = new GVolForce();
    system.addForce(force);
    //read from stdin
    int numParticles = 0;
    double id, x, y, z, radius, charge;
    double epsilon, sigma, bornr;
    double gamma;
    bool ishydrogen;
    vector<Vec3> positions;
    std::cin >> numParticles;
    int ih;
    double ang2nm = 0.1;
    double kcalmol2kjmol = 4.184;
    for(int i=0;i<numParticles;i++){
      std::cin >> id >> x >> y >> z >> radius >> charge >> gamma >> ih;
      system.addParticle(1.0);
      positions.push_back(Vec3(x, y, z)*ang2nm);
      ishydrogen = (ih > 0);
      radius *= ang2nm;
      gamma *= kcalmol2kjmol/(ang2nm*ang2nm);
      force->addParticle(radius, gamma, ishydrogen);      
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

    //#ifdef NOTNOW
    // validate force by moving an atom
    double offset = 1.e-3;
    int pmove = 121;
    positions[pmove][0] += offset;
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double de = -state.getForces()[pmove][0]*offset;
    std::cout << "Energy: " <<  energy2  << std::endl;
    std::cout << "Energy Change: " <<  energy2 - energy1  << std::endl;
    std::cout << "Energy Change from Gradient: " <<  de  << std::endl;
    //#endif
}

int main() {
  try {
    registerGVolReferenceKernelFactories();
    testForce();
	//        testChangingParameters();
  }
  catch(const std::exception& e) {
    std::cout << "exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
