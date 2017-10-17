/* -------------------------------------------------------------------------- *
 *                             OpenMM-AGBNP                                  *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include "AGBNPForce.h"
#include "internal/AGBNPForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace AGBNPPlugin;
using namespace OpenMM;
using namespace std;

AGBNPForce::AGBNPForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), version(1) {
}

int AGBNPForce::addParticle(double radius, double gamma, double vdw_alpha, double charge, bool ishydrogen){
  ParticleInfo particle(radius, gamma, vdw_alpha, charge, ishydrogen);
  particles.push_back(particle);
  return particles.size()-1;
}


AGBNPForce::NonbondedMethod AGBNPForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void AGBNPForce::setNonbondedMethod(NonbondedMethod method) {
    nonbondedMethod = method;
}

double AGBNPForce::getCutoffDistance() const {
    return cutoffDistance;
}

void AGBNPForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void AGBNPForce::getParticleParameters(int index,  double& radius, double& gamma, double &vdw_alpha, double &charge,
				      bool& ishydrogen) const { 

    ASSERT_VALID_INDEX(index, particles);
    radius = particles[index].radius;
    gamma = particles[index].gamma;
    vdw_alpha = particles[index].vdw_alpha;
    charge = particles[index].charge;
    ishydrogen = particles[index].ishydrogen;
}

ForceImpl* AGBNPForce::createImpl() const {
    return new AGBNPForceImpl(*this);
}

void AGBNPForce::updateParametersInContext(Context& context) {
    dynamic_cast<AGBNPForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
