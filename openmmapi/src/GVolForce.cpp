/* -------------------------------------------------------------------------- *
 *                             OpenMM-GVol                                  *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include "GVolForce.h"
#include "internal/GVolForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace GVolPlugin;
using namespace OpenMM;
using namespace std;

GVolForce::GVolForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0) {
}

int GVolForce::addParticle(double radius, double charge, bool ishydrogen){
  ParticleInfo particle(radius, charge, ishydrogen);
  particles.push_back(particle);
  return particles.size()-1;
}


GVolForce::NonbondedMethod GVolForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void GVolForce::setNonbondedMethod(NonbondedMethod method) {
    nonbondedMethod = method;
}

double GVolForce::getCutoffDistance() const {
    return cutoffDistance;
}

void GVolForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void GVolForce::getParticleParameters(int index,  double& radius, double& gamma, 
				      bool& ishydrogen) const { 

    ASSERT_VALID_INDEX(index, particles);
    radius = particles[index].radius;
    gamma = particles[index].gamma;
    ishydrogen = particles[index].ishydrogen;
}

ForceImpl* GVolForce::createImpl() const {
    return new GVolForceImpl(*this);
}

void GVolForce::updateParametersInContext(Context& context) {
    dynamic_cast<GVolForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
