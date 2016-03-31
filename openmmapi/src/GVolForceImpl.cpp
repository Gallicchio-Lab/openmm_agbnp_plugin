/* -------------------------------------------------------------------------- *
 *                            OpenMM-GVol                                   *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/GVolForceImpl.h"
#include "GVolKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace GVolPlugin;
using namespace OpenMM;
using namespace std;

GVolForceImpl::GVolForceImpl(const GVolForce& owner) : owner(owner) {
}

GVolForceImpl::~GVolForceImpl() {
}

void GVolForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcGVolForceKernel::Name(), context);
    kernel.getAs<CalcGVolForceKernel>().initialize(context.getSystem(), owner);
}

double GVolForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  return kernel.getAs<CalcGVolForceKernel>().execute(context, includeForces, includeEnergy);
}

std::vector<std::string> GVolForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGVolForceKernel::Name());
    return names;
}

void GVolForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGVolForceKernel>().copyParametersToContext(context, owner);
}
