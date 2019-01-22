/* -------------------------------------------------------------------------- *
 *                            OpenMM-AGBNP                                   *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/AGBNPForceImpl.h"
#include "AGBNPKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>

using namespace AGBNPPlugin;
using namespace OpenMM;
using namespace std;

AGBNPForceImpl::AGBNPForceImpl(const AGBNPForce& owner) : owner(owner) {
}

AGBNPForceImpl::~AGBNPForceImpl() {
}

void AGBNPForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcAGBNPForceKernel::Name(), context);
    kernel.getAs<CalcAGBNPForceKernel>().initialize(context.getSystem(), owner);
}

double AGBNPForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if ((groups&(1<<owner.getForceGroup())) != 0)
    return kernel.getAs<CalcAGBNPForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::vector<std::string> AGBNPForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcAGBNPForceKernel::Name());
    return names;
}

void AGBNPForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcAGBNPForceKernel>().copyParametersToContext(context, owner);
}
