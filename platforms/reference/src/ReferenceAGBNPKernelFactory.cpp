/* -------------------------------------------------------------------------- *
 *                              OpenMM-AGBNP                                 *
 * -------------------------------------------------------------------------- */

#include "ReferenceAGBNPKernelFactory.h"
#include "ReferenceAGBNPKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace AGBNPPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceAGBNPKernelFactory* factory = new ReferenceAGBNPKernelFactory();
            platform.registerKernelFactory(CalcAGBNPForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerAGBNPReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceAGBNPKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcAGBNPForceKernel::Name())
        return new ReferenceCalcAGBNPForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
