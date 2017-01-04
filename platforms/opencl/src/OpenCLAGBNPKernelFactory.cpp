/* -------------------------------------------------------------------------- *
 *                            OpenMM-GVol                                   *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "OpenCLGVolKernelFactory.h"
#include "OpenCLGVolKernels.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace GVolPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLGVolKernelFactory* factory = new OpenCLGVolKernelFactory();
        platform.registerKernelFactory(CalcGVolForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerGVolOpenCLKernelFactories() {
    try {
        Platform::getPlatformByName("OpenCL");
    }
    catch (...) {
        Platform::registerPlatform(new OpenCLPlatform());
    }
    registerKernelFactories();
}

KernelImpl* OpenCLGVolKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    OpenCLContext& cl = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGVolForceKernel::Name())
        return new OpenCLCalcGVolForceKernel(name, platform, cl, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
