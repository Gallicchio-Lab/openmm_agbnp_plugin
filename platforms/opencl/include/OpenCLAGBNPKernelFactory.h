#ifndef OPENMM_OPENCLAGBNPKERNELFACTORY_H_
#define OPENMM_OPENCLAGBNPKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-AGBNP                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace AGBNPPlugin {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the AGBNP plugin.
 */

class OpenCLAGBNPKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace AGBNPPlugin

#endif /*OPENMM_OPENCLAGBNPKERNELFACTORY_H_*/
