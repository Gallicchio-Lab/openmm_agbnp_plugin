#ifndef OPENMM_OPENCLGVolKERNELFACTORY_H_
#define OPENMM_OPENCLGVolKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-GVol                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace GVolPlugin {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the GVol plugin.
 */

class OpenCLGVolKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace GVolPlugin

#endif /*OPENMM_OPENCLGVolKERNELFACTORY_H_*/
