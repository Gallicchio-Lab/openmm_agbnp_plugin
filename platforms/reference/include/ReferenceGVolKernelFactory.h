#ifndef OPENMM_REFERENCEGVolKERNELFACTORY_H_
#define OPENMM_REFERENCEGVolKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GVol                              *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the 
 * GVol plugin.
 */

class ReferenceGVolKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEGVolKERNELFACTORY_H_*/
