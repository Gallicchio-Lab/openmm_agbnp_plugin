#ifndef OPENMM_REFERENCEAGBNPKERNELFACTORY_H_
#define OPENMM_REFERENCEAGBNPKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-AGBNP                              *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the 
 * AGBNP plugin.
 */

class ReferenceAGBNPKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEAGBNPKERNELFACTORY_H_*/
