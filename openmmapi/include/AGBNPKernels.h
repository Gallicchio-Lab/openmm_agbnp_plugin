#ifndef GVol_KERNELS_H_
#define GVol_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMM-GVol                                 *
 * -------------------------------------------------------------------------- */

#include "GVolForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace GVolPlugin {

/**
 * This kernel is invoked by GVolForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGVolForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcGVolForce";
    }
    CalcGVolForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GVolForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const GVolForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GVolForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const GVolForce& force) = 0;
};

} // namespace GVolPlugin

#endif /*GVol_KERNELS_H_*/
