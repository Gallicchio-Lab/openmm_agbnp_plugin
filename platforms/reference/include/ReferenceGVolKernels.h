#ifndef REFERENCE_GVol_KERNELS_H_
#define REFERENCE_GVol_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-GVol                                    *
 * -------------------------------------------------------------------------- */
#include "GVolKernels.h"
#include "openmm/Platform.h"
#include <vector>
#include "gaussvol.h"

namespace GVolPlugin {

/**
 * This kernel is invoked by GVolForce to calculate the forces acting 
 * on the system and the energy of the system.
 */
class ReferenceCalcGVolForceKernel : public CalcGVolForceKernel {
public:
    ReferenceCalcGVolForceKernel(std::string name, const OpenMM::Platform& platform) : CalcGVolForceKernel(name, platform) {
    gvol = 0;
    }
  ~ReferenceCalcGVolForceKernel(){
    if(gvol) delete gvol;
    positions.clear();
    ishydrogen.clear();
    radii.clear();
    gammas.clear();
    free_volume.clear();
    self_volume.clear();
    surface_areas.clear();
    surf_force.clear();
  }


    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GVolForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GVolForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GVolForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GVolForce& force);
 
private:
    GaussVol *gvol; // gaussvol instance
    //inputs
    int numParticles;
    std::vector<RealVec> positions;
    std::vector<bool> ishydrogen;
    std::vector<RealOpenMM> radii;
    std::vector<RealOpenMM> gammas;
    //outputs
    RealOpenMM surf_energy;
    std::vector<RealOpenMM> free_volume, self_volume, surface_areas;
    std::vector<RealVec> surf_force;
};

} // namespace GVolPlugin

#endif /*REFERENCE_GVol_KERNELS_H_*/
