#ifndef REFERENCE_AGBNP_KERNELS_H_
#define REFERENCE_AGBNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-AGBNP                                    *
 * -------------------------------------------------------------------------- */
#include "AGBNPKernels.h"
#include "AGBNPUtils.h"
#include "openmm/Platform.h"
#include <vector>
#include "gaussvol.h"

namespace AGBNPPlugin {

/**
 * This kernel is invoked by AGBNPForce to calculate the forces acting 
 * on the system and the energy of the system.
 */
class ReferenceCalcAGBNPForceKernel : public CalcAGBNPForceKernel {
public:
    ReferenceCalcAGBNPForceKernel(std::string name, const OpenMM::Platform& platform) : CalcAGBNPForceKernel(name, platform) {
    gvol = 0;
    }
  ~ReferenceCalcAGBNPForceKernel(){
    if(gvol) delete gvol;
    positions.clear();
    ishydrogen.clear();
    radii_vdw.clear();
    radii_large.clear();
    gammas.clear();
    vdw_alpha.clear();
    charge.clear();
    free_volume.clear();
    self_volume.clear();
    surface_areas.clear();
    vol_force.clear();
    vol_dv.clear();
    volume_scaling_factor.clear();
    inverse_born_radius.clear();
    inverse_born_radius_fp.clear();
    born_radius.clear();
  }


    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the AGBNPForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const AGBNPForce& force);
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
     * @param force      the AGBNPForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const AGBNPForce& force);
 
private:
    GaussVol *gvol; // gaussvol instance
    unsigned int version; //1 or 2
    //inputs
    int numParticles;
    std::vector<RealVec> positions;
    std::vector<int> ishydrogen;
    std::vector<RealOpenMM> radii_vdw;
    std::vector<RealOpenMM> radii_large;
    std::vector<RealOpenMM> gammas;
    std::vector<RealOpenMM> vdw_alpha;
    std::vector<RealOpenMM> charge;
    //outputs
    std::vector<RealOpenMM> free_volume, self_volume, surface_areas;
    std::vector<RealVec> vol_force;
    std::vector<RealOpenMM> vol_dv;
    AGBNPI42DLookupTable *i4_lut;
    std::vector<RealOpenMM> volume_scaling_factor;
    std::vector<RealOpenMM> inverse_born_radius;
    std::vector<RealOpenMM> inverse_born_radius_fp;
    std::vector<RealOpenMM> born_radius;
    double solvent_radius;
    
    double executeGVolSA(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    double executeAGBNP1(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    double executeAGBNP2(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);


    
};


 //class to record MS particles
 class MSParticle {
 public:
   double vol;
   double freevol;
   double selfvol;
   RealVec pos;
   int parent1;
   int parent2;
 };

 
} // namespace AGBNPPlugin

#endif /*REFERENCE_AGBNP_KERNELS_H_*/
