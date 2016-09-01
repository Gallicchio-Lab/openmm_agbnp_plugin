#ifndef OPENCL_GVol_KERNELS_H_
#define OPENCL_GVol_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-GVol                                    *
 * -------------------------------------------------------------------------- */

#include "GVolKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
using namespace std;

namespace GVolPlugin {


class GOverlap_Tree {
 public:
  int size;
  int direction;
  int next_slot_to_process;
  //list of overlaps
  vector<int> level;               //level (0=root, 1=atoms, 2=2-body, 3=3-body, etc.)
  vector<float> a;                 // Gaussian exponent
  vector<float> volume;            // Gaussian volume
  vector<float> cx, cy, cz;        //coordinates of center
  vector<float> dvv1;              // derivative wrt volume of first atom (also stores F1..i in GPU version)
  vector<float> dv1x, dv1y, dv1z;  // derivative wrt position of first atom (also stores P1..i in GPU version) 
  vector<float> gamma1i;           // sum gammai for this overlap
  vector<float> self_volume;       //self volume of overlap
  vector<float> sfp;               //switching function derivatives  
  vector<int> atom;                //the last atom forming the overlap
  vector<int> root_index;        //the index of the parent (or -1 for null)
  vector<int> children_startindex; //start index in tree of list of children (or -1 for null)
  vector<int> children_count;      //number of children
  vector<int> processed;           //a flag to mark overlap that has been processed
  vector<int> process;             //a flag to mark an overlap to be processed
};

/**
 * This kernel is invoked by GVolForce to calculate the forces acting on the system and the energy of the system.
 */
class OpenCLCalcGVolForceKernel : public CalcGVolForceKernel {
public:
    OpenCLCalcGVolForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::OpenCLContext& cl, const OpenMM::System& system) :
  CalcGVolForceKernel(name, platform), hasInitializedKernel(false), hasCreatedKernels(false), cl(cl), system(system) {
    }

    ~OpenCLCalcGVolForceKernel();
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
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
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

    /**
     * Copy overlap tree to device
     *
     */
    int copy_tree_to_device(void);

    /* init tree */
    //int init_overlap_trees_ofatoms(vector<float> &x, vector<float> &y, vector<float> &z, vector<float> &r, vector<float> &gamma);

private:
    const GVolForce *gvol_force;

    int numParticles;
    bool useCutoff;
    bool usePeriodic;
    bool useExclusions;
    double cutoffDistance;
    bool hasInitializedKernel;
    bool hasCreatedKernels;
    OpenMM::OpenCLContext& cl;
    const OpenMM::System& system;
    OpenMM::OpenCLArray* radiusParam1;
    OpenMM::OpenCLArray* radiusParam2;
    OpenMM::OpenCLArray* gammaParam1;
    OpenMM::OpenCLArray* gammaParam2;
    OpenMM::OpenCLArray* ishydrogenParam;
    /* list of overlap trees (one for each atom) */
    int total_tree_size;
    int num_sections;
    int num_twobody_max;
    int ov_work_group_size;
    vector<int> tree_size;
    vector<int> padded_tree_size;
    vector<int> tree_pointer;
    /* self volume coefficients */
    OpenMM::OpenCLArray* ovVolCoeff;
    /* overlap tree buffers */
    OpenMM::OpenCLArray* ovAtomTreePointer;
    OpenMM::OpenCLArray* ovAtomTreeSize;
    OpenMM::OpenCLArray* NIterations;
    OpenMM::OpenCLArray* ovAtomTreePaddedSize;
    OpenMM::OpenCLArray* ovAtomLock;
    OpenMM::OpenCLArray* ovAtomTreeLock;
    OpenMM::OpenCLArray* ovLevel;
    OpenMM::OpenCLArray* ovG; // real4: Gaussian position + exponent
    OpenMM::OpenCLArray* ovVolume;
    OpenMM::OpenCLArray* ovVSfp;
    OpenMM::OpenCLArray* ovSelfVolume;
    OpenMM::OpenCLArray* ovGamma1i;
    OpenMM::OpenCLArray* ovCountBuffer;
    /* volume derivatives */
    OpenMM::OpenCLArray* ovDV1; // real4: dV12/dr1 + dV12/dV1
    OpenMM::OpenCLArray* ovDV2; // real4: dPsi12/dr2

    OpenMM::OpenCLArray* ovLastAtom;
    OpenMM::OpenCLArray* ovRootIndex;
    OpenMM::OpenCLArray* ovChildrenStartIndex;
    OpenMM::OpenCLArray* ovChildrenCount;
    OpenMM::OpenCLArray* ovProcessedFlag;
    OpenMM::OpenCLArray* ovOKtoProcessFlag;
    OpenMM::OpenCLArray* ovAtomBuffer;
    OpenMM::OpenCLArray* ovChildrenReported;

    OpenMM::OpenCLArray* ovEnergyBuffer_long;

    cl::Kernel resetBufferKernel;
    cl::Kernel resetOvCountKernel;
    cl::Kernel resetTree;
    cl::Kernel resetSelfVolumesKernel;
    cl::Kernel InitOverlapTreeKernel_1body_1;
    cl::Kernel InitOverlapTreeKernel_1body_2;
    cl::Kernel InitOverlapTreeCountKernel;
    cl::Kernel reduceovCountBufferKernel;
    cl::Kernel InitOverlapTreeKernel;
    cl::Kernel ComputeOverlapTreeKernel;
    cl::Kernel ComputeOverlapTree_1passKernel;
    cl::Kernel computeSelfVolumesKernel;
    cl::Kernel reduceSelfVolumesKernel_tree;
    cl::Kernel reduceSelfVolumesKernel_buffer;
    cl::Kernel resetTreeKernel;
    cl::Kernel SortOverlapTree2bodyKernel;
    cl::Kernel resetComputeOverlapTreeKernel;
    cl::Kernel ResetRescanOverlapTreeKernel;
    cl::Kernel InitRescanOverlapTreeKernel;
    cl::Kernel RescanOverlapTreeKernel;

    /* Gaussian atomic parameters */
    vector<float> gaussian_exponent;
    vector<float> gaussian_volume;
    OpenMM::OpenCLArray* GaussianExponent;
    OpenMM::OpenCLArray* GaussianVolume;
    /* gamma parameters */
    vector<float> atomic_gamma;
    OpenMM::OpenCLArray* AtomicGamma;
    
    int niterations;

    int temp_buffer_size;
    OpenMM::OpenCLArray*  gvol_buffer_temp;
    OpenMM::OpenCLArray*  tree_pos_buffer_temp;
    OpenMM::OpenCLArray*  i_buffer_temp;
    OpenMM::OpenCLArray*  atomj_buffer_temp;
};

} // namespace GVolPlugin

#endif /*OPENCL_GVol_KERNELS_H_*/
