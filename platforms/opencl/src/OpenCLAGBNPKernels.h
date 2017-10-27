#ifndef OPENCL_AGBNP_KERNELS_H_
#define OPENCL_AGBNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-AGBNP                                    *
 * -------------------------------------------------------------------------- */

#include "AGBNPUtils.h"
#include "AGBNPKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
using namespace std;

namespace AGBNPPlugin {


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
 * This kernel is invoked by AGBNPForce to calculate the forces acting on the system and the energy of the system.
 */
class OpenCLCalcAGBNPForceKernel : public CalcAGBNPForceKernel {
public:
    OpenCLCalcAGBNPForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::OpenCLContext& cl, const OpenMM::System& system) :
  CalcAGBNPForceKernel(name, platform), hasInitializedKernel(false), hasCreatedKernels(false), cl(cl), system(system) {
    }

    ~OpenCLCalcAGBNPForceKernel();
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
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
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

    /**
     * Copy overlap tree to device
     *
     */
    int copy_tree_to_device(void);

    /* init tree */
    void init_tree_size(int pad_modulo, 
			vector<int>& noverlaps, vector<int>& noverlaps_2body);

private:
    const AGBNPForce *gvol_force;

    int numParticles;
    unsigned int version;
    bool useCutoff;
    bool usePeriodic;
    bool useExclusions;
    double cutoffDistance;
    int maxTiles;
    bool hasInitializedKernel;
    bool hasCreatedKernels;
    OpenMM::OpenCLContext& cl;
    const OpenMM::System& system;
    OpenMM::OpenCLArray* radiusParam1;
    OpenMM::OpenCLArray* radiusParam2;
    OpenMM::OpenCLArray* gammaParam1;
    OpenMM::OpenCLArray* gammaParam2;
    OpenMM::OpenCLArray* ishydrogenParam;
    OpenMM::OpenCLArray* chargeParam;
    OpenMM::OpenCLArray* alphaParam;
    OpenMM::OpenCLArray* testBuffer;

    OpenMM::OpenCLArray* radtypeScreened;
    OpenMM::OpenCLArray* radtypeScreener;
    
    OpenMM::OpenCLArray* selfVolume;
    OpenMM::OpenCLArray* volScalingFactor;
    OpenMM::OpenCLArray* BornRadius;
    OpenMM::OpenCLArray* invBornRadius;
    OpenMM::OpenCLArray* invBornRadius_fp;
    OpenMM::OpenCLArray* GBDerY;
    OpenMM::OpenCLArray* GBDerBrU;
    OpenMM::OpenCLArray* GBDerU;
    OpenMM::OpenCLArray* VdWDerBrW;
    OpenMM::OpenCLArray* VdWDerW;
    
    OpenMM::OpenCLArray* selfVolumeBuffer_long;
    OpenMM::OpenCLArray* selfVolumeBuffer;
    OpenMM::OpenCLArray* AccumulationBuffer1_long;
    OpenMM::OpenCLArray* AccumulationBuffer1_real;
    OpenMM::OpenCLArray* AccumulationBuffer2_long;
    OpenMM::OpenCLArray* AccumulationBuffer2_real;
    
    
    // tree sizes etc
    int total_atoms_in_tree;
    int total_tree_size;
    int num_sections;
    int ov_work_group_size; //thread group size
    int num_compute_units;
    vector<int> tree_size;
    vector<int> padded_tree_size;
    vector<int> atom_tree_pointer; //pointers to 1-body atom slots
    vector<int> tree_pointer;      //pointers to tree sections
    vector<int> natoms_in_tree;    //no. atoms in each tree section
    vector<int> first_atom;        //the first atom in each tree section

    /* self volume coefficients */
    OpenMM::OpenCLArray* ovVolCoeff;
    /* overlap tree buffers */
    OpenMM::OpenCLArray* ovAtomTreePointer;
    OpenMM::OpenCLArray* ovAtomTreeSize;
    OpenMM::OpenCLArray* ovTreePointer;
    OpenMM::OpenCLArray* ovNumAtomsInTree;
    OpenMM::OpenCLArray* ovFirstAtom;
    OpenMM::OpenCLArray* NIterations;
    OpenMM::OpenCLArray* ovAtomTreePaddedSize;
    OpenMM::OpenCLArray* ovAtomTreeLock;
    OpenMM::OpenCLArray* ovLevel;
    OpenMM::OpenCLArray* ovG; // real4: Gaussian position + exponent
    OpenMM::OpenCLArray* ovVolume;
    OpenMM::OpenCLArray* ovVSfp;
    OpenMM::OpenCLArray* ovSelfVolume;
    OpenMM::OpenCLArray* ovVolEnergy;
    OpenMM::OpenCLArray* ovGamma1i;
    /* volume derivatives */
    OpenMM::OpenCLArray* ovDV1; // real4: dV12/dr1 + dV12/dV1 for each overlap
    OpenMM::OpenCLArray* ovDV2; // volume gradient accumulator
    OpenMM::OpenCLArray* ovPF;  //(P) and (F) aux variables
    
    OpenMM::OpenCLArray* ovLastAtom;
    OpenMM::OpenCLArray* ovRootIndex;
    OpenMM::OpenCLArray* ovChildrenStartIndex;
    OpenMM::OpenCLArray* ovChildrenCount;
    OpenMM::OpenCLArray* ovChildrenCountTop;
    OpenMM::OpenCLArray* ovChildrenCountBottom;
    OpenMM::OpenCLArray* ovProcessedFlag;
    OpenMM::OpenCLArray* ovOKtoProcessFlag;
    OpenMM::OpenCLArray* ovAtomBuffer;
    OpenMM::OpenCLArray* ovChildrenReported;


    cl::Kernel resetBufferKernel;
    cl::Kernel resetOvCountKernel;
    cl::Kernel resetTree;
    cl::Kernel resetSelfVolumesKernel;
    cl::Kernel InitOverlapTreeKernel_1body_1;
    cl::Kernel InitOverlapTreeKernel_1body_2;

    cl::Kernel InitOverlapTreeCountKernel;
    int InitOverlapTreeCountKernel_first_nbarg;

    
    cl::Kernel reduceovCountBufferKernel;
    
    cl::Kernel InitOverlapTreeKernel;
    int InitOverlapTreeKernel_first_nbarg;

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
    cl::Kernel RescanOverlapTreeGammasKernel_W;
    cl::Kernel InitOverlapTreeGammasKernel_1body_W;
    cl::Kernel computeVolumeEnergyKernel;

    /* Gaussian atomic parameters */
    vector<float> gaussian_exponent;
    vector<float> gaussian_volume;
    OpenMM::OpenCLArray* GaussianExponent;
    OpenMM::OpenCLArray* GaussianVolume;
    /* gamma parameters */
    vector<float> atomic_gamma;
    OpenMM::OpenCLArray* AtomicGamma;

    vector<bool> atom_ishydrogen;
    
    int niterations;

    int temp_buffer_size;
    OpenMM::OpenCLArray*  gvol_buffer_temp;
    OpenMM::OpenCLArray*  tree_pos_buffer_temp;
    OpenMM::OpenCLArray*  i_buffer_temp;
    OpenMM::OpenCLArray*  atomj_buffer_temp;

    //Born radii and such
    int ntypes_screener;
    AGBNPI42DLookupTable *i4_lut;    
    int i4_table_size; //x grid
    float i4_rmin, i4_rmax;  //x grid
    vector<float>y_i4; //function values
    vector<float>y2_i4; //derivatives
    OpenMM::OpenCLArray* i4YValues;
    OpenMM::OpenCLArray* i4Y2Values;
    OpenMM::OpenCLArray* testF;
    OpenMM::OpenCLArray* testDerF;
    
    cl::Kernel testHashKernel;
    cl::Kernel testLookupKernel;
    cl::Kernel initBornRadiiKernel;

    cl::Kernel inverseBornRadiiKernel;
    int inverseBornRadiiKernel_first_nbarg;

    cl::Kernel reduceBornRadiiKernel;
    cl::Kernel VdWEnergyKernel;
    cl::Kernel initVdWGBDerBornKernel;

    cl::Kernel VdWGBDerBornKernel;
    int VdWGBDerBornKernel_first_nbarg;
    
    cl::Kernel reduceVdWGBDerBornKernel;
    cl::Kernel initGBEnergyKernel;
    
    cl::Kernel GBPairEnergyKernel;
    int GBPairEnergyKernel_first_nbarg;
    
    cl::Kernel reduceGBEnergyKernel;


    int verbose_level;

    void executeInitKernels(ContextImpl& context, bool includeForces, bool includeEnergy);
    double executeGVolSA(ContextImpl& context, bool includeForces, bool includeEnergy);
    double executeAGBNP1(ContextImpl& context, bool includeForces, bool includeEnergy);
    double executeAGBNP2(ContextImpl& context, bool includeForces, bool includeEnergy); 

};

} // namespace AGBNPPlugin

#endif /*OPENCL_AGBNP_KERNELS_H_*/
