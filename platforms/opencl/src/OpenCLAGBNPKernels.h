#ifndef OPENCL_AGBNP_KERNELS_H_
#define OPENCL_AGBNP_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                           OpenMM-AGBNP                                    *
 * -------------------------------------------------------------------------- */

#include "AGBNPUtils.h"
#include "AGBNPKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/reference/RealVec.h"
using namespace std;

namespace AGBNPPlugin {

/**
 * This kernel is invoked by AGBNPForce to calculate the forces acting on the system and the energy of the system.
 */
class OpenCLCalcAGBNPForceKernel : public CalcAGBNPForceKernel {
public:
    OpenCLCalcAGBNPForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::OpenCLContext& cl, const OpenMM::System& system) :
  CalcAGBNPForceKernel(name, platform), cl(cl), system(system) {

    hasCreatedKernels = false;
    hasInitializedKernels = false;
    
    radtypeScreened = NULL;
    radtypeScreener = NULL;
    
    selfVolume = NULL;
    selfVolumeLargeR = NULL;
    Semaphor = NULL;
    volScalingFactor = NULL;
    BornRadius = NULL;
    invBornRadius = NULL;
    invBornRadius_fp = NULL;
    GBDerY = NULL;
    GBDerBrU = NULL;
    GBDerU = NULL;
    VdWDerBrW = NULL;
    VdWDerW = NULL;
    
    GaussianExponent = NULL;
    GaussianVolume = NULL;
    GaussianExponentLargeR = NULL;
    GaussianVolumeLargeR = NULL;
    
    AtomicGamma = NULL;
    grad = NULL;
	
    
    i4_lut = NULL;    
    i4YValues = NULL;
    i4Y2Values = NULL;
    testF = NULL;
    testDerF = NULL;
    
    PanicButton = NULL;
    pinnedPanicButtonBuffer = NULL;
    pinnedPanicButtonMemory = NULL;

    do_ms = false;
    MSparticle1 = NULL;
    MSparticle2 = NULL;
    
    MScount1 = NULL;
    MScount2 = NULL;
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

    class OpenCLOverlapTree {
    public:
      OpenCLOverlapTree(void){
	ovAtomTreePointer = NULL;
	ovAtomTreeSize = NULL;
	ovTreePointer = NULL;
	ovNumAtomsInTree = NULL;
	ovFirstAtom = NULL;
	NIterations = NULL;
	ovAtomTreePaddedSize = NULL;
	ovAtomTreeLock = NULL;
	ovLevel = NULL;
	ovG = NULL;
	ovVolume = NULL;
	ovVsp = NULL;
	ovVSfp = NULL;
	ovSelfVolume = NULL;
	ovVolEnergy = NULL;
	ovGamma1i = NULL;
	ovDV1 = NULL;
	ovDV2 = NULL;
	ovPF = NULL;
	ovLastAtom = NULL;
	ovRootIndex = NULL;
	ovChildrenStartIndex = NULL;
	ovChildrenCount = NULL;
	ovChildrenCountTop = NULL;
	ovChildrenCountBottom = NULL;
	ovProcessedFlag = NULL;
	ovOKtoProcessFlag = NULL;
	ovChildrenReported = NULL;

	ovAtomBuffer = NULL;	    
	EnergyBuffer_long = NULL;
	selfVolumeBuffer_long = NULL;
	selfVolumeBuffer = NULL;
	AccumulationBuffer1_long = NULL;
	AccumulationBuffer1_real = NULL;
	AccumulationBuffer2_long = NULL;
	AccumulationBuffer2_real = NULL;
	gradBuffers_long = NULL;
	
	temp_buffer_size = -1;
	gvol_buffer_temp = NULL;
	tree_pos_buffer_temp = NULL;
	i_buffer_temp = NULL;
	atomj_buffer_temp = NULL;
	
	has_saved_noverlaps = false;
	tree_size_boost = 2;//6;//debug 2 is default

	hasExceededTempBuffer = false;    

      };

      ~OpenCLOverlapTree(void){
	delete ovAtomTreePointer;
	delete ovAtomTreeSize;
	delete ovTreePointer;
	delete ovNumAtomsInTree;
	delete ovFirstAtom;
	delete NIterations;
	delete ovAtomTreePaddedSize;
	delete ovAtomTreeLock;
	delete ovLevel;
	delete ovG;
	delete ovVolume;
	delete ovVsp;
	delete ovVSfp;
	delete ovSelfVolume;
	delete ovVolEnergy;
	delete ovGamma1i;
	delete ovDV1;
	delete ovDV2;
	delete ovPF;
	delete ovLastAtom;
	delete ovRootIndex;
	delete ovChildrenStartIndex;
	delete ovChildrenCount;
	delete ovChildrenCountTop;
	delete ovChildrenCountBottom;
	delete ovProcessedFlag;
	delete ovOKtoProcessFlag;
	delete ovChildrenReported;

	delete ovAtomBuffer;	    
	delete selfVolumeBuffer_long;
	delete selfVolumeBuffer;
	delete EnergyBuffer_long;
	delete AccumulationBuffer1_long;
	delete AccumulationBuffer1_real;
	delete AccumulationBuffer2_long;
	delete AccumulationBuffer2_real;
	delete gradBuffers_long;
	
	delete gvol_buffer_temp;
	delete tree_pos_buffer_temp;
	delete i_buffer_temp;
	delete atomj_buffer_temp;
      }; 
      
      //initializes tree sections and sizes with number of atoms and number of overlaps
      void init_tree_size(int num_atoms, int padded_num_atoms, int num_compute_units, int pad_modulo, vector<int>& noverlaps_current);

      //simpler version with precomputed tree sizes
      void init_tree_size(int padded_num_atoms, int tree_section_size, int num_compute_units, int pad_modulo);
      
      //resizes tree buffers
      void resize_tree_buffers(OpenMM::OpenCLContext& cl, int ov_work_group_size);
      
      //copies the tree framework to OpenCL device memory
      int copy_tree_to_device(void);
      

      // host variables and buffers
      int num_atoms;
      int padded_num_atoms;
      int total_atoms_in_tree;
      int total_tree_size;
      int num_sections;
      vector<int> tree_size;
      vector<int> padded_tree_size;
      vector<int> atom_tree_pointer; //pointers to 1-body atom slots
      vector<int> tree_pointer;      //pointers to tree sections
      vector<int> natoms_in_tree;    //no. atoms in each tree section
      vector<int> first_atom;        //the first atom in each tree section

      /* overlap tree buffers on Device */
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
      OpenMM::OpenCLArray* ovVsp;
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
      OpenMM::OpenCLArray* ovChildrenReported;

      OpenMM::OpenCLArray* ovAtomBuffer;
      OpenMM::OpenCLArray* EnergyBuffer_long;
      OpenMM::OpenCLArray* selfVolumeBuffer_long;
      OpenMM::OpenCLArray* selfVolumeBuffer;
      OpenMM::OpenCLArray* AccumulationBuffer1_long;
      OpenMM::OpenCLArray* AccumulationBuffer1_real;
      OpenMM::OpenCLArray* AccumulationBuffer2_long;
      OpenMM::OpenCLArray* AccumulationBuffer2_real;
      OpenMM::OpenCLArray* gradBuffers_long;
      
      int temp_buffer_size;
      OpenMM::OpenCLArray*  gvol_buffer_temp;
      OpenMM::OpenCLArray*  tree_pos_buffer_temp;
      OpenMM::OpenCLArray*  i_buffer_temp;
      OpenMM::OpenCLArray*  atomj_buffer_temp;

      double tree_size_boost;
      int has_saved_noverlaps;
      vector<int> saved_noverlaps;

      bool hasExceededTempBuffer;
    };//class OpenCLOverlapTree


    //a class to mange MS particle OpenCL buffers
    class OpenCLMSParticle {
    public:
      OpenCLMSParticle(int count, int size, int ntiles, int tile_size, OpenMM::OpenCLContext& cl): ms_count(count), ms_size(size), ntiles(ntiles), tile_size(tile_size), cl(cl) {
	MScount =     OpenCLArray::create<cl_int>(cl, ntiles, "MScount");
	MSptr =       OpenCLArray::create<cl_int>(cl, size, "MSptr");
	MSpVol0 =     OpenCLArray::create<cl_float>(cl, size, "MSpVol0");
	MSpVolLarge = OpenCLArray::create<cl_float>(cl, size, "MSpVolLarge");
	MSpVolvdW =   OpenCLArray::create<cl_float>(cl, size, "MSpVolvdW");
	MSpsspLarge = OpenCLArray::create<cl_float>(cl, size, "MSpsspLarge");
	MSpsspvdW =   OpenCLArray::create<cl_float>(cl, size, "MSpsspvdW");
	MSpPos =      OpenCLArray::create<mm_float4>(cl, size, "MSpPos");
	MSpParent1 =  OpenCLArray::create<cl_int>(cl, size, "MSpParent1");
	MSpParent2 =  OpenCLArray::create<cl_int>(cl, size, "MSpParent2");
	MSpgder =     OpenCLArray::create<mm_float4>(cl, size, "MSpgder");
	MSphder =     OpenCLArray::create<mm_float4>(cl, size, "MSphder");
	MSpfms =      OpenCLArray::create<cl_float>(cl, size, "MSpfms");
	MSpG0Large =  OpenCLArray::create<cl_float>(cl, size, "MSpG0Large");
	MSpG0vdW =    OpenCLArray::create<cl_float>(cl, size, "MSpG0vdW");
	MSpGaussExponent =    OpenCLArray::create<cl_float>(cl, size, "MSpGaussExponent");
	MSpGamma =    OpenCLArray::create<cl_float>(cl, size, "MSpGamma");
	MSpSelfVolume = OpenCLArray::create<cl_float>(cl, size, "MSpSelfVolume");
	MSgrad = OpenCLArray::create<mm_float4>(cl, size, "MSgrad");
	MSsemaphor =  OpenCLArray::create<cl_int>(cl, size, "MSsemaphor");
	vector<int> zeros(size);
	for(int i=0; i<size; i++) zeros[i] = 0;
	MSsemaphor->upload(zeros);
      };

      ~OpenCLMSParticle(void){
	delete MScount;
	delete MSptr;
	delete MSpVol0;
	delete MSpVolLarge;
	delete MSpVolvdW;
	delete MSpsspLarge;
	delete MSpsspvdW;
	delete MSpPos;
	delete MSpParent1;
	delete MSpParent2;
	delete MSpgder;
	delete MSphder;
	delete MSpfms;
	delete MSpG0Large;
	delete MSpG0vdW;
	delete MSpGaussExponent;
	delete MSpGamma;
	delete MSpSelfVolume;
	delete MSgrad;
	delete MSsemaphor;
      };

      void resize(int count, int size, int tile_size){
	ms_count = count;
	this->tile_size = tile_size;
	if(ms_size < size){
	  ms_size = size;
	  delete MSptr;
	  delete MSpVol0;
	  delete MSpVolLarge;
	  delete MSpVolvdW;
	  delete MSpsspLarge;
	  delete MSpsspvdW;
	  delete MSpPos;
	  delete MSpParent1;
	  delete MSpParent2;
	  delete MSpgder;
	  delete MSphder;
	  delete MSpfms;
	  delete MSpG0Large;
	  delete MSpG0vdW;
	  delete MSpGaussExponent;
	  delete MSpGamma;
	  delete MSpSelfVolume;
	  delete MSgrad;
	  delete MSsemaphor;
	  MSptr =       OpenCLArray::create<cl_int>(cl, size, "MSptr");
	  MSpVol0 =     OpenCLArray::create<cl_float>(cl, size, "MSpVol0");
	  MSpVolLarge = OpenCLArray::create<cl_float>(cl, size, "MSpVolLarge");
	  MSpVolvdW =   OpenCLArray::create<cl_float>(cl, size, "MSpVolvdW");
	  MSpsspLarge = OpenCLArray::create<cl_float>(cl, size, "MSpsspLarge");
	  MSpsspvdW =   OpenCLArray::create<cl_float>(cl, size, "MSpsspvdW");
	  MSpPos =      OpenCLArray::create<mm_float4>(cl, size, "MSpPos");
	  MSpParent1 =  OpenCLArray::create<cl_int>(cl, size, "MSpParent1");
	  MSpParent2 =  OpenCLArray::create<cl_int>(cl, size, "MSpParent2");
	  MSpgder =     OpenCLArray::create<mm_float4>(cl, size, "MSpgder");
	  MSphder =     OpenCLArray::create<mm_float4>(cl, size, "MSphder");
	  MSpfms =      OpenCLArray::create<cl_float>(cl, size, "MSpfms");
	  MSpG0Large =  OpenCLArray::create<cl_float>(cl, size, "MSpG0Large");
	  MSpG0vdW =    OpenCLArray::create<cl_float>(cl, size, "MSpG0vdW");
	  MSpGaussExponent =    OpenCLArray::create<cl_float>(cl, size, "MSpGaussExponent");
	  MSpGamma =    OpenCLArray::create<cl_float>(cl, size, "MSpGamma");
	  MSpSelfVolume = OpenCLArray::create<cl_float>(cl, size, "MSpSelfVolume");
	  MSgrad = OpenCLArray::create<mm_float4>(cl, size, "MSgrad");
	  MSsemaphor =  OpenCLArray::create<cl_int>(cl, size, "MSsemaphor");
	  vector<int> zeros(size);
	  for(int i=0; i<size; i++) zeros[i] = 0;
	  MSsemaphor->upload(zeros);
	}
      };
      
      OpenMM::OpenCLContext& cl;
      int ms_size;
      int ms_count;
      int ntiles;//a tile for each warp
      int tile_size;//size of a tile
      OpenMM::OpenCLArray* MScount;//number of MS particle for each tile
      OpenMM::OpenCLArray* MSptr;//pointer to MS list for each MS particle
      OpenMM::OpenCLArray* MSpVol0;
      OpenMM::OpenCLArray* MSpVolLarge;
      OpenMM::OpenCLArray* MSpVolvdW;
      OpenMM::OpenCLArray* MSpsspLarge;
      OpenMM::OpenCLArray* MSpsspvdW;
      OpenMM::OpenCLArray* MSpPos;
      OpenMM::OpenCLArray* MSpParent1;
      OpenMM::OpenCLArray* MSpParent2;
      OpenMM::OpenCLArray* MSpgder;
      OpenMM::OpenCLArray* MSphder;
      OpenMM::OpenCLArray* MSpfms;
      OpenMM::OpenCLArray* MSpG0Large;
      OpenMM::OpenCLArray* MSpG0vdW;
      OpenMM::OpenCLArray* MSpGaussExponent;
      OpenMM::OpenCLArray* MSpGamma;
      OpenMM::OpenCLArray* MSpSelfVolume;
      OpenMM::OpenCLArray* MSgrad;
      OpenMM::OpenCLArray* MSsemaphor;
    };//class OpenCLMSParticle

    
private:
    const AGBNPForce *gvol_force;

    int numParticles;
    unsigned int version;
    bool useCutoff;
    bool usePeriodic;
    bool useExclusions;
    double cutoffDistance;
    double roffset;
    float common_gamma;
    int maxTiles;
    bool hasInitializedKernels;
    bool hasCreatedKernels;
    OpenMM::OpenCLContext& cl;
    const OpenMM::System& system;
    int ov_work_group_size; //thread group size
    int num_compute_units;
    
    OpenCLOverlapTree *gtree;   //tree of atomic overlaps
    OpenCLOverlapTree *gtreems; //tree of MS particles overlaps

    double solvent_radius; //solvent probe radius for AGBNP2

    OpenMM::OpenCLArray* radiusParam1;
    OpenMM::OpenCLArray* radiusParam2;
    OpenMM::OpenCLArray* gammaParam1;
    OpenMM::OpenCLArray* gammaParam2;
    OpenMM::OpenCLArray* ishydrogenParam;
    OpenMM::OpenCLArray* chargeParam;
    OpenMM::OpenCLArray* alphaParam;

    //C++ vectors corresponding to parameter buffers above
    vector<cl_float> radiusVector1; //enlarged radii
    vector<cl_float> radiusVector2; //vdw radii
    vector<cl_float> gammaVector1;  //gamma/radius_offset
    vector<cl_float> gammaVector2;  //-gamma/radius_offset
    vector<cl_float> chargeVector;  //charge
    vector<cl_float> alphaVector;   //alpha vdw parameter
    vector<cl_int> ishydrogenVector;

    OpenMM::OpenCLArray* testBuffer;
    
    OpenMM::OpenCLArray* radtypeScreened;
    OpenMM::OpenCLArray* radtypeScreener;
    
    OpenMM::OpenCLArray* selfVolume; //vdw radii
    OpenMM::OpenCLArray* selfVolumeLargeR; //large radii
    OpenMM::OpenCLArray* Semaphor;
    OpenMM::OpenCLArray* volScalingFactor;
    OpenMM::OpenCLArray* BornRadius;
    OpenMM::OpenCLArray* invBornRadius;
    OpenMM::OpenCLArray* invBornRadius_fp;
    OpenMM::OpenCLArray* GBDerY;
    OpenMM::OpenCLArray* GBDerBrU;
    OpenMM::OpenCLArray* GBDerU;
    OpenMM::OpenCLArray* VdWDerBrW;
    OpenMM::OpenCLArray* VdWDerW;
    OpenMM::OpenCLArray* grad;

    cl::Kernel resetBufferKernel;
    cl::Kernel resetOvCountKernel;
    cl::Kernel resetTree;
    cl::Kernel resetSelfVolumesKernel;
    cl::Kernel InitOverlapTreeKernel_1body_1;
    cl::Kernel InitOverlapTreeKernel_1body_2;

    cl::Kernel InitOverlapTreeCountKernel;
    int InitOverlapTreeCountKernel_first_nbarg;

    int reduceSelfVolumesSVArgIndex;
    int InitOverlapTreeKernel_1body_1_GaArgIndex;
    int InitOverlapTreeKernel_1body_1_GvArgIndex;
    int InitOverlapTreeCountKernelGaArgIndex;
    int InitOverlapTreeCountKernelGvArgIndex;
    int InitOverlapTreeKernelGaArgIndex;
    int InitOverlapTreeKernelGvArgIndex;
    int ComputeOverlapTree_1passGaArgIndex;
    int ComputeOverlapTree_1passGvArgIndex;
    int computeSelfVolumesGaArgIndex;
    
    
    cl::Kernel reduceovCountBufferKernel;
    
    cl::Kernel InitOverlapTreeKernel;
    int InitOverlapTreeKernel_first_nbarg;

    cl::Kernel ComputeOverlapTreeKernel;
    cl::Kernel ComputeOverlapTree_1passKernel;
    cl::Kernel computeSelfVolumesKernel;
    cl::Kernel reduceSelfVolumesKernel_tree;
    cl::Kernel reduceSelfVolumesKernel_buffer;
    cl::Kernel updateSelfVolumesForcesKernel;

    cl::Kernel resetTreeKernel;
    cl::Kernel SortOverlapTree2bodyKernel;
    cl::Kernel resetComputeOverlapTreeKernel;
    cl::Kernel ResetRescanOverlapTreeKernel;
    cl::Kernel InitRescanOverlapTreeKernel;
    cl::Kernel RescanOverlapTreeKernel;
    cl::Kernel RescanOverlapTreeGammasKernel_W;
    cl::Kernel InitOverlapTreeGammasKernel_1body_W;
    //cl::Kernel computeVolumeEnergyKernel;

    /* Gaussian atomic parameters */
    vector<float> gaussian_exponent;
    vector<float> gaussian_volume;
    OpenMM::OpenCLArray* GaussianExponent;
    OpenMM::OpenCLArray* GaussianVolume;
    OpenMM::OpenCLArray* GaussianExponentLargeR;
    OpenMM::OpenCLArray* GaussianVolumeLargeR;
    /* gamma parameters */
    vector<float> atomic_gamma;
    OpenMM::OpenCLArray* AtomicGamma;

    vector<int> atom_ishydrogen;
    
    int niterations;


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

    //flag to give up
    OpenMM::OpenCLArray* PanicButton;
    vector<cl_int> panic_button;
    cl::Buffer* pinnedPanicButtonBuffer;
    int* pinnedPanicButtonMemory;
    cl::Event downloadPanicButtonEvent;

    bool do_ms; //flag to turn off MS model if no MS particles
    OpenCLMSParticle *MSparticle1;
    OpenCLMSParticle *MSparticle2;
    cl::Kernel MSParticles1ResetKernel;
    cl::Kernel MSParticles1CountKernel;
    cl::Kernel MSParticles1CountReduceKernel;
    cl::Kernel MSParticles1StoreKernel;
    cl::Kernel MSParticles1VfreeKernel;
    cl::Kernel MSParticles1VolsKernel;
    cl::Kernel MSParticles2CountKernel;
    cl::Kernel MSInitOverlapTreeVdW_1body_1Kernel;
    cl::Kernel MSInitRescanOverlapTreeLargeR_1body_1Kernel;
    cl::Kernel MSresetTreeKernel;
    cl::Kernel MSResetTreeCountKernel;
    cl::Kernel MSInitOverlapTreeCountKernel;
    cl::Kernel MSreduceovCountBufferKernel;
    cl::Kernel MSInitOverlapTreeKernel;
    cl::Kernel MSresetComputeOverlapTreeKernel;
    cl::Kernel MSComputeOverlapTree_1passKernel;
    cl::Kernel MSresetBufferKernel;
    cl::Kernel MSresetSelfVolumesKernel;
    cl::Kernel MScomputeSelfVolumesKernel;
    cl::Kernel MSreduceSelfVolumesKernel_buffer;
    cl::Kernel MSinitEnergyBufferKernel;
    cl::Kernel MSupdateEnergyBufferKernel;
    cl::Kernel MSupdateSelfVolumesForcesKernel;
    cl::Kernel MSaddSelfVolumesKernel;
    cl::Kernel MSaddSelfVolumesFromLongKernel;
    cl::Kernel MSResetRescanOverlapTreeKernel;
    cl::Kernel MSInitRescanOverlapTreeKernel;
    cl::Kernel MSRescanOverlapTreeKernel;
    
    OpenMM::OpenCLArray* MScount1;
    OpenMM::OpenCLArray* MScount2;
    
    /*
    OpenMM::OpenCLArray* test_input_buffer;
    cl::Kernel TestScanWarpKernel;
    */
};


 //class to record MS particles
 class MSParticle {
 public:
   double vol;
   double vol_large;
   double vol_vdw;
   double vol0;
   double ssp_large;
   double ssp_vdw;
   RealVec pos;
   int parent1;
   int parent2;
   RealVec gder;//used for volume derivatives
   RealVec hder;//used for positional derivatives
   double fms;
   double G0_vdw; //accumulator for derivatives
   double G0_large;
 };

 
} // namespace AGBNPPlugin

#endif /*OPENCL_AGBNP_KERNELS_H_*/
