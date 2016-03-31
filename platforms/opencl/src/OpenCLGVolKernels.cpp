/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GVol                            *
 * -------------------------------------------------------------------------- */

#include "OpenCLGVolKernels.h"
#include "OpenCLGVolKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/opencl/OpenCLNonbondedUtilities.h"
#include "openmm/opencl/OpenCLForceInfo.h"
#include <cmath>
#include <cfloat>

#include <fstream>
#include <iomanip>

//conversion factors
#define ANG (0.1f)
#define ANG3 (0.001f)

//radius offset
#define SA_DR (0.1*ANG)

//volume cutoffs in switching function
#define MIN_GVOL (FLT_MIN)
#define VOLMINA (0.01f*ANG3)
#define VOLMINB (0.1f*ANG3)

#define PI (3.14159265359)

// conversion factors from spheres to Gaussians
#define KFC (2.2269859253)

// minimum overlap volume to count
#define MAX_ORDER (12)
// TBF to-be-fixed
#define MAX_OVERLAPS_PER_ATOM (512)

using namespace GVolPlugin;
using namespace OpenMM;
using namespace std;

class OpenCLGVolForceInfo : public OpenCLForceInfo {
public:
  OpenCLGVolForceInfo(int requiredBuffers, const GVolForce& force) : OpenCLForceInfo(requiredBuffers), force(force) {
    }
    int getNumParticleGroups() {
      return force.getNumParticles();//each particle is in a different group?
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
      particles.push_back(index);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return (group1 == group2);
    }
private:
    const GVolForce& force;
};

OpenCLCalcGVolForceKernel::~OpenCLCalcGVolForceKernel() {
  //    if (params != NULL)
  //   delete params;
}



static int _nov_ = 0;


int OpenCLCalcGVolForceKernel::copy_tree_to_device(void){

  int padsize = cl.getPaddedNumAtoms();
  vector<cl_int> nn(padsize);
  int offset;
  
  offset = 0;
  for(int i = 0; i < padsize ; i++){
    tree_pointer.push_back(offset);
    nn[i] = (cl_int) offset;
    offset += padded_tree_size[i];
  }
  ovAtomTreePointer->upload(nn);

  for(int i = 0; i < padsize ; i++){
    nn[i] = (cl_int) 0;
  }
  ovAtomTreeSize->upload(nn);

  for(int i = 0; i < padsize ; i++){
    nn[i] = (cl_int) padded_tree_size[i];
  }
  ovAtomTreePaddedSize->upload(nn);

  return 1;
}

void OpenCLCalcGVolForceKernel::initialize(const System& system, const GVolForce& force) {
    if (cl.getPlatformData().contexts.size() > 1)
      throw OpenMMException("GVolForce does not support using multiple OpenCL devices");
    OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
    int elementSize = (cl.getUseDoublePrecision() ? sizeof(cl_double) : sizeof(cl_float));   
    numParticles = force.getNumParticles();
    if (numParticles == 0)
        return;
    radiusParam1 = new OpenCLArray(cl, cl.getPaddedNumAtoms(), sizeof(cl_float), "radiusParam1");
    radiusParam2 = new OpenCLArray(cl, cl.getPaddedNumAtoms(), sizeof(cl_float), "radiusParam2");
    gammaParam1 = new OpenCLArray(cl, cl.getPaddedNumAtoms(), sizeof(cl_float), "gammaParam1");
    gammaParam2 = new OpenCLArray(cl, cl.getPaddedNumAtoms(), sizeof(cl_float), "gammaParam2");
    ishydrogenParam = new OpenCLArray(cl, cl.getPaddedNumAtoms(), sizeof(cl_int), "ishydrogenParam");

    bool useLong = cl.getSupports64BitGlobalAtomics();

    // this the accumulation buffer for overlap atom-level data (self-volumes, etc.)
    // note that each thread gets a separate buffer of size Natoms (rather than each thread block as in the
    // non-bonded algorithm). This may limits the max number of atoms.

    //cl.addAutoclearBuffer(*ovAtomBuffer);

    vector<cl_float> radiusVector1(cl.getPaddedNumAtoms());
    vector<cl_float> radiusVector2(cl.getPaddedNumAtoms());
    vector<cl_float> gammaVector1(cl.getPaddedNumAtoms());
    vector<cl_float> gammaVector2(cl.getPaddedNumAtoms());
    vector<cl_int> ishydrogenVector(cl.getPaddedNumAtoms());
    for (int i = 0; i < numParticles; i++) {
      double radius, gamma;
      bool ishydrogen;
      force.getParticleParameters(i, radius, gamma, ishydrogen);
	radiusVector1[i] = (cl_float) radius;
	radiusVector2[i] = (cl_float) (radius - SA_DR);
	gammaVector1[i] = (cl_float) gamma/SA_DR;
	gammaVector2[i] = (cl_float) (-gamma/SA_DR);
	ishydrogenVector[i] = ishydrogen ? 1 : 0;
    }
    radiusParam1->upload(radiusVector1);
    radiusParam2->upload(radiusVector2);
    gammaParam1->upload(gammaVector1);
    gammaParam2->upload(gammaVector2);
    ishydrogenParam->upload(ishydrogenVector);

    useCutoff = (force.getNonbondedMethod() != GVolForce::NoCutoff);
    usePeriodic = (force.getNonbondedMethod() != GVolForce::NoCutoff && force.getNonbondedMethod() != GVolForce::CutoffNonPeriodic);
    useExclusions = false;
    cutoffDistance = force.getCutoffDistance();

    niterations = 0;
}

double OpenCLCalcGVolForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
  bool useLong = cl.getSupports64BitGlobalAtomics();
  bool verbose = false;

  niterations += 1;

  if (!hasCreatedKernels) {
    hasCreatedKernels = true;


    {
      //init overlap trees

      /* padded tree sizes */
      total_tree_size = 0;
      tree_size.clear();
      padded_tree_size.clear();
      tree_pointer.clear();
      int size = MAX_OVERLAPS_PER_ATOM;
      int TileSize =  OpenCLContext::TileSize;
      int npadsize = TileSize*((size+TileSize-1)/TileSize);
      for(int i = 0; i<cl.getPaddedNumAtoms();i++){
	tree_size.push_back(0);
	padded_tree_size.push_back(npadsize);
	total_tree_size += npadsize;
      }

      //maximum number of tree reduction blocks so that
      //each section is no smaller than 1024 slots
      int num_reduction_blocks_max = 8; //total_tree_size/nb.getForceThreadBlockSize() + 1;
      //pick the largest of the default number of blocks and the max above
      num_reduction_blocks = nb.getNumForceBuffers();
      if(num_reduction_blocks > num_reduction_blocks_max) num_reduction_blocks = num_reduction_blocks_max;
      
      //now make total tree size allocated on device 
      //a multiple of the reduction blocks*tile size to ensure
      //that each section of the tree is a multiple of work group size
      
      //int n = num_reduction_blocks*nb.getForceThreadBlockSize();
      //total_tree_size = n*((total_tree_size+n-1)/n);

      if(verbose)
	std::cout << "Tree size: " << total_tree_size << std::endl;

      //now copy overlap tree arrays to device
      ovAtomTreePointer = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomTreePointer");
      ovAtomTreeSize = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomTreeSize");
      ovAtomTreePaddedSize = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomTreePaddedSize");
      ovAtomTreeLock = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomTreeLock");
      ovLevel = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovLevel");
      ovG = OpenCLArray::create<mm_float4>(cl, total_tree_size, "ovG"); //gaussian position + exponent
      ovVolume = OpenCLArray::create<cl_float>(cl, total_tree_size, "ovVolume");
      ovVSfp = OpenCLArray::create<cl_float>(cl, total_tree_size, "ovVSfp");
      ovSelfVolume = OpenCLArray::create<cl_float>(cl, total_tree_size, "ovSelfVolume");
      ovGamma1i = OpenCLArray::create<cl_float>(cl, total_tree_size, "ovGamma1i");
      ovDV1 = OpenCLArray::create<mm_float4>(cl, total_tree_size, "ovDV1"); //dV12/dr1 + dV12/dV1
      ovDV2 = OpenCLArray::create<mm_float4>(cl, total_tree_size, "ovDV2"); //dPsi12/dr2

      ovLastAtom = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovLastAtom");
      ovRootIndex = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovRootIndex");
      ovChildrenStartIndex = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovChildrenStartIndex");
      ovChildrenCount = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovChildrenCount");
      ovProcessedFlag = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovProcessedFlag");
      ovOKtoProcessFlag = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovOKtoProcessFlag");
      ovChildrenReported = OpenCLArray::create<cl_int>(cl, total_tree_size, "ovChildrenReported");

      /* reduction buffer holds volume (in w) and derivatives (in xyz) */
      ovAtomBuffer = OpenCLArray::create<mm_float4>(cl, cl.getPaddedNumAtoms()*num_reduction_blocks, "ovAtomBuffer");

      //atomic parameters
      GaussianExponent = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(), "GaussianExponent");
      GaussianVolume = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(), "GaussianVolume");
      AtomicGamma = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(),"AtomicGamma");

      OpenCLCalcGVolForceKernel::copy_tree_to_device();
    }

    {
      //Reset tree kernel
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      
      map<string, string> replacements;
      string file, kernel_name;
      cl::Program program;
      int index;
      cl::Kernel kernel;

      kernel_name = "resetTree";
      file = cl.replaceStrings(OpenCLGVolKernelSources::GVolResetTree, replacements);
      program = cl.createProgram(file, defines);

      //reset tree kernel
      resetTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = resetTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovSelfVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV2->getDeviceBuffer());

      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenReported->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeLock->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ishydrogenParam->getDeviceBuffer());

      //reset buffer kernel
      kernel_name = "resetBuffer";
      resetBufferKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = resetBufferKernel;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_reduction_blocks);
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      
      //reset tree counters kernel
      kernel_name = "resetSelfVolumes";
      resetSelfVolumesKernel = cl::Kernel(program, kernel_name.c_str());
      kernel = resetSelfVolumesKernel;
      index = 0;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenReported->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovSelfVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV2->getDeviceBuffer());
    }


    {
      //Tree construction 
      cl::Program program;
      cl::Kernel kernel;
      string kernel_name;
      int index;

      //pass 1
      map<string, string> pairValueDefines;
      if (useCutoff)
	pairValueDefines["USE_CUTOFF"] = "1";
      if (usePeriodic)
	pairValueDefines["USE_PERIODIC"] = "1";
      pairValueDefines["USE_EXCLUSIONS"] = "1";
      pairValueDefines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      pairValueDefines["CUTOFF"] = cl.doubleToString(cutoffDistance);
      pairValueDefines["CUTOFF_SQUARED"] = cl.doubleToString(cutoffDistance*cutoffDistance);
      pairValueDefines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      pairValueDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      pairValueDefines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      pairValueDefines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      int numExclusionTiles = nb.getExclusionTiles().getSize();
      pairValueDefines["NUM_TILES_WITH_EXCLUSIONS"] = cl.intToString(numExclusionTiles);
      int numContexts = cl.getPlatformData().contexts.size();
      int startExclusionIndex = cl.getContextIndex()*numExclusionTiles/numContexts;
      int endExclusionIndex = (cl.getContextIndex()+1)*numExclusionTiles/numContexts;
      pairValueDefines["FIRST_EXCLUSION_TILE"] = cl.intToString(startExclusionIndex);
      pairValueDefines["LAST_EXCLUSION_TILE"] = cl.intToString(endExclusionIndex);

      map<string, string> replacements;

      replacements["KFC"] = cl.doubleToString((double)KFC);
      replacements["VOLMINA"] = cl.doubleToString((double)VOLMINA);
      replacements["VOLMINB"] = cl.doubleToString((double)VOLMINB);
      replacements["MIN_GVOL"] = cl.doubleToString((double)MIN_GVOL);

      replacements["ATOM_PARAMETER_DATA"] = 
	"real4 g; \n"
	"real  v; \n"
	"real  gamma; \n";

      replacements["PARAMETER_ARGUMENTS"] = "";

      replacements["INIT_VARS"] = "";

      replacements["LOAD_ATOM1_PARAMETERS"] = 
	"real a1 = global_gaussian_exponent[atom1]; \n"
	"real v1 = global_gaussian_volume[atom1];\n"
	"real gamma1 = global_atomic_gamma[atom1];\n";

      replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] =
	"localData[localAtomIndex].g.w = a1;\n"
	"localData[localAtomIndex].v = v1;\n"
	"localData[localAtomIndex].gamma = gamma1;\n";

      replacements["LOAD_ATOM2_PARAMETERS"] = 
	"real a2 = localData[localAtom2Index].g.w;\n"
	"real v2 = localData[localAtom2Index].v;\n"
	"real gamma2 = localData[localAtom2Index].gamma;\n";

      replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = 
	"localData[localAtomIndex].g.w = global_gaussian_exponent[j];\n"
	"localData[localAtomIndex].v = global_gaussian_volume[j];\n"
	"localData[localAtomIndex].gamma = global_atomic_gamma[j];\n";

      //this kernel proably perform better with group_size = warp_size
      //replacements["SYNC_WARPS"] = "";//no need for group barriers if group_size = warp_size
      replacements["WORK_GROUP_SIZE"] = cl.intToString(OpenCLContext::TileSize);

      replacements["ACQUIRE_LOCK"] =
	 "if(tgx == 0){//first atom in warp \n"
	"	//acquire tree lock for the tile\n"
	"	//If lock is not acquired all of the threads in the warp will wait\n"
	"	GetSemaphor(&ovAtomTreeLock[atom1]);\n"
	"      }\n";


      replacements["RELEASE_LOCK"] =
	"if(tgx==0){//first thread in warp\n"
	"  //release lock\n"
	"  ReleaseSemaphor(&ovAtomTreeLock[atom1]);\n"
	"}\n";

      replacements["COMPUTE_INTERACTION"] =
		"	real a12 = a1 + a2; \n"
		"	real deltai = 1./a12; \n"
	        "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
		"	real gvol = (v1*v2/powr(PI/df,1.5f))*ef; \n"
                "       real dgvol = -2.0f*df*gvol; \n"
                "       real dgvolv = v1 > 0 ? gvol/v1 : 0; \n" 
		"	real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "       //switching function \n"
                "       real s, sp; \n"
                "       if(gvol > VolMinB ){ \n"
                "           s = 1.0f; \n"
                "           sp = 0.0f; \n"
                "       }else if(gvol < VolMinA){ \n"
                "           s = 0.0f; \n"
                "           sp = 0.0f; \n"
                "       }else{ \n"
                "         real swd = 1.f/( VolMinB - VolMinA ); \n"
                "         real swu = (gvol - VolMinA)*swd; \n"
                "         real swu2 = swu*swu; \n"
                "         real swu3 = swu*swu2; \n"
                "         s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "         sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
	        "       }\n"
                "       // switching function end \n"
                "	real sfp = sp*gvol + s; \n"
                "       gvol = s*gvol; \n"
		"	/* at this point have:\n"
		"	     1. gvol: overlap  between atom1 and atom2\n"
		"	     2. a12: gaussian exponent of overlap\n"
		"	     3. v12=gvol: volume of overlap\n"
		"	     4. c12: gaussian center of overlap\n"
		"	     These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
		"	     volume is large enough.\n"
		"	 */\n"
		"	 if(gvol > Min_GVol ){\n"
		"	   int pp = ovAtomTreePointer[atom1]; //parent\n"
		"	   ovChildrenStartIndex[pp] = pp+1; //point parent to first child\n"
		"	   atomic_inc(&ovChildrenCount[pp]);     //increment children count\n"
		"	   int endslot = pp + ovAtomTreeSize[atom1];\n"
		"	   ovLevel[endslot] = 2; //two-body\n"
		"	   ovVolume[endslot] = gvol;\n"
	        "          ovVSfp[endslot] = sfp; \n"
		"	   ovGamma1i[endslot] = gamma1 + gamma2;\n"
		"	   ovLastAtom[endslot] = atom2;\n"
		"	   ovRootIndex[endslot] = pp;\n"
		"	   ovChildrenStartIndex[endslot] = -1;\n"
		"	   ovChildrenCount[endslot] = 0;\n"
		"	   ovG[endslot] = (real4)(c12.xyz, a12);\n"
	        "          ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
                "          atomic_inc(&ovAtomTreeSize[atom1]);\n"
		"         }\n";
      string InitOverlapTreeSrc = cl.replaceStrings(OpenCLGVolKernelSources::GVolOverlapTree, replacements);


      kernel_name = "InitOverlapTree_1body";
      replacements["KERNEL_NAME"] = kernel_name;

      program = cl.createProgram(InitOverlapTreeSrc, pairValueDefines);
      InitOverlapTreeKernel_1body_1 = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = InitOverlapTreeKernel_1body_1;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, radiusParam1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, gammaParam1->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ishydrogenParam->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianVolume->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());

      program = cl.createProgram(InitOverlapTreeSrc, pairValueDefines);
      InitOverlapTreeKernel_1body_2 = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = InitOverlapTreeKernel_1body_2;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, radiusParam2->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, gammaParam2->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ishydrogenParam->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianVolume->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());


      kernel_name = "InitOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      InitOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());

      index = 0;
      kernel = InitOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeLock->getDeviceBuffer());

      kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianVolume->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer() );

      kernel.setArg<cl::Buffer>(index++, nb.getExclusions().getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, nb.getExclusionTiles().getDeviceBuffer());
      if (useCutoff) {
        kernel.setArg<cl::Buffer>(index++, nb.getInteractingTiles().getDeviceBuffer());
        kernel.setArg<cl::Buffer>(index++, nb.getInteractionCount().getDeviceBuffer());
        index += 5; // The periodic box size arguments are set when the kernel is executed.
        kernel.setArg<cl_uint>(index++, nb.getInteractingTiles().getSize());
        kernel.setArg<cl::Buffer>(index++, nb.getBlockCenters().getDeviceBuffer());
        kernel.setArg<cl::Buffer>(index++, nb.getBlockBoundingBoxes().getDeviceBuffer());
        kernel.setArg<cl::Buffer>(index++, nb.getInteractingAtoms().getDeviceBuffer());
      }else{
	kernel.setArg<cl_uint>(index++, cl.getNumAtomBlocks()*(cl.getNumAtomBlocks()+1)/2);
      }
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());


      //pass 2
      kernel_name = "ComputeOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;

      ComputeOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = ComputeOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeLock->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianVolume->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());

      //pass 2
      kernel_name = "RescanOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;

      RescanOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = RescanOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeLock->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, GaussianVolume->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer() );
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());


      //2-body volumes sort kernel
      kernel_name = "SortOverlapTree2body";
      replacements["KERNEL_NAME"] = kernel_name;

      SortOverlapTree2bodyKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = SortOverlapTree2bodyKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
    }


    {
      //Self volumes kernel
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      
      map<string, string> replacements;
      string kernel_name = "computeSelfVolumes";

      string file = cl.replaceStrings(OpenCLGVolKernelSources::GVolSelfVolume, replacements);
      //ofstream output("tmp_kernel.cl");
      //output << file << std::endl;
      //output.close();

      cl::Program program = cl.createProgram(file, defines);

      computeSelfVolumesKernel = cl::Kernel(program, kernel_name.c_str());
      cl::Kernel kernel = computeSelfVolumesKernel;

      int index = 0;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());

      kernel.setArg<cl::Buffer>(index++, GaussianExponent->getDeviceBuffer() );

      kernel.setArg<cl::Buffer>(index++, ovLevel->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovVSfp->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovG->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovSelfVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV1->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV2->getDeviceBuffer());

      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenReported->getDeviceBuffer());
    }

    {
      //Self volumes reduction kernel (pass 1)
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      defines["NTILES_IN_BLOCK"] = "1";//cl.intToString(nb.getForceThreadBlockSize()/OpenCLContext::TileSize);

      map<string, string> replacements;
      string kernel_name = "reduceSelfVolumes_tree";
      
      string file = OpenCLGVolKernelSources::GVolReduceTree;
      cl::Program program = cl.createProgram(file, defines);
      reduceSelfVolumesKernel_tree = cl::Kernel(program, kernel_name.c_str());
      cl::Kernel kernel = reduceSelfVolumesKernel_tree;
      int index = 0;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_reduction_blocks);
      kernel.setArg<cl_int>(index++, total_tree_size);
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovSelfVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV2->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
    }

    {
      //Self volumes reduction kernel (pass 2)
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      defines["NTILES_IN_BLOCK"] = "1";//cl.intToString(nb.getForceThreadBlockSize()/OpenCLContext::TileSize);

      map<string, string> replacements;
      string kernel_name = "reduceSelfVolumes_buffer";
      
      string file = OpenCLGVolKernelSources::GVolReduceTree;
      cl::Program program = cl.createProgram(file, defines);
      reduceSelfVolumesKernel_buffer = cl::Kernel(program, kernel_name.c_str());
      cl::Kernel kernel = reduceSelfVolumesKernel_buffer;
      int index = 0;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_reduction_blocks);
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovGamma1i->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, (useLong ? cl.getLongForceBuffer().getDeviceBuffer() : cl.getForceBuffers().getDeviceBuffer())); //master force buffer
      kernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());
    }
  }


  if(verbose){
    std::cout <<  "Num atoms: " << cl.getNumAtoms() << std::endl;
    std::cout <<  "Padded Num Atoms: " << cl.getPaddedNumAtoms() << std::endl;
    std::cout <<  "Num Blocks: " << cl.getNumAtomBlocks() << std::endl;
    
    
    std::cout <<  "Tile size: " << OpenCLContext::TileSize << std::endl;
    std::cout <<  "getNumForceThreadBlocks: " << nb.getNumForceThreadBlocks() << std::endl;
    std::cout <<  "NumReductionBlocks: " << num_reduction_blocks << std::endl;
    std::cout <<  "Force Work Group Size: " << nb.getForceThreadBlockSize() << std::endl;

    std::cout <<  "Tree Size: " <<  total_tree_size << std::endl;
  }

  if(false){
    vector<cl_int> processed(total_tree_size);
    vector<cl_int> oktoprocess(total_tree_size);
    vector<cl_int> children_start(total_tree_size);
    vector<cl_int> children_count(total_tree_size);
    vector<cl_int> parent_index(total_tree_size);
    ovProcessedFlag->download(processed);
    ovOKtoProcessFlag->download(oktoprocess);
    ovChildrenStartIndex->download(children_start);
    ovChildrenCount->download(children_count);
    ovRootIndex->download(parent_index);
    std::cout << "Before:" << std::endl;
    for(int i=0;i<total_tree_size;i++){
      std::cout << i << " " << oktoprocess[i] << processed[i] << " " << children_start[i] << " " << children_count[i]  << " " << parent_index[i] <<std::endl;
    }
  }

  cl.executeKernel(resetTreeKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
  cl.executeKernel(resetBufferKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
  cl.executeKernel(InitOverlapTreeKernel_1body_1, cl.getPaddedNumAtoms());
  cl.executeKernel(InitOverlapTreeKernel, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
  cl.executeKernel(SortOverlapTree2bodyKernel, cl.getPaddedNumAtoms());
  cl.executeKernel(ComputeOverlapTreeKernel, cl.getPaddedNumAtoms(), OpenCLContext::TileSize);
  cl.executeKernel(resetSelfVolumesKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
  cl.executeKernel(computeSelfVolumesKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
  cl.executeKernel(reduceSelfVolumesKernel_tree, nb.getForceThreadBlockSize()*num_reduction_blocks, nb.getForceThreadBlockSize());
  cl.executeKernel(reduceSelfVolumesKernel_buffer, cl.getPaddedNumAtoms());
    
    
  //accumulated self volumes
  if(true){
    float mol_volume = 0.0;
    vector<mm_float4> dv2(num_reduction_blocks*cl.getPaddedNumAtoms());
    ovAtomBuffer->download(dv2);
    
    std::cout << "Self Volumes:" << std::endl;
    for(int iat = 0; iat < numParticles; iat++){
      std::cout << iat << " " << dv2[iat].w << std::endl;
      mol_volume += dv2[iat].w;
    }
    std::cout << "Volume (from self volumes):" << mol_volume <<std::endl;
  }

  
  if(true){
    //rescan for energy with reduced radii
    cl.executeKernel(resetBufferKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
    cl.executeKernel(InitOverlapTreeKernel_1body_2, cl.getPaddedNumAtoms());
    cl.executeKernel(RescanOverlapTreeKernel, cl.getPaddedNumAtoms(), OpenCLContext::TileSize);
    cl.executeKernel(resetSelfVolumesKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
    cl.executeKernel(computeSelfVolumesKernel, OpenCLContext::TileSize*num_reduction_blocks, OpenCLContext::TileSize);
    cl.executeKernel(reduceSelfVolumesKernel_tree, nb.getForceThreadBlockSize()*num_reduction_blocks, nb.getForceThreadBlockSize());
    cl.executeKernel(reduceSelfVolumesKernel_buffer, cl.getPaddedNumAtoms());
  }


  if(false){
    float self_volume = 0.0;
    vector<cl_float> self_volumes(total_tree_size);
    vector<cl_float> volumes(total_tree_size);
    vector<cl_float> gammas(total_tree_size);
    vector<cl_int> last_atom(total_tree_size);
    vector<cl_int> level(total_tree_size);
    vector<cl_int> parent(total_tree_size);
    vector<cl_int> children_start_index(total_tree_size);
    vector<cl_int> children_count(total_tree_size);
    vector<mm_float4> g(total_tree_size);
    vector<mm_float4> dv2(total_tree_size);
    vector<cl_int> tree_size(cl.getPaddedNumAtoms());
    vector<cl_float> sfp(total_tree_size);

    ovSelfVolume->download(self_volumes);
    ovVolume->download(volumes);
    ovVolume->download(volumes);
    ovLevel->download(level);
    ovLastAtom->download(last_atom);
    ovRootIndex->download(parent);
    ovChildrenStartIndex->download(children_start_index);
    ovChildrenCount->download(children_count);
    ovAtomTreeSize->download(tree_size);
    ovG->download(g);
    ovGamma1i->download(gammas);
    ovDV2->download(dv2);
    ovVSfp->download(sfp);

    std::cout << "Tree After:" << std::endl;
    for(int iat = 0; iat < cl.getPaddedNumAtoms(); iat++){
    std::cout << "Tree for atom: " << iat << " " << " size= " << tree_size[iat] << std::endl;
      int pp = tree_pointer[iat];
      int np = padded_tree_size[iat];
      self_volume += self_volumes[pp];
      std::cout << "slot level LastAtom parent ChStart ChCount SelfV V gamma a x y z dedx dedy dedz sfp" << endl;
      for(int i = pp; i < pp + np ; i++){
	std::cout << std::setprecision(4) << std::setw(6) << i << " "  << std::setw(7) << (int)level[i] << " " << std::setw(7) << (int)last_atom[i] << " " << std::setw(7) << (int)parent[i] << " "  << std::setw(7) << (int)children_start_index[i] << " " << std::setw(7) <<  (int)children_count[i] << " " << std::setw(15) << (float)self_volumes[i] << " " << std::setw(10) << (float)volumes[i]  << " " << std::setw(10) << (float)gammas[i] << " " << std::setw(10) << g[i].w << " " << std::setw(10) << g[i].x << " " << std::setw(10) << g[i].y << " " << std::setw(10) << g[i].z << std::setw(10) << dv2[i].x << " " << std::setw(10) << dv2[i].y << " " << std::setw(10) << dv2[i].z << " " << sfp[i] << std::endl;
      }
    }
    std::cout << "Volume (from self volumes):" << self_volume <<std::endl;
  }

  //accumulated self volumes
  if(false){
    float mol_volume = 0.0;
    vector<mm_float4> dv2(num_reduction_blocks*cl.getPaddedNumAtoms());
    ovAtomBuffer->download(dv2);

    std::cout << "Self Volumes:" << std::endl;
    for(int iat = 0; iat < numParticles; iat++){
      std::cout << iat << " " << dv2[iat].w << std::endl;
      mol_volume += dv2[iat].w;
    }
    std::cout << "Volume (from self volumes):" << mol_volume <<std::endl;
  }


  
  return 0.0;
}

void OpenCLCalcGVolForceKernel::copyParametersToContext(ContextImpl& context, const GVolForce& force) {
    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*force.getNumParticles()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*force.getNumParticles()/numContexts;
    if (numParticles != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numParticles == 0)
        return;
    
    // Record the per-bond parameters.
    
    //vector<mm_float2> paramVector(numBonds);
    //for (int i = 0; i < numBonds; i++) {
    //    int atom1, atom2;
    //   double length, k;
    //   force.getBondParameters(startIndex+i, atom1, atom2, length, k);
    //    paramVector[i] = mm_float2((cl_float) length, (cl_float) k);
    // }
    //params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cl.invalidateMolecules();
}

