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
#include <algorithm>

#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/RealVec.h"
#include "gaussvol.h"

//conversion factors
#define ANG (0.1f)
#define ANG3 (0.001f)

//radius offset
#define SA_DR (0.1*ANG)

//volume cutoffs in switching function
#define MIN_GVOL (FLT_MIN)
#define VOLMIN0 (0.009f*ANG3)
#define VOLMINA (0.01f*ANG3)
#define VOLMINB (0.1f*ANG3)

#define PI (3.14159265359)

// conversion factors from spheres to Gaussians
#define KFC (2.2269859253)

// minimum overlap volume to count
#define MAX_ORDER (12)

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

void init_tree_size(int num_atoms, int num_sections, int TileSize,
		    vector<int>& noverlaps, vector<int>& noverlaps_2body,
		    int& total_tree_size, 
		    vector<int>& tree_size, vector<int>& padded_tree_size, vector<int>& tree_pointer){
  
  total_tree_size = 0;
  tree_size.clear();
  padded_tree_size.clear();
  tree_pointer.clear();

  //max size
  int natoms_per_section = num_atoms/num_sections; //assume divisible
  int max_size = 0;
  int offset = 0;
  for(int section = 0; section < num_sections; section++, offset += natoms_per_section){
    int nn = 0;
    for(int i = 0; i < natoms_per_section ; i++){
      int iat = offset + i;
      nn += noverlaps[iat];
    }
    if(nn>max_size) max_size = nn;
  }
  // double estimate
  int size = max_size * 2;
  // now pad
  int npadsize = TileSize*((size+TileSize-1)/TileSize);
  
  tree_size.resize(num_sections);
  padded_tree_size.resize(num_sections);
  tree_pointer.resize(num_atoms);
  offset = 0;
  for(int section = 0; section < num_sections; section++, offset += natoms_per_section){
    tree_size[section] = 0;
    padded_tree_size[section] = npadsize;
    for(int i = 0; i < natoms_per_section; i++){
      int iat = offset + i;
      int slot = section*npadsize + i;
      tree_pointer[iat] = slot;
    }
    total_tree_size += npadsize;
  }
}
		    


int OpenCLCalcGVolForceKernel::copy_tree_to_device(void){

  int padsize = cl.getPaddedNumAtoms();
  vector<cl_int> nn(padsize);
  int num_blocks = cl.getNumAtomBlocks();
  vector<cl_int> ns(num_blocks);
  int offset;
  
  offset = 0;
  for(int i = 0; i < padsize ; i++){
    tree_pointer.push_back(offset);
    nn[i] = (cl_int) tree_pointer[i];
  }
  ovAtomTreePointer->upload(nn);

  for(int i = 0; i < num_blocks ; i++){
    ns[i] = (cl_int) tree_size[i];
  }
  ovAtomTreeSize->upload(ns);

  for(int i = 0; i < num_blocks ; i++){
    ns[i] = (cl_int) padded_tree_size[i];
  }
  ovAtomTreePaddedSize->upload(ns);

  return 1;
}

void OpenCLCalcGVolForceKernel::initialize(const System& system, const GVolForce& force) {
    if (cl.getPlatformData().contexts.size() > 1)
      throw OpenMMException("GVolForce does not support using multiple OpenCL devices");
    
    OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
    int elementSize = (cl.getUseDoublePrecision() ? sizeof(cl_double) : sizeof(cl_float));   

    numParticles = cl.getNumAtoms();//force.getNumParticles();
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

    gvol_force = &force;
    niterations = 0;
    hasCreatedKernels = false;
}

double OpenCLCalcGVolForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
  bool useLong = cl.getSupports64BitGlobalAtomics();
  bool verbose = false;

  niterations += 1;

  if (!hasCreatedKernels) {
    hasCreatedKernels = true;


    {
      //run CPU version once to estimate sizes
      GaussVol *gvol;
      std::vector<RealVec> positions;
      std::vector<bool> ishydrogen;
      std::vector<RealOpenMM> radii;
      std::vector<RealOpenMM> gammas;
      //outputs
      RealOpenMM surf_energy;
      std::vector<RealOpenMM> free_volume, self_volume, surface_areas;
      std::vector<RealVec> surf_force;
      int numParticles = cl.getNumAtoms();
      //input lists
      positions.resize(numParticles);
      radii.resize(numParticles);
      gammas.resize(numParticles);
      ishydrogen.resize(numParticles);
      //output lists
      free_volume.resize(numParticles);
      self_volume.resize(numParticles);
      surface_areas.resize(numParticles);
      surf_force.resize(numParticles);
      for (int i = 0; i < numParticles; i++){
	double r, g;
	bool h;
	gvol_force->getParticleParameters(i, r, g, h);
	radii[i] = r;
	gammas[i] = g;
	ishydrogen[i] = h;
      }
      gvol = new GaussVol(numParticles, radii, gammas, ishydrogen);
      vector<mm_float4> posq; 
      cl.getPosq().download(posq);
      for(int i=0;i<numParticles;i++){
 	positions[i] = RealVec((RealOpenMM)posq[i].x,(RealOpenMM)posq[i].y,(RealOpenMM)posq[i].z);
      }
      int init = 0;

      gvol->enerforc(init, positions, surf_energy, surf_force, free_volume, self_volume, surface_areas);
      vector<int> noverlaps(cl.getPaddedNumAtoms());
      vector<int> noverlaps_2body(cl.getPaddedNumAtoms());
      for(int i = 0; i<cl.getPaddedNumAtoms(); i++) noverlaps[i] = 0;
      for(int i = 0; i<cl.getPaddedNumAtoms(); i++) noverlaps_2body[i] = 0;
      gvol->getstat(noverlaps, noverlaps_2body);

      //compute maximum number of 2-body overlaps
      int nn = 0, nn2 = 0;
      for(int i = 0; i < noverlaps_2body.size(); i++){
	nn += noverlaps[i];
	nn2 += noverlaps_2body[i];
      }
      if(verbose) cout << "Number of overlaps: " << nn << endl;
      if(verbose) cout << "Number of 2-body overlaps: " << nn2 << endl;
      
      //compute maximum number of 2-body overlaps
      num_twobody_max = 0;
      for(int i = 0; i < noverlaps_2body.size(); i++){
	if(noverlaps_2body[i] > num_twobody_max) num_twobody_max = noverlaps_2body[i];
      }
      // double estimate
      num_twobody_max *= 2;//1.5f;
      //pad mod 8
      num_twobody_max = 8*((num_twobody_max+8-1)/8);
      if(verbose) cout << "Max2Body: " << num_twobody_max << endl;

      cout << "Device: " << cl.getDevice().getInfo<CL_DEVICE_NAME>()  << endl;
      cout << "MaxSharedMem: " << cl.getDevice().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()  << endl;
      cout << "CompUnits: " << cl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()  << endl;
      cout << "Max Work Group Size: " << cl.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()  << endl;
      cout << "Supports 64bit Atomics: " << useLong << endl;

      num_sections = cl.getNumAtomBlocks(); //number of tree sections
      if(verbose) cout << "Num Tree Sections: " << num_sections  << endl;
      //this is the group size for the 3-body etc. tree construction kernel
      int max_threads = nb.getForceThreadBlockSize()*nb.getNumForceThreadBlocks();
      if(verbose) cout << "Max Threads: " << max_threads  << endl;
      ov_work_group_size = max_threads/num_sections;
      //reduce down to a multiple of TileSize
      ov_work_group_size = (ov_work_group_size/OpenCLContext::TileSize)*OpenCLContext::TileSize;
      int ov_work_group_size_max = 8*OpenCLContext::TileSize;
      if(ov_work_group_size > ov_work_group_size_max) ov_work_group_size = ov_work_group_size_max;

      //ov_work_group_size = (nb.getForceThreadBlockSize() < work_group_size_min) ? work_group_size_min : nb.getForceThreadBlockSize();

      // use default block size for both pair kernel and tree kernels
      ov_work_group_size = nb.getForceThreadBlockSize();

      if(verbose) cout << "2-body kernels Group Size: " << nb.getForceThreadBlockSize() << endl;
      if(verbose) cout << "Overlap Group Size: " << ov_work_group_size << endl;

      //figures out tree sizes, etc.
      int num_atoms = cl.getPaddedNumAtoms();
      int pad_modulo = max(nb.getForceThreadBlockSize(),ov_work_group_size);
      init_tree_size(num_atoms, num_sections, pad_modulo,
		     noverlaps, noverlaps_2body,
		     total_tree_size, tree_size, padded_tree_size, tree_pointer);
      if(true){
	for(int i = 0; i < num_sections; i++){
	  cout << "Tn: " << tree_size[i] << " " << padded_tree_size[i] << endl;
	}
	std::cout << "Tree size: " << total_tree_size << std::endl;
      }
      
      //now make total tree size allocated on device 
      //a multiple of the reduction blocks*tile size to ensure
      //that each section of the tree is a multiple of work group size
      
      //int n = num_reduction_blocks*nb.getForceThreadBlockSize();
      //total_tree_size = n*((total_tree_size+n-1)/n);

      if(verbose)
	std::cout << "Tree size: " << total_tree_size << std::endl;

      //now copy overlap tree arrays to device
      ovAtomTreePointer = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomTreePointer");
      ovAtomTreeSize = OpenCLArray::create<cl_int>(cl, num_sections, "ovAtomTreeSize");
      NIterations = OpenCLArray::create<cl_int>(cl, num_sections, "NIterations");
      ovAtomTreePaddedSize = OpenCLArray::create<cl_int>(cl, num_sections, "ovAtomTreePaddedSize");
      ovAtomTreeLock = OpenCLArray::create<cl_int>(cl, num_sections, "ovAtomTreeLock");
      ovAtomLock = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms(), "ovAtomLock");
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
      ovAtomBuffer = OpenCLArray::create<mm_float4>(cl, cl.getPaddedNumAtoms()*num_sections, "ovAtomBuffer");

      //atomic parameters
      GaussianExponent = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(), "GaussianExponent");
      GaussianVolume = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(), "GaussianVolume");
      AtomicGamma = OpenCLArray::create<cl_float>(cl, cl.getPaddedNumAtoms(), "AtomicGamma");

      //buffers to hold temporary lists of overlaps
      ovCountBuffer = OpenCLArray::create<cl_int>(cl, cl.getPaddedNumAtoms()*nb.getNumForceThreadBlocks(), "ovCountBuffer");

      //"long" version of energy buffer
      ovEnergyBuffer_long = OpenCLArray::create<cl_long>(cl, cl.getPaddedNumAtoms(), "ovEnergyBuffer_long");

      OpenCLCalcGVolForceKernel::copy_tree_to_device();
    }

    {
      //Reset tree kernel
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(num_sections);
      defines["ATOMS_PER_SECTION"] = cl.intToString(cl.getPaddedNumAtoms()/num_sections);
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      
      map<string, string> replacements;
      string file, kernel_name;
      cl::Program program;
      int index;
      cl::Kernel kernel;

      kernel_name = "resetTree";
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      file = cl.replaceStrings(OpenCLGVolKernelSources::GVolResetTree, replacements);
      program = cl.createProgram(file, defines);
      //reset tree kernel
      resetTreeKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
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
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      resetBufferKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = resetBufferKernel;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_sections);
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomLock->getDeviceBuffer());
      if(useLong) kernel.setArg<cl::Buffer>(index++, ovEnergyBuffer_long->getDeviceBuffer());

      //reset ovCount kernel
      kernel_name = "resetOvCount";
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      resetOvCountKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = resetOvCountKernel;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, nb.getNumForceThreadBlocks());
      kernel.setArg<cl::Buffer>(index++, ovCountBuffer->getDeviceBuffer());
      
      //reset tree counters kernel
      kernel_name = "resetSelfVolumes";
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      resetSelfVolumesKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      kernel = resetSelfVolumesKernel;
      index = 0;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
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
      pairValueDefines["ATOMS_PER_SECTION"] = cl.intToString(cl.getPaddedNumAtoms()/num_sections);
      pairValueDefines["MAX_2BODY_SIZE"] = cl.intToString(num_twobody_max);
      pairValueDefines["OV_WORK_GROUP_SIZE"] = cl.intToString(ov_work_group_size);

      map<string, string> replacements;

      replacements["KFC"] = cl.doubleToString((double)KFC);
      replacements["VOLMIN0"] = cl.doubleToString((double)VOLMIN0);
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
	"localData[localAtomIndex].gamma = global_atomic_gamma[j];\n"
	"localData[localAtomIndex].ov_count = 0;\n";




      replacements["ACQUIRE_TREE_LOCK"] =
	 "if(tgx == 0){//first atom in warp \n"	
	"	//acquire tree lock for the tile\n"
	"	//If lock is not acquired all of the threads in the warp will wait\n"
	"     do { children_count = atomic_xchg(&ovChildrenCount[atom1_tree_ptr], -1); } while(children_count < 0);\n"
        "}\n"
	"if(tgx != 0){//now the other threads in warp get count \n"
	"     children_count = atomic_xchg(&ovChildrenCount[atom1_tree_ptr], -1); \n"
	"}\n";

      replacements["RELEASE_TREE_LOCK"] =
	  "        //release lock\n"
	  "        atomic_xchg(&ovChildrenCount[atom1_tree_ptr], children_count); \n";

      //tree locks were used in the 2-body tree construction kernel. no more
      //replacements["ACQUIRE_TREE_LOCK"] = "";
      //replacements["RELEASE_TREE_LOCK"] = "";

      replacements["COMPUTE_INTERACTION_COUNT"] =
		"	real a12 = a1 + a2; \n"
		"	real deltai = 1./a12; \n"
	        "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
		"	real dfp = df/PI; \n"
		"	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
		"	if(gvol > VolMin0 ){ \n"
                "          ov_count += 1; //atomic_inc(&(ovCountBuffer[count_buffer_ptr + atom1])); \n"
		"       } \n";


      replacements["COMPUTE_INTERACTION_STORE1"] =
		"	real a12 = a1 + a2; \n"
		"	real deltai = 1./a12; \n"
	        "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
		"	real dfp = df/PI; \n"
		"	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n" 
		"	  real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
	        "         }\n"
                "         // switching function end \n"
                "	  real sfp = sp*gvol + s; \n"
                "         gvol = s*gvol; \n"
		"         /* at this point have:\n"
		"	     1. gvol: overlap  between atom1 and atom2\n"
		"	     2. a12: gaussian exponent of overlap\n"
		"	     3. v12=gvol: volume of overlap\n"
		"	     4. c12: gaussian center of overlap\n"
		"	     These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
		"	     volume is large enough.\n"
		"	 */\n"
	//"        children_count = atomic_inc(&ovChildrenStartIndex[atom1_tree_ptr]); \n"
		"        int endslot = atom1_children_start + children_count; \n"
		"        ovLevel[endslot] = 2; //two-body\n"
		"	 ovVolume[endslot] = gvol;\n"
	        "        ovVSfp[endslot] = sfp; \n"
		"	 ovGamma1i[endslot] = gamma1 + gamma2;\n"
		"	 ovLastAtom[endslot] = atom2;\n"
		"	 ovRootIndex[endslot] = atom1_tree_ptr;\n"
		"	 ovChildrenStartIndex[endslot] = -1;\n"
		"	 ovChildrenCount[endslot] = 0;\n"
		"	 ovG[endslot] = (real4)(c12.xyz, a12);\n"
	        "        ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
	        "        children_count += 1;\n"
		"      }\n";


      replacements["COMPUTE_INTERACTION_STORE2"] =
		"	real a12 = a1 + a2; \n"
		"	real deltai = 1./a12; \n"
	        "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
		"	real dfp = df/PI; \n"
		"	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       if(gvol > VolMinA){\n"
                "         real dgvol = -2.0f*df*gvol; \n"
                "         real dgvolv = v1 > 0 ? gvol/v1 : 0; \n" 
		"	  real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "         //switching function \n"
                "         real s = 0, sp = 0; \n"
                "         if(gvol > VolMinB ){ \n"
                "             s = 1.0f; \n"
                "             sp = 0.0f; \n"
                "         }else{ \n"
                "             real swd = 1.f/( VolMinB - VolMinA ); \n"
                "             real swu = (gvol - VolMinA)*swd; \n"
                "             real swu2 = swu*swu; \n"
                "             real swu3 = swu*swu2; \n"
                "             s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "             sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
	        "         }\n"
                "         // switching function end \n"
                "	  real sfp = sp*gvol + s; \n"
                "         gvol = s*gvol; \n"
		"	  /* at this point have:\n"
		"	     1. gvol: overlap  between atom1 and atom2\n"
		"	     2. a12: gaussian exponent of overlap\n"
		"	     3. v12=gvol: volume of overlap\n"
		"	     4. c12: gaussian center of overlap\n"
		"	     These, together with atom2 (last_atom) are entered into the tree for atom 1 if\n"
		"	     volume is large enough.\n"
		"	 */\n"
		"	  int endslot = ovChildrenStartIndex[slot] + ov_count; \n"
		"	  ovLevel[endslot] = level + 1; //two-body\n"
		"	  ovVolume[endslot] = gvol;\n"
	        "         ovVSfp[endslot] = sfp; \n"
		"	  ovGamma1i[endslot] = gamma1 + gamma2;\n"
		"	  ovLastAtom[endslot] = atom2;\n"
		"	  ovRootIndex[endslot] = slot;\n"
		"	  ovChildrenStartIndex[endslot] = -1;\n"
		"	  ovChildrenCount[endslot] = 0;\n"
		"	  ovG[endslot] = (real4)(c12.xyz, a12);\n"
	        "         ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);\n"
	        "         ovProcessedFlag[endslot] = 0;\n"
                "         ovOKtoProcessFlag[endslot] = 1;\n"
                "         ov_count += 1; \n"
		"       }\n";


      replacements["COMPUTE_INTERACTION_RESCAN"] =
		"	real a12 = a1 + a2; \n"
		"	real deltai = 1./a12; \n"
	        "       real df = a1*a2*deltai; \n"
                "       real ef = exp(-df*r2); \n"
		"	real dfp = df/PI; \n"
		"	real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef; \n"
                "       real dgvol = -2.0f*df*gvol; \n"
                "       real dgvolv = v1 > 0 ? gvol/v1 : 0; \n" 
		"       real4 c12 = deltai*(a1*posq1 + a2*posq2); \n"
                "       //switching function \n"
                "       real s = 0, sp = 0; \n"
                "       if(gvol > VolMinB ){ \n"
                "           s = 1.0f; \n"
                "           sp = 0.0f; \n"
                "       }else{ \n"
                "           real swd = 1.f/( VolMinB - VolMinA ); \n"
                "           real swu = (gvol - VolMinA)*swd; \n"
                "           real swu2 = swu*swu; \n"
                "           real swu3 = swu*swu2; \n"
                "           s = swu3*(10.f-15.f*swu+6.f*swu2); \n"
                "           sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2); \n"
	        "       }\n"
                "       // switching function end \n"
                "       real sfp = sp*gvol + s; \n"
                "       gvol = s*gvol; \n"
		"       ovVolume[slot] = gvol;\n"
	        "       ovVSfp[slot] = sfp; \n"
		"       ovG[slot] = (real4)(c12.xyz, a12);\n"
	        "       ovDV1[slot] = (real4)(-delta.xyz*dgvol,dgvolv);\n";

      string InitOverlapTreeSrc = cl.replaceStrings(OpenCLGVolKernelSources::GVolOverlapTree, replacements);


      kernel_name = "InitOverlapTree_1body";
      replacements["KERNEL_NAME"] = kernel_name;


      if(verbose) cout << "compiling GVolOverlapTree ..." ;
      program = cl.createProgram(InitOverlapTreeSrc, pairValueDefines);
      if(verbose) cout << " done. " << endl;
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      InitOverlapTreeKernel_1body_1 = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      int reset_tree_size = 1;
      index = 0;
      kernel = InitOverlapTreeKernel_1body_1;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_sections);
      kernel.setArg<cl_int>(index++, reset_tree_size);
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, NIterations->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
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

      if(verbose) cout << "compiling " << kernel_name << " ... ";
      program = cl.createProgram(InitOverlapTreeSrc, pairValueDefines);
      if(verbose) cout << " done. " << endl;
      InitOverlapTreeKernel_1body_2 = cl::Kernel(program, kernel_name.c_str());
      reset_tree_size = 0;
      index = 0;
      kernel = InitOverlapTreeKernel_1body_2;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_sections);
      kernel.setArg<cl_int>(index++, reset_tree_size);
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, NIterations->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
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

      kernel_name = "InitOverlapTreeCount";
      replacements["KERNEL_NAME"] = kernel_name;

      if(verbose) cout << "compiling " << kernel_name << " ... ";
      InitOverlapTreeCountKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = InitOverlapTreeCountKernel;
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
      kernel.setArg<cl::Buffer>(index++, ovCountBuffer->getDeviceBuffer());

      kernel_name = "reduceovCountBuffer";
      replacements["KERNEL_NAME"] = kernel_name;
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      reduceovCountBufferKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = reduceovCountBufferKernel;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, nb.getNumForceThreadBlocks());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovCountBuffer->getDeviceBuffer());

      kernel_name = "InitOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      InitOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
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
      kernel.setArg<cl::Buffer>(index++, ovAtomLock->getDeviceBuffer());

      kernel_name = "resetComputeOverlapTree";
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      program = cl.createProgram(InitOverlapTreeSrc, pairValueDefines);
      resetComputeOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = resetComputeOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovRootIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      


      //pass 2
      kernel_name = "ComputeOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      if(verbose) cout << "compiling " << kernel_name << " ... ";
      ComputeOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      if(verbose) cout << " done. " << endl;
      index = 0;
      kernel = ComputeOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, NIterations->getDeviceBuffer());
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
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenReported->getDeviceBuffer());

      if(verbose){
	int kernelMaxThreads = ComputeOverlapTreeKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(cl.getDevice());
	std::cout <<  "ComputeOverlapTreeKernel Max group: " << kernelMaxThreads   << std::endl;
      }


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





      //rescan kernels

      kernel_name = "ResetRescanOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      ResetRescanOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = ResetRescanOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePaddedSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());

      kernel_name = "InitRescanOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      InitRescanOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = InitRescanOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenStartIndex->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenCount->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());

      kernel_name = "RescanOverlapTree";
      replacements["KERNEL_NAME"] = kernel_name;
      RescanOverlapTreeKernel = cl::Kernel(program, kernel_name.c_str());
      index = 0;
      kernel = RescanOverlapTreeKernel;
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomTreeSize->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, NIterations->getDeviceBuffer());
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
      kernel.setArg<cl::Buffer>(index++, ovProcessedFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovOKtoProcessFlag->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovChildrenReported->getDeviceBuffer());
    }


    {
      //Self volumes kernel
      map<string, string> defines;
      defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
      defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
      defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
      defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
      defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
      defines["ATOMS_PER_SECTION"] = cl.intToString(cl.getPaddedNumAtoms()/num_sections);
      defines["OV_WORK_GROUP_SIZE"] = cl.intToString(ov_work_group_size);
      

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
      kernel.setArg<cl::Buffer>(index++, NIterations->getDeviceBuffer());
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
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      if(cl.getSupports64BitGlobalAtomics()){
	kernel.setArg<cl::Buffer>(index++, cl.getLongForceBuffer().getDeviceBuffer());
	kernel.setArg<cl::Buffer>(index++, ovEnergyBuffer_long->getDeviceBuffer());
      }

    }

#ifdef NOTUSED
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
      kernel.setArg<cl_int>(index++, num_sections);
      kernel.setArg<cl_int>(index++, total_tree_size);
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovSelfVolume->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovDV2->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovLastAtom->getDeviceBuffer());
    }
#endif

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
      
      //cout << "compiling " << kernel_name << endl;

      string file = OpenCLGVolKernelSources::GVolReduceTree;
      cl::Program program = cl.createProgram(file, defines);
      reduceSelfVolumesKernel_buffer = cl::Kernel(program, kernel_name.c_str());
      cl::Kernel kernel = reduceSelfVolumesKernel_buffer;
      int index = 0;
      kernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
      kernel.setArg<cl_int>(index++, num_sections);
      kernel.setArg<cl::Buffer>(index++, ovAtomTreePointer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, AtomicGamma->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, ovAtomBuffer->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, (useLong ? cl.getLongForceBuffer().getDeviceBuffer() : cl.getForceBuffers().getDeviceBuffer())); //master force buffer
      if(useLong) kernel.setArg<cl::Buffer>(index++, ovEnergyBuffer_long->getDeviceBuffer());
      kernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());

      
    }

    if(true){
      std::cout <<  "Num atoms: " << cl.getNumAtoms() << std::endl;
      std::cout <<  "Padded Num Atoms: " << cl.getPaddedNumAtoms() << std::endl;
      std::cout <<  "Num Blocks: " << cl.getNumAtomBlocks() << std::endl;
      std::cout <<  "Num Force Buffers: " << nb.getNumForceBuffers() << std::endl;
      
      std::cout <<  "Tile size: " << OpenCLContext::TileSize << std::endl;
      std::cout <<  "getNumForceThreadBlocks: " << nb.getNumForceThreadBlocks() << std::endl;
      std::cout <<  "Num Tree Sections: " << num_sections << std::endl;
      std::cout <<  "Force Work Group Size: " << nb.getForceThreadBlockSize() << std::endl;
      std::cout <<  "Overlap Work Group Size: " << ov_work_group_size << std::endl;
      std::cout <<  "Tree Size: " <<  total_tree_size << std::endl;
    }
  }

  if(verbose) cout << "Executing resetTreeKernel" << endl;
  cl.executeKernel(resetTreeKernel, nb.getForceThreadBlockSize()*cl.getNumAtomBlocks(), nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing resetBufferKernel" << endl;
  cl.executeKernel(resetBufferKernel, nb.getForceThreadBlockSize()*num_sections, nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing resetOvCountKernel" << endl;
  cl.executeKernel(resetOvCountKernel, nb.getForceThreadBlockSize()*nb.getNumForceThreadBlocks(), nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing InitOverlapTreeKernel_1body_1" << endl;
  cl.executeKernel(InitOverlapTreeKernel_1body_1, nb.getForceThreadBlockSize()*num_sections, nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing InitOverlapTreeCountKernel" << endl;
  cl.executeKernel(InitOverlapTreeCountKernel, nb.getForceThreadBlockSize()*nb.getNumForceThreadBlocks(), nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing reduceovCountBufferKernel" << endl;
  cl.executeKernel(reduceovCountBufferKernel, cl.getPaddedNumAtoms(), cl.getPaddedNumAtoms()/num_sections); //Nat threads, one group per tree
  if(verbose) cout << "Executing InitOverlapTreeKernel" << endl;
  cl.executeKernel(InitOverlapTreeKernel, nb.getForceThreadBlockSize()*nb.getNumForceThreadBlocks(), nb.getForceThreadBlockSize());
  if(verbose) cout << "Executing SortOverlapTree2bodyKernel" << endl;
  cl.executeKernel(SortOverlapTree2bodyKernel, cl.getPaddedNumAtoms(), 1);
  if(verbose) cout << "Executing resetComputeOverlapTreeKernel" << endl;
  cl.executeKernel(resetComputeOverlapTreeKernel, cl.getPaddedNumAtoms());
  if(verbose) cout << "Executing ComputeOverlapTreeKernel" << endl;
  cl.executeKernel(ComputeOverlapTreeKernel, ov_work_group_size*num_sections, ov_work_group_size);
  if(verbose) cout << "Executing resetSelfVolumesKernel" << endl;
  cl.executeKernel(resetSelfVolumesKernel, ov_work_group_size*num_sections, ov_work_group_size);
  if(verbose) cout << "Executing computeSelfVolumesKernel" << endl;
  cl.executeKernel(computeSelfVolumesKernel, ov_work_group_size*num_sections, ov_work_group_size);
  if(true){
    //rescan for energy with reduced radii
    cl.executeKernel(InitOverlapTreeKernel_1body_2, nb.getForceThreadBlockSize()*num_sections, nb.getForceThreadBlockSize());
    cl.executeKernel(ResetRescanOverlapTreeKernel, ov_work_group_size*num_sections, ov_work_group_size);
    cl.executeKernel(InitRescanOverlapTreeKernel, cl.getPaddedNumAtoms());
    cl.executeKernel(RescanOverlapTreeKernel, ov_work_group_size*num_sections, ov_work_group_size);
    cl.executeKernel(resetSelfVolumesKernel, ov_work_group_size*num_sections, ov_work_group_size);
    cl.executeKernel(computeSelfVolumesKernel, ov_work_group_size*num_sections, ov_work_group_size);
  }
  if(verbose) cout << "Executing reduceSelfVolumesKernel_buffer" << endl;
  cl.executeKernel(reduceSelfVolumesKernel_buffer, cl.getPaddedNumAtoms());
  if(verbose) cout << "Done GVOLSA kernels execution" << endl;

  if(false){
    vector<int> size(num_sections);
    vector<int> niter(num_sections);
    ovAtomTreeSize->download(size);
    cout << "Sizes: ";
    for(int section = 0; section < num_sections; section++){
      std::cout << size[section] << " ";
    }
    std::cout << endl;
    NIterations->download(niter);
    cout << "Niter: ";
    for(int section = 0; section < num_sections; section++){
      std::cout << niter[section] << " ";
    }
    std::cout << endl;

  }


  if(verbose){
    float self_volume = 0.0;
    vector<cl_float> self_volumes(total_tree_size);
    vector<cl_float> volumes(total_tree_size);
    vector<cl_float> gammas(total_tree_size);
    vector<cl_int> last_atom(total_tree_size);
    vector<cl_int> level(total_tree_size);
    vector<cl_int> parent(total_tree_size);
    vector<cl_int> children_start_index(total_tree_size);
    vector<cl_int> children_count(total_tree_size);
    vector<cl_int> children_reported(total_tree_size);
    vector<mm_float4> g(total_tree_size);
    vector<mm_float4> dv2(total_tree_size);
    vector<cl_float> sfp(total_tree_size);
    vector<int> size(num_sections);
    vector<cl_int> processed(total_tree_size);
    vector<cl_int> oktoprocess(total_tree_size);


    ovSelfVolume->download(self_volumes);
    ovVolume->download(volumes);
    ovVolume->download(volumes);
    ovLevel->download(level);
    ovLastAtom->download(last_atom);
    ovRootIndex->download(parent);
    ovChildrenStartIndex->download(children_start_index);
    ovChildrenCount->download(children_count);
    ovChildrenReported->download(children_reported);
    ovG->download(g);
    ovGamma1i->download(gammas);
    ovDV2->download(dv2);
    ovVSfp->download(sfp);
    ovAtomTreeSize->download(size);
    ovProcessedFlag->download(processed);
    ovOKtoProcessFlag->download(oktoprocess);



    std::cout << "Tree After Reset:" << std::endl;
    int nat = cl.getPaddedNumAtoms()/num_sections;
    for(int section = 0; section < num_sections; section++){
      std::cout << "Tree for sections: " << section << " " << " size= " << size[section] << std::endl;
      int iat = nat*section;
      int pp = tree_pointer[iat];
      int np = padded_tree_size[section];
      np = 256;
      //self_volume += self_volumes[pp];
      std::cout << "slot level LastAtom parent ChStart ChCount SelfV V gamma a x y z dedx dedy dedz sfp processed ok2process children_reported" << endl;
      for(int i = pp; i < pp + np ; i++){
	std::cout << std::setprecision(4) << std::setw(6) << i << " "  << std::setw(7) << (int)level[i] << " " << std::setw(7) << (int)last_atom[i] << " " << std::setw(7) << (int)parent[i] << " "  << std::setw(7) << (int)children_start_index[i] << " " << std::setw(7) <<  (int)children_count[i] << " " << std::setw(15) << (float)self_volumes[i] << " " << std::setw(10) << (float)volumes[i]  << " " << std::setw(10) << (float)gammas[i] << " " << std::setw(10) << g[i].w << " " << std::setw(10) << g[i].x << " " << std::setw(10) << g[i].y << " " << std::setw(10) << g[i].z << " " << std::setw(10) << dv2[i].x << " " << std::setw(10) << dv2[i].y << " " << std::setw(10) << dv2[i].z << " " << std::setw(10) << sfp[i] << " " << processed[i] << " " << oktoprocess[i] << " " << children_reported[i] << std::endl;
      }
    }
    //std::cout << "Volume (from self volumes):" << self_volume <<std::endl;
  }


  //gammas
  if(false){
    float mol_volume = 0.0;
    vector<cl_float> gamma(num_sections);
    AtomicGamma->download(gamma);
    std::cout << "Gammas:" << std::endl;
    for(int iat = 0; iat < numParticles; iat++){
      std::cout << iat << " " << gamma[iat] << std::endl;
    }
  }




  //force/energy buffer
  if(false){
    if(useLong){
      vector<cl_float> energies(cl.getPaddedNumAtoms());
      vector<cl_long> energies_long(cl.getPaddedNumAtoms());
      cl.getEnergyBuffer().download(energies);
      std::cout << "OpenMM Energy Buffer:" << std::endl;
      for(int i = 0; i < cl.getPaddedNumAtoms(); i++){
	std::cout << i << " " << energies[i] << std::endl;
      }
      ovEnergyBuffer_long->download(energies_long);
      std::cout << "Long Energy Buffer:" << std::endl;
      float scale = 1/(float) 0x100000000;
      for(int i = 0; i < cl.getPaddedNumAtoms(); i++){
	std::cout << i << " " << scale*energies_long[i] << std::endl;
      }
    }else{
      float mol_volume = 0.0;
      vector<mm_float4> dv2(num_sections*cl.getPaddedNumAtoms());
      ovAtomBuffer->download(dv2);
      
      std::cout << "Atom Buffer:" << std::endl;
      for(int i = 0; i < num_sections*cl.getPaddedNumAtoms(); i++){
	if(i%cl.getPaddedNumAtoms()==0){
	  std::cout << "Tree: " << i/cl.getPaddedNumAtoms() << " =======================" << endl;
	}
	std::cout << i%cl.getPaddedNumAtoms() << " " << dv2[i].x << " " << dv2[i].y << " " << dv2[i].z << " " << dv2[i].w << std::endl;
      }
    }



    //std::cout << "Self Volumes:" << std::endl;
    //for(int iat = 0; iat < numParticles; iat++){
    //  std::cout << iat << " " << dv2[iat].w << std::endl;
    //  mol_volume += dv2[iat].w;
    //}
    //std::cout << "Volume (from self volumes):" << mol_volume <<std::endl;
  }

  if(verbose) cout << "Done with GVolSA energy" << endl;
  
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

