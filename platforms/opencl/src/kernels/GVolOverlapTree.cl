#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)
#define Min_GVol (MIN_GVOL)
#define VolMin0 (VOLMIN0)
#define VolMinA (VOLMINA)
#define VolMinB (VOLMINB)

/* memory locking functions from http://www.cmsoft.com.br/opencl-tutorial/opencl-c99-atomics/
   by Douglas Coimbra de Andrade.
   occupied = 0 <- lock available
   occupied = 1 <- lock busy
  */
void GetSemaphor(__global int * semaphor) {
   int occupied = atomic_xchg(semaphor, 1);
   while(occupied > 0) //try until occupied = 0
   {
     occupied = atomic_xchg(semaphor, 1);
   }
}
void ReleaseSemaphor(__global int * semaphor)
{
   int prevVal = atomic_xchg(semaphor, 0);
}


typedef struct {
  real4 posq;
  int   ov_count;
  ATOM_PARAMETER_DATA
} AtomData;


typedef struct {
  int  atom2;
  real gvol;
  real sfp;
  real gamma;
  real4 ovG;
  real4 ovDV1;
} OverlapData;




//this kernel initializes the tree with 1-body overlaps
//it assumes that no. of atoms in tree section is < groups size
__kernel void InitOverlapTree_1body(
    unsigned const int             num_padded_atoms,
    unsigned const int             num_sections,
    unsigned const int             reset_tree_size,
    __global const int*   restrict ovTreePointer, 
    __global const int*   restrict ovNumAtomsInTree, 
    __global const int*   restrict ovFirstAtom, 
    __global       int*   restrict ovAtomTreeSize,    //sizes of tree sections
    __global       int*   restrict NIterations,      
    __global const int*   restrict ovAtomTreePaddedSize, 
    __global const int*   restrict ovAtomTreePointer,    //pointers to atoms in tree
    __global const real4* restrict posq, //atomic positions
    __global const float* restrict radiusParam, //atomic radius
    __global const float* restrict gammaParam, //gamma
    __global const int*   restrict ishydrogenParam, //1=hydrogen atom

    __global       real* restrict GaussianExponent, //atomic Gaussian exponent
    __global       real* restrict GaussianVolume, //atomic Gaussian volume
    __global       real* restrict AtomicGamma, //atomic Gaussian gamma

    __global       int*   restrict ovLevel, //this and below define tree
    __global       real*  restrict ovVolume,
    __global       real*  restrict ovVSfp,
    __global       real*  restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,
    __global       int*   restrict ovLastAtom,
    __global       int*   restrict ovRootIndex,
    __global       int*   restrict ovChildrenStartIndex,
    __global volatile int*   restrict ovChildrenCount
){
  const uint id = get_local_id(0); 
  unsigned int section = get_group_id(0);
  while(section < num_sections){
    int natoms_in_section = ovNumAtomsInTree[section];
    int iat = id;
    while(iat < natoms_in_section){
      int atom = ovFirstAtom[section] + iat;    

      bool h = (ishydrogenParam[atom] > 0);
      real r = radiusParam[atom];
      real a = KFC/(r*r);
      real v = h ? 0 : 4.f*PI*powr(r,3)/3.f;
      real g = h ? 0 : gammaParam[atom];
      real4 c = posq[atom];
      
      GaussianExponent[atom] = a;
      GaussianVolume[atom] = v;
      AtomicGamma[atom] = g;
      
      int slot = ovAtomTreePointer[atom];
      ovLevel[slot] = 1;
      ovVolume[slot] = v;
      ovVSfp[slot] = 1;
      ovGamma1i[slot] = g;
      ovG[slot] = (real4)(c.xyz,a);
      ovDV1[slot] = (real4)0.f;
      ovLastAtom[slot] = atom;

      iat += get_local_size(0);
    }
    if(id==0) {
      if(reset_tree_size) ovAtomTreeSize[section] = natoms_in_section;
      NIterations[section] = 0;
    }

    section += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
  }
}



//this kernel counts the no. of 2-body overlaps for each atom, stores in ovChildrenCount
__kernel void InitOverlapTreeCount(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
#ifdef USE_CUTOFF
    __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms, unsigned int maxTiles, __global const ushort2* exclusionTiles,
#else
    unsigned int numTiles,
#endif
    __global       int*  restrict ovChildrenCount
){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  __local AtomData localData[FORCE_WORK_GROUP_SIZE];
  const unsigned int localAtomIndex = get_local_id(0);
  INIT_VARS

#ifdef USE_CUTOFF
  //OpenMM's neighbor list stores tiles with exclusions separately from other tiles
      
  // First loop: process tiles that contain exclusions
  // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
  const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
    const ushort2 tileIndices = exclusionTiles[pos];
    uint x = tileIndices.x;
    uint y = tileIndices.y;
    if(y>x) {uint t = y; y = x; x = t;};//swap so that y<x
    
    uint atom1 = y*TILE_SIZE + tgx;
    int parent_slot = ovAtomTreePointer[atom1];
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real a1 = global_gaussian_exponent[atom1];
    real v1 = global_gaussian_volume[atom1];
    
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].g.w = global_gaussian_exponent[j];
    localData[localAtomIndex].v = global_gaussian_volume[j];

    SYNC_WARPS;
    if(y==x){//diagonal tile
    
      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real a2 = localData[localAtom2Index].g.w;
	real v2 = localData[localAtom2Index].v;
	int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && atom1 < atom2 && r2 < CUTOFF_SQUARED) {
	  COMPUTE_INTERACTION_COUNT
	}
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

    }else{//off-diagonal tile, pairs are unique, don't need to check atom1<atom2

      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real a2 = localData[localAtom2Index].g.w;
	real v2 = localData[localAtom2Index].v;
	int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED) {
	  COMPUTE_INTERACTION_COUNT
	}
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

    }
    SYNC_WARPS;
  }
#endif //USE_CUTOFF

  //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
  __local int atomIndices[FORCE_WORK_GROUP_SIZE];
  unsigned int numTiles = interactionCount[0];
  if(numTiles > maxTiles)
    return; // There wasn't enough memory for the neighbor list.
#endif
  int pos = (int) (warp*(long)numTiles/totalWarps);
  int end = (int) ((warp+1)*(long)numTiles/totalWarps);
  while (pos < end) {
#ifdef USE_CUTOFF
    // y-atom block of the tile
    // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below 
    uint y = tiles[pos];
    //uint iat = y*TILE_SIZE + tgx;
    //uint jat = interactingAtoms[pos*TILE_SIZE + tgx];
#else
    // find x and y coordinates of the tile such that y <= x
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }
#endif    

    uint atom1 = y*TILE_SIZE + tgx;
#ifndef USE_CUTOFF
    //the parent is taken as the atom with the smaller index, w/o cutoffs atom1 < atom2 because y<x
    int parent_slot = ovAtomTreePointer[atom1];
#endif
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real a1 = global_gaussian_exponent[atom1];
    real v1 = global_gaussian_volume[atom1];

#ifdef USE_CUTOFF
    //uint j = (iat < jat) ? jat : iat;
    uint j = interactingAtoms[pos*TILE_SIZE + tgx];
    atomIndices[get_local_id(0)] = j;
    if(j<PADDED_NUM_ATOMS){
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].g.w = global_gaussian_exponent[j];
      localData[localAtomIndex].v = global_gaussian_volume[j];
    }
#else
    uint j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].g.w = global_gaussian_exponent[j];
    localData[localAtomIndex].v = global_gaussian_volume[j];
#endif
    
    SYNC_WARPS;

    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real a2 = localData[localAtom2Index].g.w;
      real v2 = localData[localAtom2Index].v;
#ifdef USE_CUTOFF
      int atom2 = atomIndices[localAtom2Index];
#else
      int atom2 = x*TILE_SIZE + tj;
#endif
      bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
      compute = compute && r2 < CUTOFF_SQUARED;
#else
      //when not using a neighbor list we are getting diagonal tiles here
      if(x == y) compute = compute && atom1 < atom2;
#endif
      if (compute){
#ifdef USE_CUTOFF
	//the parent is taken as the atom with the smaller index
	int parent_slot = (atom1 < atom2) ? ovAtomTreePointer[atom1] : ovAtomTreePointer[atom2];
#endif
	COMPUTE_INTERACTION_COUNT
      }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }

    SYNC_WARPS;
    pos++;
  }       

}

#ifdef NOTNOW
// version of InitOverlapTreeCount optimized for CPU devices
//  1 CPU core, instead of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and process them 
__kernel __attribute__((reqd_work_group_size(1,1,1))) 
void InitOverlapTreeCount_cpu(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
#ifdef USE_CUTOFF
    __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms,
    unsigned int maxTiles,
    __global const ushort2* exclusionTiles,
#else
    unsigned int numTiles,
#endif
    __global       int*  restrict ovChildrenCount
    ){
  uint id = get_global_id(0);
  uint ncores = get_global_size(0);
  __local AtomData localData[TILE_SIZE];

  INIT_VARS

  uint warp = id;
  uint totalWarps = ncores;
    
#ifdef USE_CUTOFF
  //OpenMM's neighbor list stores tiles with exclusions separately from other tiles
  
  // First loop: process tiles that contain exclusions
  // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
  const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
    const ushort2 tileIndices = exclusionTiles[pos];
    uint x = tileIndices.x;
    uint y = tileIndices.y;

    // Load the data for this tile in local memory
    for (int j = 0; j < TILE_SIZE; j++) {
      unsigned int atom2 = x*TILE_SIZE + j;
      localData[j].posq = posq[atom2];
      localData[j].g.w = global_gaussian_exponent[atom2];
      localData[j].v = global_gaussian_volume[atom2];
    }
    
    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // load atom1 parameters from global arrays
      real4 posq1 = posq[atom1];
      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];
      
      int atom1_tree_ptr = ovAtomTreePointer[atom1];
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	uint atom2 = x*TILE_SIZE+j;
	
	// load atom2 parameters from local arrays
	real4 posq2 = localData[j].posq;
	real a2 = localData[j].g.w;
	real v2 = localData[j].v;
	
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

	bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED;
	//for diagonal tile make sure that each pair is processed only once	
	if(y==x) compute = compute && atom1 < atom2;
	if (compute) {
	  COMPUTE_INTERACTION_COUNT
	 }
      }
    }
  }
#endif //USE_CUTOFF


  //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
  __local int atomIndices[TILE_SIZE];
  unsigned int numTiles = interactionCount[0];
  if(numTiles > maxTiles)
    return; // There wasn't enough memory for the neighbor list.
#endif
  int pos = (int) (warp*(long)numTiles/totalWarps);
  int end = (int) ((warp+1)*(long)numTiles/totalWarps);
  while (pos < end) {
#ifdef USE_CUTOFF
    // y-atom block of the tile
    // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below 
    uint y = tiles[pos];
#else
    // find x and y coordinates of the tile such that y <= x
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }
#endif    

    // Load the data for this tile in local memory
    for (int localAtomIndex = 0; localAtomIndex < TILE_SIZE; localAtomIndex++) {
#ifdef USE_CUTOFF
      unsigned int j = interactingAtoms[pos*TILE_SIZE+localAtomIndex];
      atomIndices[localAtomIndex] = j;
      if (j < PADDED_NUM_ATOMS) {
	localData[localAtomIndex].posq = posq[j];
	localData[localAtomIndex].g.w = global_gaussian_exponent[j];
	localData[localAtomIndex].v = global_gaussian_volume[j];
      }
#else
      unsigned int j = x*TILE_SIZE + localAtomIndex;
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].g.w = global_gaussian_exponent[j];
      localData[localAtomIndex].v = global_gaussian_volume[j];
#endif
    }
      
    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // load atom1 parameters from global arrays
      real4 posq1 = posq[atom1];
      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];
      
      int atom1_tree_ptr = ovAtomTreePointer[atom1];
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	// load atom2 parameters from local arrays
	real4 posq2 = localData[j].posq;
	real a2 = localData[j].g.w;
	real v2 = localData[j].v;
	
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
	uint atom2 = atomIndices[j];
#else
	uint atom2 = x*TILE_SIZE + j;
#endif
	bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
	compute = compute && r2 < CUTOFF_SQUARED;
#else
	//when not using a neighbor list we are getting diagonal tiles here
	if(x == y) compute = compute && atom1 < atom2;
#endif
	if (compute){
	  COMPUTE_INTERACTION_COUNT
	}
      }
    }
    pos++;
  }
  
}
#endif


//this kernel counts the no. of 2-body overlaps for each atom, stores in ovChildrenCount
__kernel void InitOverlapTree(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global       int* restrict ovAtomTreeSize,       //actual sizes
    __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
    __global const real* restrict global_atomic_gamma, //atomic gammas
#ifdef USE_CUTOFF
        __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms, unsigned int maxTiles, __global const ushort2* exclusionTiles,
#else
    unsigned int numTiles,
#endif
    __global       int*  restrict ovLevel, //this and below define tree
    __global       real* restrict ovVolume,
    __global       real* restrict ovVSfp,
    __global       real* restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,

    __global       int*  restrict ovLastAtom,
    __global       int*  restrict ovRootIndex,
    __global       int*  restrict ovChildrenStartIndex,
    __global       int*  restrict ovChildrenCount,
    __global       int*  restrict ovChildrenCountTop,
    __global       int*  restrict ovChildrenCountBottom
){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  __local AtomData localData[FORCE_WORK_GROUP_SIZE];
  const unsigned int localAtomIndex = get_local_id(0);

  INIT_VARS

#ifdef USE_CUTOFF
  //OpenMM's neighbor list stores tiles with exclusions separately from other tiles
      
  // First loop: process tiles that contain exclusions
  // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
  const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
    const ushort2 tileIndices = exclusionTiles[pos];
    uint x = tileIndices.x;
    uint y = tileIndices.y;
    if(y>x) {uint t = y; y = x; x = t;};//swap so that y<x
    
    uint atom1 = y*TILE_SIZE + tgx;
    int parent_slot = ovAtomTreePointer[atom1];
    int parent_children_start = ovChildrenStartIndex[parent_slot];
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    LOAD_ATOM1_PARAMETERS
      
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
      
    SYNC_WARPS;

    if(y==x){//diagonal tile
    
      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
       
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	LOAD_ATOM2_PARAMETERS
	  int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && atom1 < atom2 && r2 < CUTOFF_SQUARED) {
	  int child_atom = atom2;
	  COMPUTE_INTERACTION_STORE1
	    }
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

   }else{

      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
       
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	LOAD_ATOM2_PARAMETERS
	  int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED) {
	  int child_atom = atom2;
	  COMPUTE_INTERACTION_STORE1
	    }
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

    }

    SYNC_WARPS;
  }
#endif //USE_CUTOFF
    

  //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
  __local int atomIndices[FORCE_WORK_GROUP_SIZE];
  unsigned int numTiles = interactionCount[0];
  if(numTiles > maxTiles)
    return; // There wasn't enough memory for the neighbor list.
#endif
  int pos = (int) (warp*(long)numTiles/totalWarps);
  int end = (int) ((warp+1)*(long)numTiles/totalWarps);
  while (pos < end) {
#ifdef USE_CUTOFF
    // y-atom block of the tile
    // atoms in x-atom block (y <= x?) are retrieved from interactingAtoms[] below 
    uint y = tiles[pos];
    //uint iat = y*TILE_SIZE + tgx;
    //uint jat = interactingAtoms[pos*TILE_SIZE + tgx];
#else
    // find x and y coordinates of the tile such that y <= x
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }
#endif
    
    uint atom1 = y*TILE_SIZE + tgx;
#ifndef USE_CUTOFF
    //the parent is taken as the atom with the smaller index, w/o cutoffs atom1 < atom2 because y<x
    int parent_slot = ovAtomTreePointer[atom1];
    int parent_children_start = ovChildrenStartIndex[parent_slot];
#endif
    
    //int atom1_tree_ptr = ovAtomTreePointer[atom1];
    //int atom1_children_start = ovChildrenStartIndex[atom1_tree_ptr];

    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    LOAD_ATOM1_PARAMETERS

#ifdef USE_CUTOFF
    uint j = interactingAtoms[pos*TILE_SIZE + tgx];
    atomIndices[get_local_id(0)] = j;
    if(j<PADDED_NUM_ATOMS){
	localData[localAtomIndex].posq = posq[j];
	LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
    }
#else
    uint j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
#endif
    localData[localAtomIndex].ov_count = 0;
    
    SYNC_WARPS;

    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {

      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      LOAD_ATOM2_PARAMETERS
#ifdef USE_CUTOFF
      int atom2 = atomIndices[localAtom2Index];
#else
      int atom2 = x*TILE_SIZE + tj;
#endif
      bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
      compute = compute && r2 < CUTOFF_SQUARED;
#else
      //when not using a neighbor list we are getting diagonal tiles here
      if(x == y) compute = compute && atom1 < atom2;
#endif
      if (compute){
#ifdef USE_CUTOFF
	//the parent is taken as the atom with the smaller index
	bool ordered = atom1 < atom2;
	int parent_slot = (ordered) ? ovAtomTreePointer[atom1] : ovAtomTreePointer[atom2];
	int child_atom = (ordered) ? atom2 : atom1 ;
	if(!ordered) delta = -delta; //parent and child are reversed (atom2>atom1)
	int parent_children_start = ovChildrenStartIndex[parent_slot];
#else
	int child_atom = atom2;	
#endif
	COMPUTE_INTERACTION_STORE1
       }
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }
    
    SYNC_WARPS;
    pos++;
  }
  
}

#ifdef NOTNOW
// version of InitOverlapTreeCount optimized for CPU devices
//  1 CPU core, instead of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and process them 
__kernel __attribute__((reqd_work_group_size(1,1,1))) 
void InitOverlapTree_cpu(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global       int* restrict ovAtomTreeSize,       //actual sizes
    __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
    __global const real* restrict global_atomic_gamma, //atomic gammas
#ifdef USE_CUTOFF
    __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms,
    unsigned int maxTiles,
    __global const ushort2* exclusionTiles,
#else
    unsigned int numTiles,
#endif
    __global       int*  restrict ovLevel, //this and below define tree
    __global       real* restrict ovVolume,
    __global       real* restrict ovVSfp,
    __global       real* restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,

    __global       int*  restrict ovLastAtom,
    __global       int*  restrict ovRootIndex,
    __global       int*  restrict ovChildrenStartIndex,
    __global       int*  restrict ovChildrenCount,
    __global       int*  restrict ovChildrenCountTop,
    __global       int*  restrict ovChildrenCountBottom
    ){
  
  uint id = get_global_id(0);
  uint ncores = get_global_size(0);
  __local AtomData localData[TILE_SIZE];
  
  INIT_VARS

  uint warp = id;
  uint totalWarps = ncores;

#ifdef USE_CUTOFF
  //OpenMM's neighbor list stores tiles with exclusions separately from other tiles
  
  // First loop: process tiles that contain exclusions
  // (this is imposed by OpenMM's neighbor list format, AGBNP does not actually have exclusions)
  const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
  for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
    const ushort2 tileIndices = exclusionTiles[pos];
    uint x = tileIndices.x;
    uint y = tileIndices.y;


    // Load the data for this tile in local memory
    for (int j = 0; j < TILE_SIZE; j++) {
      unsigned int atom2 = x*TILE_SIZE + j;
      localData[j].posq = posq[atom2];
      localData[j].g.w = global_gaussian_exponent[atom2];
      localData[j].v = global_gaussian_volume[atom2];
      localData[j].gamma = global_atomic_gamma[atom2];
    }
    
    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // load atom1 parameters from global arrays
      real4 posq1 = posq[atom1];
      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];
      real gamma1 = global_atomic_gamma[atom1];
      
      int atom1_tree_ptr = ovAtomTreePointer[atom1];
      int atom1_children_start = ovChildrenStartIndex[atom1_tree_ptr];
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	uint atom2 = x*TILE_SIZE+j;
	
	// load atom2 parameters from local arrays
	real4 posq2 = localData[j].posq;
	real a2 = localData[j].g.w;
	real v2 = localData[j].v;
	real gamma2 = localData[j].gamma;
	
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;

	bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE && r2 < CUTOFF_SQUARED;
	//for diagonal tile make sure that each pair is processed only once	
	if(y==x) compute = compute && atom1 < atom2;
	if (compute) {
	  COMPUTE_INTERACTION_STORE1
	 }
      }
    }
  }
#endif //USE_CUTOFF

  //second loop, tiles without exclusions or all interactions if not using cutoffs
#ifdef USE_CUTOFF
  __local int atomIndices[TILE_SIZE];
  unsigned int numTiles = interactionCount[0];
  if(numTiles > maxTiles)
    return; // There wasn't enough memory for the neighbor list.
#endif
  int pos = (int) (warp*(long)numTiles/totalWarps);
  int end = (int) ((warp+1)*(long)numTiles/totalWarps);
  while (pos < end) {
#ifdef USE_CUTOFF
    // y-atom block of the tile
    // atoms in x-atom block (y <= x) are retrieved from interactingAtoms[] below 
    uint y = tiles[pos];
#else
    // find x and y coordinates of the tile such that y <= x
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }
#endif
    
    // Load the data for this tile in local memory
    for (int localAtomIndex = 0; localAtomIndex < TILE_SIZE; localAtomIndex++) {
#ifdef USE_CUTOFF
      unsigned int j = interactingAtoms[pos*TILE_SIZE+localAtomIndex];
      atomIndices[localAtomIndex] = j;
      if (j < PADDED_NUM_ATOMS) {
	localData[localAtomIndex].posq = posq[j];
	localData[localAtomIndex].g.w = global_gaussian_exponent[j];
	localData[localAtomIndex].v = global_gaussian_volume[j];
	localData[localAtomIndex].gamma = global_atomic_gamma[j];
      }
#else
      unsigned int j = x*TILE_SIZE + localAtomIndex;
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].g.w = global_gaussian_exponent[j];
      localData[localAtomIndex].v = global_gaussian_volume[j];
      localData[localAtomIndex].gamma = global_atomic_gamma[j];
#endif
      localData[localAtomIndex].ov_count = 0;
    }

    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // load atom1 parameters from global arrays
      real4 posq1 = posq[atom1];
      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];
      real gamma1 = global_atomic_gamma[atom1];
      
      int atom1_tree_ptr = ovAtomTreePointer[atom1];
      int atom1_children_start = ovChildrenStartIndex[atom1_tree_ptr];
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {

	// load atom2 parameters from local arrays
	real4 posq2 = localData[j].posq;
	real a2 = localData[j].g.w;
	real v2 = localData[j].v;
	real gamma2 = localData[j].gamma;
	
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	
#ifdef USE_CUTOFF
	uint atom2 = atomIndices[j];
#else
	uint atom2 = x*TILE_SIZE + j;
#endif
	bool compute = atom1 < NUM_ATOMS_TREE && atom2 < NUM_ATOMS_TREE;
#ifdef USE_CUTOFF
	compute = compute && r2 < CUTOFF_SQUARED;
#else
	//when not using a neighbor list we are getting diagonal tiles here
	if(x == y) compute = compute && atom1 < atom2;
#endif
	if (compute){
	  COMPUTE_INTERACTION_STORE1
	 }

      }
    }
    pos++;
  }
}
#endif

//this kernel initializes the tree to be processed by ComputeOverlapTree()
//it assumes that 2-body overlaps are in place
__kernel void resetComputeOverlapTree(const int ntrees,
    __global const int*   restrict ovTreePointer,
    __global       int*   restrict ovProcessedFlag,
    __global       int*   restrict ovOKtoProcessFlag,
    __global const int*   restrict ovAtomTreeSize,
    __global const int*   restrict ovLevel
){
  uint local_id = get_local_id(0);
  int tree = get_group_id(0);
  while( tree < ntrees ){
    uint tree_ptr = ovTreePointer[tree];
    uint tree_size = ovAtomTreeSize[tree];
    uint endslot = tree_ptr + tree_size;
    uint slot = tree_ptr + local_id;
    while(slot < endslot){
      if(ovLevel[slot] == 1){
	ovProcessedFlag[slot] = 1;
	ovOKtoProcessFlag[slot] = 0;
      }else if(ovLevel[slot] == 2){
	ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.                                                                                                                       
	ovOKtoProcessFlag[slot] = 1;
      }
      slot += get_local_size(0);
    }
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}

//===================================================
//Utilities to do parallel prefix sum of an array ("scan").
//The input is an integer array and the output is the sum of
//the elements up to that element:
//a[i] -> Sum(j=0)to(j=i) a[j] ("inclusive")
//a[i] -> Sum(j=0)to(j=i-1) a[j] ("exclusive")
//
//Derived from NVIDIA's GPU Computing SDK
//https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclMarchingCubes/Scan.cl
//
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
//
//Note that the input is the item corresponding to the current thread (not the array)
//the output can be extracted from the second half of l_Data[] (I think)
//or directly from the return value (current thread)
//This version works only for a single work group so it is limited
//to array sizes = to max work group size
inline uint scan1Inclusive(uint idata, __local uint *l_Data, uint size){
  uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;
  barrier(CLK_LOCAL_MEM_FENCE);  

  for(uint offset = 1; offset < size; offset <<= 1){
    barrier(CLK_LOCAL_MEM_FENCE);
    int t = l_Data[pos] + l_Data[pos - offset];
    barrier(CLK_LOCAL_MEM_FENCE);
    l_Data[pos] = t;
  }

  barrier(CLK_LOCAL_MEM_FENCE);  
  return l_Data[pos];
}
inline uint scan1Exclusive(uint idata, __local uint *l_Data, uint size){
  return scan1Inclusive(idata, l_Data, size) - idata;
}


//scan of general size over global buffers (size must be multiple of work group size)
//repeated application of scan over work-group chunks
inline void scangExclusive(__global uint * buffer, __local uint *l_Data, uint size){
  uint gsize = get_local_size(0);
  uint niter = size/gsize;
  uint id = get_local_id(0);
  __local uint psum;

  uint i = id;

  uint sum = scan1Exclusive(buffer[i], l_Data, gsize);
  if(id == gsize-1) psum = sum + buffer[i];
  buffer[i] = sum;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  i += gsize;

  while(i<size){
    uint sum = scan1Exclusive(buffer[i], l_Data, gsize) + psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == gsize-1) psum = sum + buffer[i];
    buffer[i] = sum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    i += gsize;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}

//=====================================================================


//used for the "scan" of children counts to get children start indexes
//assumes that work group size = OV_WORK_GROUP_SIZE
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void reduceovCountBuffer(const int ntrees,
			 __global const int*  restrict ovTreePointer,
			 __global const int*  restrict ovAtomTreePointer,    //pointers to atom trees
			 __global       int*  restrict ovAtomTreeSize,       //actual sizes
			 __global const int*  restrict ovAtomTreePaddedSize, //actual sizes
			 __global       int*  restrict ovChildrenStartIndex,
			 __global       int*  restrict ovChildrenCount,
			 __global       int*  restrict ovChildrenCountTop,
			 __global       int*  restrict ovChildrenCountBottom){
  unsigned int local_id = get_local_id(0);
  unsigned int gsize = get_local_size(0);
  __local uint temp[2*OV_WORK_GROUP_SIZE];

  int tree = get_group_id(0);
  while(tree < ntrees){
 
    uint tree_size = ovAtomTreeSize[tree];
    uint tree_ptr = ovTreePointer[tree];

    if(tree_size <= gsize){

      // number of 1-body overlaps is less than
      // group size, can use faster scan routine

      uint atom_ptr = tree_ptr + local_id; 
      int children_count = 0;
      if (local_id < tree_size) {
	children_count = ovChildrenCount[atom_ptr];
      }
      unsigned int sum = scan1Exclusive(children_count, temp, gsize);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(local_id < tree_size) {
	ovChildrenStartIndex[atom_ptr] = tree_ptr + tree_size + sum;
	// resets to top and bottom counters
	ovChildrenCountTop[atom_ptr] = 0;
	ovChildrenCountBottom[atom_ptr] = 0;
      }
      if(local_id == tree_size - 1){
	//update tree size to include 2-body
	ovAtomTreeSize[tree] += sum + children_count;
      }

    }else{

      // do scan of an array of arbitrary size

      uint padded_tree_size = gsize*((tree_size + gsize - 1)/gsize);
      for(uint i = local_id; i < padded_tree_size ; i += gsize){
	uint atom_ptr = tree_ptr + i;
	ovChildrenStartIndex[atom_ptr] = (i < tree_size) ? ovChildrenCount[atom_ptr] : 0;
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      scangExclusive(&(ovChildrenStartIndex[tree_ptr]), temp, padded_tree_size);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if(local_id == 0){
	ovAtomTreeSize[tree] += ovChildrenStartIndex[tree_ptr + tree_size-1] + ovChildrenCount[tree_ptr + tree_size-1];
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      for(uint i = local_id; i < padded_tree_size ; i += gsize){
	if(i < tree_size){
	  ovChildrenStartIndex[tree_ptr + i] += tree_ptr + tree_size;
	}else{
	  ovChildrenStartIndex[tree_ptr + i] = 0;
	}
      }
      for(uint atom_ptr = tree_ptr + local_id; atom_ptr < tree_ptr + padded_tree_size ; atom_ptr += gsize){
	ovChildrenCountTop[atom_ptr] = 0;
      }
      for(uint atom_ptr = tree_ptr + local_id; atom_ptr < tree_ptr + padded_tree_size ; atom_ptr += gsize){
	ovChildrenCountBottom[atom_ptr] = 0;
      }
    }
    //next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

}


// insertion sort
inline void sortVolumes2body(
		        unsigned          const int            idx,
			unsigned          const int            nx,
			__global       real* restrict ovVolume,
			__global       real* restrict ovVSfp,
			__global       real* restrict ovGamma1i,
			__global       real4* restrict ovG,
			__global       real4* restrict ovDV1,
			__global       int*  restrict ovLastAtom){

  if(nx>0){
    
    for (unsigned int k = idx + 1; k < idx + nx; k++){
      
      real v     = ovVolume[k];
      real sfp   = ovVSfp[k];
      real gamma = ovGamma1i[k];
      real4 g4   = ovG[k];
      real4 dv1  = ovDV1[k];
      int   atom = ovLastAtom[k];
      
      unsigned int j = k - 1;
      while (j >= idx && ovVolume[j] < v){
	  ovVolume[j + 1]   = ovVolume[j];
	  ovVSfp[j + 1]     = ovVSfp[j];
	  ovGamma1i[j + 1]  = ovGamma1i[j];
	  ovG[j + 1]        = ovG[j];
	  ovDV1[j + 1]      = ovDV1[j];
	  ovLastAtom[j + 1] = ovLastAtom[j];
	  j -= 1;
      }
      ovVolume[j + 1]   = v;
      ovVSfp[j + 1]     = sfp;
      ovGamma1i[j + 1]  = gamma;
      ovG[j + 1]        = g4;
      ovDV1[j + 1]      = dv1;
      ovLastAtom[j + 1] = atom;
    }
  }
}

/* this kernel sorts the 2-body portions of the tree according to volume. It is structured so that each thread gets one atom */
__kernel void SortOverlapTree2body(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global const int* restrict ovAtomTreeSize,       //actual sizes
    __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
    __global       int*  restrict ovLevel, //this and below define tree
    __global       real* restrict ovVolume,
    __global       real* restrict ovVSfp,
    __global       real* restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,
    __global       int*  restrict ovLastAtom,
    __global       int*  restrict ovRootIndex,
    __global const int*  restrict ovChildrenStartIndex,
    __global const int*  restrict ovChildrenCount
				   ){
  uint atom = get_global_id(0); // to start

  while(atom < NUM_ATOMS_TREE){

    uint atom_ptr = ovAtomTreePointer[atom];
    int size = ovChildrenCount[atom_ptr];
    int offset = ovChildrenStartIndex[atom_ptr];

    if(size > 0 && offset >= 0 && ovLastAtom[atom_ptr] >= 0){
      // sort 2-body volumes of atom
      sortVolumes2body(offset, size,
		       ovVolume,
		       ovVSfp,
		       ovGamma1i,
		       ovG,
		       ovDV1,
		       ovLastAtom);
    }
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


//this kernel completes the tree with 3-body and higher overlaps.
//Each workgroup of size OV_WORK_GROUP_SIZE is assigned to a tree section
//Then starting from the top of the tree section, for each slot i to process do:
//1. Retrieve siblings j and process overlaps i<j, count non-zero overlaps
//2. Do a parallel prefix sum (scan) of the array of counts to fill ovChildrenStartIndex[]
//3. Re-process the i<j overlaps and saves them starting at ovChildrenStartIndex[]
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void ComputeOverlapTree(const int ntrees,
   __global const int* restrict ovTreePointer,
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global       int* restrict ovAtomTreeSize,       //actual sizes
   __global       int* restrict NIterations,
   __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
   __global       int* restrict ovAtomTreeLock,       //tree locks
   __global const real4* restrict posq, //atomic positions
   __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
   __global const real* restrict global_gaussian_volume, //atomic Gaussian prefactor
   __global const real* restrict global_atomic_gamma, //atomic Gaussian prefactor
   __global       int*  restrict ovLevel, //this and below define tree
   __global       real* restrict ovVolume,
   __global       real* restrict ovVSfp,
   __global       real* restrict ovGamma1i,
   __global       real4* restrict ovG,
   __global       real4* restrict ovDV1,
   
   __global       int*  restrict ovLastAtom,
   __global       int*  restrict ovRootIndex,
   __global       int*  restrict ovChildrenStartIndex,
   __global       int*  restrict ovChildrenCount,
   __global volatile int*   restrict ovProcessedFlag,
   __global volatile int*   restrict ovOKtoProcessFlag,
   __global volatile int*   restrict ovChildrenReported
   ){

  const uint local_id = get_local_id(0);

  __local          uint temp[2*OV_WORK_GROUP_SIZE];
  __local volatile uint nprocessed;
  __local volatile uint tree_size;
  __local volatile uint niterations;
  const uint gsize = OV_WORK_GROUP_SIZE;
  uint ov_count = 0;

  uint tree = get_group_id(0);
  while(tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];


    //initializes local working copy of tree size;
    if(local_id == 0) tree_size = ovAtomTreeSize[tree];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //this is the number of translations of the calculations window
    //to cover the tree section
    //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE  
    const uint nsections = ovAtomTreePaddedSize[tree]/ gsize ;
    //start at the top of the tree and iterate until the end of the tree is reached
    for(uint isection=0; isection < nsections; isection++){
      uint slot = tree_ptr + isection*OV_WORK_GROUP_SIZE + local_id; //the slot to work on
      
      if(local_id == 0) niterations = 0;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //useless?
      
      do{
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	if(local_id == OV_WORK_GROUP_SIZE-1) nprocessed = 0;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	int parent = ovRootIndex[slot];
	int level = ovLevel[slot];
	int atom1 = ovLastAtom[slot];
	real4 posq1 = (real4)(ovG[slot].xyz,0);
	real a1 = ovG[slot].w;
	real v1 = ovVolume[slot];
	real gamma1 = ovGamma1i[slot];
	int processed = ovProcessedFlag[slot];
	int ok2process = ovOKtoProcessFlag[slot];
	
	//pass 1: compute number of children to spawn 
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	ov_count = 0;
	bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom1 >= 0);
	if(letsgo){
	  // slot of first and last sibling
	  int start = ovChildrenStartIndex[parent];
	  int count = ovChildrenCount[parent];
	  if(start >= 0 && count > 0){
	    int end  = start+count;
	    // the sibling index of this sibling
	    int this_sibling = slot - start;
	    // i<j scan over siblings
	    for(int i= start + this_sibling + 1; i<end ; i++){
	      int atom2 = ovLastAtom[i];
	      if(atom2 >= 0){
		real4 posq2 = posq[atom2];
		real a2 = global_gaussian_exponent[atom2];
		real v2 = global_gaussian_volume[atom2];
		real gamma2 = global_atomic_gamma[atom2];
		//Gaussian overlap
		real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
		real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
		COMPUTE_INTERACTION_2COUNT
		  }
	    }
	  }
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//step2: derive children start indexes from children counts
	uint sum = scan1Exclusive(ov_count, temp, gsize); //(inclusive, exclusive)
	barrier(CLK_LOCAL_MEM_FENCE);
	if(letsgo){
	  ovChildrenStartIndex[slot] = tree_ptr + tree_size + sum;
	}
	
	//figures out the new tree size
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //global to sync ovChildrenStartIndex and Count
	if(local_id == OV_WORK_GROUP_SIZE-1) {
	  uint np = sum + ov_count;
	  nprocessed = np; //for the last thread this is the total number of new overlaps
	  tree_size += np;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//pass3: compute volumes etc. of children and store them in the tree
	//starting at ovChildrenStartIndex. Mark the children as okay to process
	//mark this slot as processed.
	ov_count = 0;
	if(letsgo){
	  // slot of first and last sibling
	  int count = ovChildrenCount[parent];
	  int start = ovChildrenStartIndex[parent];
	  if(start >= 0 && count > 0){ //? should never happen as at least slot must be a child of parent
	    int end  = start+count;
	    // the sibling index of this sibling
	    int this_sibling = slot - start;
	    // i<j scan over siblings
	    for(int i= start + this_sibling + 1; i<end ; i++){
	      int atom2 = ovLastAtom[i];
	      if(atom2 >= 0){
		real4 posq2 = posq[atom2];
		real a2 = global_gaussian_exponent[atom2];
		real v2 = global_gaussian_volume[atom2];
		real gamma2 = global_atomic_gamma[atom2];
		//Gaussian overlap
		real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
		real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
		COMPUTE_INTERACTION_STORE2
		  }
	    }
	  }
	  ovProcessedFlag[slot] = 1;
	  ovOKtoProcessFlag[slot] = 0;
	  ovChildrenCount[slot] = ov_count;
	}
	if(local_id == 0) niterations += 1;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //to sync ovProcessedFlag etc.
      }while(nprocessed > 0 && niterations < gsize); //matches do{}while
      
      if(local_id == 0){
	if(niterations > NIterations[tree]) NIterations[tree] = niterations;
      }
      
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //to sync ovProcessedFlag etc.
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //stores tree size in global mem
    if(local_id == 0) ovAtomTreeSize[tree] = tree_size;
    //next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
  }

}


//this kernel completes the tree with 3-body and higher overlaps avoiding 2 passes over overlap volumes.
//Each workgroup of size OV_WORK_GROUP_SIZE is assigned to a tree section
//Then starting from the top of the tree section, for each slot i to process do:
//1. Retrieve siblings j and process overlaps i<j, count non-zero overlaps
//2. Do a parallel prefix sum (scan) of the array of counts to fill ovChildrenStartIndex[]
//3. Re-process the i<j overlaps and saves them starting at ovChildrenStartIndex[]
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
  void ComputeOverlapTree_1pass(const int ntrees,
   __global const int* restrict ovTreePointer,
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global       int* restrict ovAtomTreeSize,       //actual sizes
   __global       int* restrict NIterations,
   __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
   __global       int* restrict ovAtomTreeLock,       //tree locks
   __global const real4* restrict posq, //atomic positions
   __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
   __global const real* restrict global_gaussian_volume, //atomic Gaussian prefactor
   __global const real* restrict global_atomic_gamma, //atomic Gaussian prefactor
   __global       int*  restrict ovLevel, //this and below define tree
   __global       real* restrict ovVolume,
   __global       real* restrict ovVSfp,
   __global       real* restrict ovGamma1i,
   __global       real4* restrict ovG,
   __global       real4* restrict ovDV1,
   
   __global       int*  restrict ovLastAtom,
   __global       int*  restrict ovRootIndex,
   __global       int*  restrict ovChildrenStartIndex,
   __global       int*  restrict ovChildrenCount,
   __global volatile int*   restrict ovProcessedFlag,
   __global volatile int*   restrict ovOKtoProcessFlag,
   __global volatile int*   restrict ovChildrenReported,

   // temporary buffers
   unsigned const int buffer_size,
   __global       real*  restrict gvol_buffer,
   __global       uint*  restrict tree_pos_buffer, // where to store in tree
   __global       int*   restrict i_buffer,
   __global       int*   restrict atomj_buffer
   ){

  const uint local_id = get_local_id(0);
  __local          uint temp[2*OV_WORK_GROUP_SIZE];
  __local volatile uint nprocessed;
  __local volatile uint tree_size;
  __local volatile uint niterations;
  const uint gsize = OV_WORK_GROUP_SIZE;



  __local volatile uint n_buffer; //how many overlaps in buffer to process
  __local volatile uint buffer_pos[OV_WORK_GROUP_SIZE]; //where to store in temp. buffers
  __local volatile uint parent1_buffer[OV_WORK_GROUP_SIZE]; //tree slot of "i" overlap
  __local volatile uint level1_buffer[OV_WORK_GROUP_SIZE]; //overlap level of of "i" overlap
  __local volatile real4 posq1_buffer[OV_WORK_GROUP_SIZE]; //position of "i" overlap
  __local volatile real a1_buffer[OV_WORK_GROUP_SIZE]; //a parameter of "i" overlap
  __local volatile real v1_buffer[OV_WORK_GROUP_SIZE]; //volume of "i" overlap
  __local volatile real gamma1_buffer[OV_WORK_GROUP_SIZE]; //gamma parameter of "i" overlap
  __local volatile uint children_count[OV_WORK_GROUP_SIZE]; //number of children

  uint tree = get_group_id(0);
  while(tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];
    uint buffer_offset = tree * (buffer_size/ntrees);

    //initializes local working copy of tree size;
    if(local_id == OV_WORK_GROUP_SIZE-1) tree_size = ovAtomTreeSize[tree];
    barrier(CLK_LOCAL_MEM_FENCE);

    //this is the number of translations of the calculations window
    //to cover the tree section
    //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE  
    const uint nsections = ovAtomTreePaddedSize[tree]/ gsize ;
    //start at the top of the tree and iterate until the end of the tree is reached
    for(uint isection=0; isection < nsections; isection++){
      uint slot = tree_ptr + isection*OV_WORK_GROUP_SIZE + local_id; //the slot to work on
      
      if(local_id == OV_WORK_GROUP_SIZE-1) niterations = 0;
      
      do{
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	if(local_id == OV_WORK_GROUP_SIZE-1) nprocessed = 0;
	
	int parent = ovRootIndex[slot];
	int atom1 = ovLastAtom[slot];
	int processed = ovProcessedFlag[slot];
	int ok2process = ovOKtoProcessFlag[slot];
	bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom1 >= 0);
	
	
	//
	// -- phase I: fill-up buffers --
	//
	
	// step 1: load overlap "i" parameters in local buffers
	level1_buffer[local_id] = ovLevel[slot];
	posq1_buffer[local_id] = (real4)(ovG[slot].xyz,0);
	a1_buffer[local_id] = ovG[slot].w;
	v1_buffer[local_id] = ovVolume[slot];
	gamma1_buffer[local_id] = ovGamma1i[slot];
	parent1_buffer[local_id] = slot;
	children_count[local_id] = 0;
	
	//  step 2: compute buffer pointers and number of overlaps
	uint ov_count = 0;
	int sibling_start = ovChildrenStartIndex[parent];
	int sibling_count = ovChildrenCount[parent];
	int my_sibling_idx = slot - sibling_start;
	if(letsgo){
	  // store number of interactions, that is the number" of younger" siblings,
	  // this will undergo a prefix sum below
	  ov_count = sibling_count - (my_sibling_idx + 1);
	}
	uint sum = scan1Exclusive(ov_count, temp, gsize);
	buffer_pos[local_id] = buffer_offset + sum;
	if(local_id == OV_WORK_GROUP_SIZE - 1) n_buffer = sum + ov_count;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(n_buffer > 0){ //something to process
	  
	  //  step 3: insert data for overlaps "j" into global buffers
	  if(letsgo && sibling_start >= 0 && sibling_count > 0){
	    int end  = sibling_start+sibling_count;
	    // store last atom of j overlap in global buffer
	    uint pos = buffer_pos[local_id];
	    for(int i= sibling_start + my_sibling_idx + 1; i<end ; i++, pos++){
	      i_buffer[pos] = local_id;
	      atomj_buffer[pos] = ovLastAtom[i];
	    }
	  }
	  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	  
	  //
	  // phase II: compute overlap volumes, compute number of non-zero volumes
	  //           (here threads work in tandem on all overlaps in global buffers)
	  
	  // step 1: compute overlap volumes
	  uint pos = local_id + buffer_offset;
	  while(pos < n_buffer + buffer_offset){
	    int overlap1 = i_buffer[pos];
	    int atom2 = atomj_buffer[pos];
	    uint fij = 0;
	    real vij = 0;
	    if(atom2 >= 0){
	      real4 posq1 = posq1_buffer[overlap1];
	      real a1 = a1_buffer[overlap1];
	      real v1 = v1_buffer[overlap1];
	      real4 posq2 = posq[atom2];
	      real a2 = global_gaussian_exponent[atom2];
	      real v2 = global_gaussian_volume[atom2];
	      //Gaussian overlap
	      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
	      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	      COMPUTE_INTERACTION_GVOLONLY
	      fij = gvol > VolMinA ? 1 : 0;
	      vij = gvol;
	    }
	    tree_pos_buffer[pos] = fij; //for prefix sum below
	    gvol_buffer[pos] = vij;
	    pos += gsize;
	  }
	  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	  
	  //step 2: prefix sum over "fij" flag buffer to compute number of non-zero overlaps and 
	  //        their placement in the tree.
	  uint padded_n_buffer = ( (n_buffer/OV_WORK_GROUP_SIZE) + 1) * OV_WORK_GROUP_SIZE;
	  int np = 0;
	  if(local_id == OV_WORK_GROUP_SIZE-1) np = tree_pos_buffer[buffer_offset+n_buffer-1]; 
	  scangExclusive(&(tree_pos_buffer[buffer_offset]), temp, padded_n_buffer);
	  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	  if(local_id == OV_WORK_GROUP_SIZE-1){ //retrieve total number of non-zero overlaps
	    nprocessed = tree_pos_buffer[buffer_offset + n_buffer-1] + np;
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);
	  

	  //step 3: compute other quantities for non-zero volumes and store in tree
	  pos = local_id + buffer_offset;
	  while(pos < n_buffer + buffer_offset){
	    int overlap1 = i_buffer[pos];
	    int atom2 = atomj_buffer[pos];
	    real gvol = gvol_buffer[pos];
	    uint endslot = tree_ptr + tree_size + tree_pos_buffer[pos];
	    if(atom2 >= 0 && gvol > VolMinA){
	      int level = level1_buffer[overlap1] + 1;
	      int parent_slot = parent1_buffer[overlap1];
	      real4 posq1 = posq1_buffer[overlap1];
	      real a1 = a1_buffer[overlap1];
	      real v1 = v1_buffer[overlap1];
	      real gamma1 = gamma1_buffer[overlap1];
	      real4 posq2 = posq[atom2];
	      real a2 = global_gaussian_exponent[atom2];
	      real gamma2 = global_atomic_gamma[atom2];
	      
	      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
	      COMPUTE_INTERACTION_OTHER
		ovLevel[endslot] = level;
	      ovVolume[endslot] = gvol;
	      ovVSfp[endslot] = sfp;
	      ovGamma1i[endslot] = gamma1 + gamma2;
	      ovLastAtom[endslot] = atom2;
	      ovRootIndex[endslot] = parent_slot;
	      ovChildrenStartIndex[endslot] = -1;
	      ovChildrenCount[endslot] = 0;
	      ovG[endslot] = (real4)(c12.xyz, a12);
	      ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);
	      ovProcessedFlag[endslot] = 0;
	      ovOKtoProcessFlag[endslot] = 1;
	      //update parent children counter
	      atomic_inc(&children_count[overlap1]);
	    }
	    pos += gsize;
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);

	  //scan of children counts to figure out children start indexes
	  sum = scan1Exclusive(children_count[local_id], temp, gsize);      
	  if(letsgo){
	    if(children_count[local_id] > 0){
	      ovChildrenStartIndex[slot] = tree_ptr + tree_size + sum;
	      ovChildrenCount[slot] = children_count[local_id];
	    }
	    ovProcessedFlag[slot] = 1;
	    ovOKtoProcessFlag[slot] = 0;
	  }
	  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //global to sync ovChildrenStartIndex etc.
	  //figures out the new tree size
	  if(local_id == OV_WORK_GROUP_SIZE-1) {
	    tree_size += nprocessed;
	    niterations += 1;
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);
	  
	}//n_buffer > 0
	
      }while(nprocessed > 0 && niterations < gsize); //matches do{}while
      
      if(local_id == OV_WORK_GROUP_SIZE-1){
	if(niterations > NIterations[tree]) NIterations[tree] = niterations;
      }

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); //to sync ovProcessedFlag etc.
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //stores tree size in global mem
    if(local_id == 0) ovAtomTreeSize[tree] = tree_size;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


/**
 * Initialize tree for rescanning, set Processed, OKtoProcess=1 for leaves and out-of-bound,
 */
__kernel void ResetRescanOverlapTree(const int ntrees,
  __global const          int*   restrict ovTreePointer,
  __global const          int*   restrict ovAtomTreePointer,    //pointers to atom tree sections
  __global const          int*   restrict ovAtomTreeSize,    //pointers to atom tree sections
  __global const          int*   restrict ovAtomTreePaddedSize,    //pointers to atom tree sections
  __global                int*   restrict ovProcessedFlag,
  __global                int*   restrict ovOKtoProcessFlag
				     ){
  const uint local_id = get_local_id(0);
  const uint gsize = OV_WORK_GROUP_SIZE;

  uint tree = get_group_id(0);
  while(tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];
    uint tree_size = ovAtomTreePaddedSize[tree];
    uint endslot = tree_ptr + tree_size;

    for(int slot=tree_ptr+local_id; slot<endslot; slot+=gsize){
      ovProcessedFlag[slot] = 0;
    }
    for(int slot=tree_ptr+local_id; slot<endslot; slot+=gsize){
      ovOKtoProcessFlag[slot] = 0;
    }

    //next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

}

//this kernel initializes the tree to be processed by RescanOverlapTree()
__kernel void InitRescanOverlapTree(const int ntrees,
    __global const int*   restrict ovTreePointer,
    __global const int*   restrict ovAtomTreeSize,
    __global       int*   restrict ovProcessedFlag,
    __global       int*   restrict ovOKtoProcessFlag,
    __global const int*   restrict ovLevel
){
  uint local_id = get_local_id(0);
  int tree = get_group_id(0);

  while( tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];
    uint tree_size = ovAtomTreeSize[tree];
    uint endslot = tree_ptr + tree_size;
    uint slot = tree_ptr + local_id;
    while(slot < endslot){
      if(ovLevel[slot] == 1){
	ovProcessedFlag[slot] = 1;
	ovOKtoProcessFlag[slot] = 0;
      }else if(ovLevel[slot] == 2){
	ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.                                                                                                                       
	ovOKtoProcessFlag[slot] = 1;
      }
      slot += get_local_size(0);
    }
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

#ifdef NOTNOW

    if(atom < NUM_ATOMS_TREE){
	int pp = ovAtomTreePointer[atom];
	ovProcessedFlag[pp] = 1; //1body is already done
	ovOKtoProcessFlag[pp] = 0;
	int ic = ovChildrenStartIndex[pp];
	int nc = ovChildrenCount[pp];
	if(ic >= 0 && nc > 0){
	  for(int slot = ic; slot < ic+nc ; slot++){ 
	    ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.
	    ovOKtoProcessFlag[slot] = 1;
	  }
	}
    }
    atom += get_global_size(0);
#endif
}

//this kernel recomputes the overlap volumes of the current tree
//it does not modify the tree in any other way
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void RescanOverlapTree(const int ntrees,
   __global const int* restrict ovTreePointer,
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global const int* restrict ovAtomTreeSize,       //actual sizes
   __global       int* restrict NIterations,
   __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
   __global const int* restrict ovAtomTreeLock,       //tree locks
   __global const real4* restrict posq, //atomic positions
   __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
   __global const real* restrict global_gaussian_volume, //atomic Gaussian prefactor
   __global const real* restrict global_atomic_gamma, //atomic gamma
   __global const int*  restrict ovLevel, //this and below define tree
   __global       real* restrict ovVolume,
   __global       real* restrict ovVSfp,
   __global       real* restrict ovGamma1i,
   __global       real4* restrict ovG,
   __global       real4* restrict ovDV1,
   
   __global const int*  restrict ovLastAtom,
   __global const int*  restrict ovRootIndex,
   __global const int*  restrict ovChildrenStartIndex,
   __global const int*  restrict ovChildrenCount,
   __global volatile int*   restrict ovProcessedFlag,
   __global volatile int*   restrict ovOKtoProcessFlag,
   __global volatile int*   restrict ovChildrenReported
   ){

  const uint local_id = get_local_id(0);
  const uint gsize = OV_WORK_GROUP_SIZE;
  __local uint nprocessed;

  uint tree = get_group_id(0);
  while(tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];
    uint tree_size = ovAtomTreeSize[tree];

    //this is the number of translations of the calculations window
    //to cover the tree section
    //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE  
    const uint nsections = ovAtomTreePaddedSize[tree]/ gsize ;
    //start at the top of the tree and iterate until the end of the tree is reached
    for(uint isection=0; isection < nsections; isection++){
      uint slot = tree_ptr + isection*OV_WORK_GROUP_SIZE + local_id; //the slot to work on
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      int parent = ovRootIndex[slot];
      int atom = ovLastAtom[slot];
      
      //for(uint iiter = 0; iiter < 2 ; iiter++){
      do{
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	if(local_id==0) nprocessed = 0;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	int processed = ovProcessedFlag[slot];
	int ok2process = ovOKtoProcessFlag[slot];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom >= 0);
	if(letsgo){
	  atomic_inc(&nprocessed);
	  real4 posq1 = (real4)(ovG[parent].xyz,0);
	  real a1 = ovG[parent].w;
	  real v1 = ovVolume[parent];
	  real gamma1 = ovGamma1i[parent];
	  real4 posq2 = posq[atom];
	  real a2 = global_gaussian_exponent[atom];
	  real v2 = global_gaussian_volume[atom];
	  real gamma2 = global_atomic_gamma[atom];
	  //ovGamma1i[slot] = ovGamma1i[parent] + global_atomic_gamma[atom];
	  ovGamma1i[slot] = gamma1 + gamma2;
	  //Gaussian overlap
	  real4 delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
	  real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	  COMPUTE_INTERACTION_RESCAN
	    //mark itself as processed and children as okay to process
	    ovProcessedFlag[slot] = 1;
	  ovOKtoProcessFlag[slot] = 0;
	  if(ovChildrenStartIndex[slot] >= 0 && ovChildrenCount[slot] > 0){
	    for(int i = ovChildrenStartIndex[slot]; i < ovChildrenStartIndex[slot] + ovChildrenCount[slot]; i++){
	      ovOKtoProcessFlag[i] = 1;
	    }
	  }
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      } while(nprocessed > 0);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
 
    // next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}


//this kernel initializes the 1-body nodes with a new set of atomic gamma parameters
__kernel void InitOverlapTreeGammas_1body(
    unsigned const int             num_padded_atoms,
    unsigned const int             num_sections,
    __global const int*   restrict ovTreePointer, 
    __global const int*   restrict ovNumAtomsInTree, 
    __global const int*   restrict ovFirstAtom, 
    __global       int*   restrict ovAtomTreeSize,    //sizes of tree sections
    __global       int*   restrict NIterations,      
    __global const int*   restrict ovAtomTreePointer,    //pointers to atoms in tree
    __global const float* restrict gammaParam, //gamma
    __global       real*  restrict ovGamma1i
){
  const uint id = get_local_id(0); 
  unsigned int section = get_group_id(0);
  while(section < num_sections){
    int natoms_in_section = ovNumAtomsInTree[section];
    int iat = id;
    while(iat < natoms_in_section){
      int atom = ovFirstAtom[section] + iat;    

      real g = gammaParam[atom];
      int slot = ovAtomTreePointer[atom];
      ovGamma1i[slot] = g;

      iat += get_local_size(0);
    }
    if(id==0) {
      NIterations[section] = 0;
    }

    section += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
  }
}


//Same as RescanOverlapTree above:
//propagates gamma atomic parameters from the top to the bottom
//of the overlap tree,
//it *does not* recompute overlap volumes
//  used to prep calculations of volume derivatives of GB and van der Waals energies
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void RescanOverlapTreeGammas(const int ntrees,
   __global const int* restrict ovTreePointer,
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global const int* restrict ovAtomTreeSize,       //actual sizes
   __global       int* restrict NIterations,
   __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
   __global const int* restrict ovAtomTreeLock,       //tree locks
   __global const real* restrict global_atomic_gamma, //atomic gamma
   __global const int*  restrict ovLevel, //this and below define tree
   __global       real* restrict ovGamma1i,
   __global const int*  restrict ovLastAtom,
   __global const int*  restrict ovRootIndex,
   __global const int*  restrict ovChildrenStartIndex,
   __global const int*  restrict ovChildrenCount,
   __global volatile int*   restrict ovProcessedFlag,
   __global volatile int*   restrict ovOKtoProcessFlag,
   __global volatile int*   restrict ovChildrenReported
   ){

  const uint local_id = get_local_id(0);
  const uint gsize = OV_WORK_GROUP_SIZE;
  __local uint nprocessed;

  uint tree = get_group_id(0);
  while(tree < ntrees){
    uint tree_ptr = ovTreePointer[tree];
    uint tree_size = ovAtomTreeSize[tree];

    //this is the number of translations of the calculations window
    //to cover the tree section
    //make sure padded tree size is multiple of OV_WORK_GROUP_SIZE  
    const uint nsections = ovAtomTreePaddedSize[tree]/ gsize ;
    //start at the top of the tree and iterate until the end of the tree is reached
    for(uint isection=0; isection < nsections; isection++){
      uint slot = tree_ptr + isection*OV_WORK_GROUP_SIZE + local_id; //the slot to work on
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      int parent = ovRootIndex[slot];
      int atom = ovLastAtom[slot];
      
      //for(uint iiter = 0; iiter < 2 ; iiter++){
      do{
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	if(local_id==0) nprocessed = 0;
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	int processed = ovProcessedFlag[slot];
	int ok2process = ovOKtoProcessFlag[slot];
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	bool letsgo = (parent >= 0 && processed == 0 && ok2process > 0 && atom >= 0);
	if(letsgo){
	  atomic_inc(&nprocessed);
	  real gamma1 = ovGamma1i[parent];
	  real gamma2 = global_atomic_gamma[atom];
	  ovGamma1i[slot] = gamma1 + gamma2;
	  //mark itself as processed and children as okay to process
	  ovProcessedFlag[slot] = 1;
	  ovOKtoProcessFlag[slot] = 0;
	  if(ovChildrenStartIndex[slot] >= 0 && ovChildrenCount[slot] > 0){
	    for(int i = ovChildrenStartIndex[slot]; i < ovChildrenStartIndex[slot] + ovChildrenCount[slot]; i++){
	      ovOKtoProcessFlag[i] = 1;
	    }
	  }
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      } while(nprocessed > 0);
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
 
    // next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}
