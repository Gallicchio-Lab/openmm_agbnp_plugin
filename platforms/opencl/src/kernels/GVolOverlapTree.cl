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




//this kernel initializes the tree with 1-body overlaps.
__kernel void InitOverlapTree_1body(
    unsigned const int             num_padded_atoms,
    unsigned const int             num_sections,
    unsigned const int             reset_tree_size,
    __global       int*   restrict ovAtomTreeSize,       //sizes of tree sections
    __global       int*   restrict NIterations,       //sizes of tree sections
    __global const int*   restrict ovAtomTreePaddedSize, 
    __global const int*   restrict ovAtomTreePointer,    //pointers to atom tree sections
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
  unsigned int num_atoms_per_section = num_padded_atoms/num_sections;
  unsigned int section = get_group_id(0);
  __local volatile unsigned int tree_section_size; 
  while(section < num_sections){

    int id = get_local_id(0);
    if(id == 0) tree_section_size = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    while( id < num_atoms_per_section ){

      int atom = section*num_atoms_per_section + id;

      if(atom < NUM_ATOMS){
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
	ovGamma1i[slot] = gammaParam[atom];
	ovG[slot] = (real4)(c.xyz,a);
	ovDV1[slot] = (real4)0.f;
	ovLastAtom[slot] = atom;
	atomic_inc(&tree_section_size);
      }
      id += get_local_size(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(reset_tree_size > 0 && get_local_id(0)==0) ovAtomTreeSize[section] = tree_section_size;
    if(get_local_id(0)==0) NIterations[section] = 0;

    section += get_num_groups(0);
  }
}

//this kernel initializes the ovCount buffer which holds the number of 2-body
//overlaps for each atom 
__kernel void InitOverlapTreeCount(
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
    __global const real* restrict global_atomic_gamma, //atomic gammas
    
    __global const unsigned int* restrict exclusions, //this and below from pair energy kernel
    __global const ushort2* exclusionTiles,
#ifdef USE_CUTOFF
    __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
    real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, 
    unsigned int maxTiles, __global const real4* restrict blockCenter,
    __global const real4* restrict blockSize, __global const int* restrict interactingAtoms
#else
    unsigned int numTiles,
#endif
    __global       int*  restrict ovCountBuffer
    ){
    const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
    const unsigned int warp = get_global_id(0)/TILE_SIZE;
    const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
    const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
    __local AtomData localData[FORCE_WORK_GROUP_SIZE];
    const unsigned int count_buffer_ptr = get_group_id(0)*PADDED_NUM_ATOMS;
    unsigned int ov_count = 0; //2-body overlap temp. counter

    INIT_VARS

    // First loop: process tiles that contain exclusions.
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
	unsigned int atom1 = y*TILE_SIZE + tgx;
	ov_count = 0;

        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS

#ifdef USE_EXCLUSIONS
        unsigned int excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        if (x == y) {
            // This tile is on the diagonal.

            const unsigned int localAtomIndex = get_local_id(0);
            localData[localAtomIndex].posq = posq1;
            LOAD_LOCAL_PARAMETERS_FROM_1
	    

	    SYNC_WARPS;	    
	    

            for (unsigned int j = 0; j < TILE_SIZE; j++) {

                int localAtom2Index = tbx+j;
                real4 posq2 = localData[localAtom2Index].posq;
                real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    int atom2 = x*TILE_SIZE+j;

#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif

                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {
		      COMPUTE_INTERACTION_COUNT
		    }


#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif

                SYNC_WARPS;
            }

	    SYNC_WARPS;
        }
        else { // This is an off-diagonal tile.

            const unsigned int localAtomIndex = get_local_id(0);
            unsigned int j = x*TILE_SIZE + tgx;
            localData[localAtomIndex].posq = posq[j];
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

            SYNC_WARPS;
#ifdef USE_EXCLUSIONS
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {

                int localAtom2Index = tbx+tj;
                real4 posq2 = localData[localAtom2Index].posq;
                real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    int atom2 = x*TILE_SIZE+tj;

#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif

                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        COMPUTE_INTERACTION_COUNT
                    }

#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                tj = (tj + 1) & (TILE_SIZE - 1);

                SYNC_WARPS;
            }


            SYNC_WARPS;
        }
        int children_count;
	int pp = count_buffer_ptr + atom1;
	do { children_count = atomic_xchg(&ovCountBuffer[pp], -1); } while(children_count < 0);
	children_count += ov_count;
	atomic_xchg(&ovCountBuffer[pp], children_count);
    }

    ov_count = 0;
    int atom1;

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
#else
    int pos = (int) (warp*(long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    __local int atomIndices[FORCE_WORK_GROUP_SIZE];
    __local volatile int skipTiles[FORCE_WORK_GROUP_SIZE];
    skipTiles[get_local_id(0)] = -1;

    while (pos < end) {
        const bool isExcluded = false;
        bool includeTile = true;

        // Extract the coordinates of this tile.
        
        int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        if (numTiles <= maxTiles) {
            x = tiles[pos];
            real4 blockSizeX = blockSize[x];
            singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                                  0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                                  0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
        }
        else
#endif
        {
            y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            }

            // Skip over tiles that have exclusions, since they were already processed.

            SYNC_WARPS;
            while (skipTiles[tbx+TILE_SIZE-1] < pos) {
                SYNC_WARPS;
                if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                    ushort2 tile = exclusionTiles[skipBase+tgx];
                    skipTiles[get_local_id(0)] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
                }
                else
                    skipTiles[get_local_id(0)] = end;
                skipBase += TILE_SIZE;            
                currentSkipIndex = tbx;
                SYNC_WARPS;
            }
            while (skipTiles[currentSkipIndex] < pos)
                currentSkipIndex++;
            includeTile = (skipTiles[currentSkipIndex] != pos);
        }

	unsigned int atom1 = y*TILE_SIZE + tgx;
	ov_count = 0;

        if (includeTile) {
            // Load atom data for this tile.

            real4 posq1 = posq[atom1];
            LOAD_ATOM1_PARAMETERS
            unsigned int localAtomIndex = get_local_id(0);
#ifdef USE_CUTOFF
            unsigned int j = (numTiles <= maxTiles ? interactingAtoms[pos*TILE_SIZE+tgx] : x*TILE_SIZE + tgx);
#else
            unsigned int j = x*TILE_SIZE + tgx;
#endif
            atomIndices[get_local_id(0)] = j;
            if (j < PADDED_NUM_ATOMS) {
                localData[localAtomIndex].posq = posq[j];
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

            }

            SYNC_WARPS;

#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(posq1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(local_posq[get_local_id(0)], blockCenterX)
                SYNC_WARPS;
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {

                    int localAtom2Index = tbx+tj;
                    real4 posq2 = local_posq[localAtom2Index];
                    real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndices[tbx+tj];



                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION_COUNT
                        }	


                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);

                    SYNC_WARPS;
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.


                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {

                    int localAtom2Index = tbx+tj;
                    real4 posq2 = localData[localAtom2Index].posq;
                    real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndices[tbx+tj];



                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION_COUNT
                        }


#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);

                    SYNC_WARPS;
                }

                SYNC_WARPS;
            }
        


        }
	int children_count;
	int pp = count_buffer_ptr + atom1;
	do { children_count = atomic_xchg(&ovCountBuffer[pp], -1); } while(children_count < 0);
	children_count += ov_count;
	atomic_xchg(&ovCountBuffer[pp], children_count);

        pos++;
    }
}

//this kernel initializes the tree with 2-body overlaps.
//It stores 2-body overlaps using the ovChildrenStartIndex pointer of each
//atom, which has been prepared by InitOverlapTreeCount() and reduceovCountBuffer()
__kernel void InitOverlapTree(
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global       int* restrict ovAtomTreeSize,       //actual sizes
    __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
    __global       int* restrict ovAtomTreeLock,       //tree locks
    
    __global const real4* restrict posq, //atomic positions
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume
    __global const real* restrict global_atomic_gamma, //atomic gammas
    
    __global const unsigned int* restrict exclusions, //this and below from pair energy kernel
    __global const ushort2* exclusionTiles,
#ifdef USE_CUTOFF
    __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
    real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, 
    unsigned int maxTiles, __global const real4* restrict blockCenter,
    __global const real4* restrict blockSize, __global const int* restrict interactingAtoms
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
    __global       int*  restrict ovAtomLock
    ){
    const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
    const unsigned int warp = get_global_id(0)/TILE_SIZE;
    const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp [0:TILE_SIZE-1]
    const unsigned int tbx = get_local_id(0) - tgx;           //beginning of warp
    const unsigned int localAtomIndex = get_local_id(0);
    __local AtomData    localData[FORCE_WORK_GROUP_SIZE];
    int children_count;

    INIT_VARS

    // First loop: process tiles that contain exclusions.
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
	unsigned int atom1 = y*TILE_SIZE + tgx;

	int atom1_tree_ptr = ovAtomTreePointer[atom1];
	int atom1_children_start = ovChildrenStartIndex[atom1_tree_ptr];

	ACQUIRE_TREE_LOCK

        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS

#ifdef USE_EXCLUSIONS
        unsigned int excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        if (x == y) {
            // This tile is on the diagonal.


            localData[localAtomIndex].posq = posq1;
            LOAD_LOCAL_PARAMETERS_FROM_1
	    

	    SYNC_WARPS;	    
	    

            for (unsigned int j = 0; j < TILE_SIZE; j++) {

                int localAtom2Index = tbx+j;
                real4 posq2 = localData[localAtom2Index].posq;
                real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    int atom2 = x*TILE_SIZE+j;

#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif

                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {
		      COMPUTE_INTERACTION_STORE1
		    }


#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif

                SYNC_WARPS;
            }

	    SYNC_WARPS;
        }
        else { // This is an off-diagonal tile.

            unsigned int j = x*TILE_SIZE + tgx;
            localData[localAtomIndex].posq = posq[j];
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

            SYNC_WARPS;
#ifdef USE_EXCLUSIONS
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {

                int localAtom2Index = tbx+tj;
                real4 posq2 = localData[localAtom2Index].posq;
                real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    int atom2 = x*TILE_SIZE+tj;

#ifdef USE_EXCLUSIONS
                    bool isExcluded = !(excl & 0x1);
#endif

                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        COMPUTE_INTERACTION_STORE1
                    }

#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
                tj = (tj + 1) & (TILE_SIZE - 1);

                SYNC_WARPS;
            }


            SYNC_WARPS;
        }


#ifdef NOTNOW
	uint pp = ovAtomTreePointer[atom1];
	int children_count;
	do { children_count = atomic_xchg(&ovChildrenCount[pp], -1); } while(children_count < 0);//lock to protect from other work groups
	uint endslot = ovChildrenStartIndex[pp] + children_count;
	for (uint i = 0; i < TILE_SIZE; i++){
	  if(OverlapBuffer[i].atom2 >= 0){
	    ovLevel[endslot] = 2; //two-body
	    ovVolume[endslot] = OverlapBuffer[i].gvol;
	    ovVSfp[endslot] = OverlapBuffer[i].sfp;
	    ovGamma1i[endslot] = OverlapBuffer[i].gamma;
	    ovLastAtom[endslot] = OverlapBuffer[i].atom2;
	    ovG[endslot] = OverlapBuffer[i].ovG;
	    ovDV1[endslot] = OverlapBuffer[i].ovDV1;
	    ovRootIndex[endslot] = pp;
	    ovChildrenStartIndex[endslot] = -1;
	    ovChildrenCount[endslot] = 0;
	    endslot += 1;
	    children_count += 1;
	  }
	  atomic_xchg(&ovChildrenCount[pp], children_count); //releases lock
	}
#endif

	RELEASE_TREE_LOCK;
    }
    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
#else
    int pos = (int) (warp*(long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    __local int atomIndices[FORCE_WORK_GROUP_SIZE];
    __local volatile int skipTiles[FORCE_WORK_GROUP_SIZE];
    skipTiles[get_local_id(0)] = -1;

    while (pos < end) {
        const bool isExcluded = false;
        bool includeTile = true;

        // Extract the coordinates of this tile.
        
        int x, y;
        bool singlePeriodicCopy = false;

#ifdef USE_CUTOFF
        if (numTiles <= maxTiles) {
            x = tiles[pos];
            real4 blockSizeX = blockSize[x];
            singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                                  0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                                  0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
        }
        else
#endif
        {
            y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            }

            // Skip over tiles that have exclusions, since they were already processed.

            SYNC_WARPS;
            while (skipTiles[tbx+TILE_SIZE-1] < pos) {
                SYNC_WARPS;
                if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                    ushort2 tile = exclusionTiles[skipBase+tgx];
                    skipTiles[get_local_id(0)] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
                }
                else
                    skipTiles[get_local_id(0)] = end;
                skipBase += TILE_SIZE;            
                currentSkipIndex = tbx;
                SYNC_WARPS;
            }
            while (skipTiles[currentSkipIndex] < pos)
                currentSkipIndex++;
            includeTile = (skipTiles[currentSkipIndex] != pos);
        }

	unsigned int atom1 = y*TILE_SIZE + tgx;
	int atom1_tree_ptr = ovAtomTreePointer[atom1];
	int atom1_children_start = ovChildrenStartIndex[atom1_tree_ptr];

        if (includeTile) {

	    ACQUIRE_TREE_LOCK

            real4 posq1 = posq[atom1];
            LOAD_ATOM1_PARAMETERS
#ifdef USE_CUTOFF
            unsigned int j = (numTiles <= maxTiles ? interactingAtoms[pos*TILE_SIZE+tgx] : x*TILE_SIZE + tgx);
#else
            unsigned int j = x*TILE_SIZE + tgx;
#endif
            atomIndices[get_local_id(0)] = j;
            if (j < PADDED_NUM_ATOMS) {
                localData[localAtomIndex].posq = posq[j];
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

            }
            SYNC_WARPS;

#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(posq1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(local_posq[get_local_id(0)], blockCenterX)
                SYNC_WARPS;
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {

                    int localAtom2Index = tbx+tj;
                    real4 posq2 = local_posq[localAtom2Index];
                    real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndices[tbx+tj];



                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION_STORE1
                        }	


                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);

                    SYNC_WARPS;
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.


                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {

                    int localAtom2Index = tbx+tj;
                    real4 posq2 = localData[localAtom2Index].posq;
                    real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        int atom2 = atomIndices[tbx+tj];



                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_INTERACTION_STORE1
                        }


#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);

                    SYNC_WARPS;
                }


                SYNC_WARPS;
            }
        
	    RELEASE_TREE_LOCK
	}

#ifdef NOTNOW
	uint pp = ovAtomTreePointer[atom1];
	int children_count;
	do { children_count = atomic_xchg(&ovChildrenCount[pp], -1); } while(children_count < 0);//lock to protect from other work groups
	int endslot = ovChildrenStartIndex[pp] + children_count;
	for (uint i = 0; i < TILE_SIZE; i++){
	  if(OverlapBuffer[i].atom2 >= 0){
	    ovLevel[endslot] = 2; //two-body
	    ovVolume[endslot] = OverlapBuffer[i].gvol;
	    ovVSfp[endslot] = OverlapBuffer[i].sfp;
	    ovGamma1i[endslot] = OverlapBuffer[i].gamma;
	    ovLastAtom[endslot] = OverlapBuffer[i].atom2;
	    ovG[endslot] = OverlapBuffer[i].ovG;
	    ovDV1[endslot] = OverlapBuffer[i].ovDV1;
	    ovRootIndex[endslot] = pp;
	    ovChildrenStartIndex[endslot] = -1;
	    ovChildrenCount[endslot] = 0;
	    endslot += 1;
	    children_count += 1;
	  }
	}
	atomic_xchg(&ovChildrenCount[pp], children_count); //releases lock
#endif	

        pos++;
    }

}

//this kernel initializes the tree to be processed by ComputeOverlapTree()
//it assumes that 2-body overlaps are in place
__kernel void resetComputeOverlapTree(
    __global const int*   restrict ovAtomTreePointer,    //pointers to atom tree sections
    __global const int*   restrict ovChildrenStartIndex,
    __global const int*   restrict ovChildrenCount,
    __global const int*   restrict ovRootIndex,
    __global       int*   restrict ovProcessedFlag,
    __global       int*   restrict ovOKtoProcessFlag,
    __global       int*   restrict ovAtomTreeSize
){

  int atom = get_global_id(0);
  while( atom < PADDED_NUM_ATOMS){
    if(atom < NUM_ATOMS){
	int pp = ovAtomTreePointer[atom];
	ovProcessedFlag[pp] = 1; // used by ComputeOverlapTreeKernel() to skip 2-body processing, which is done by InitOverlapTree()
	ovOKtoProcessFlag[pp] = 0;
	int ic = ovChildrenStartIndex[pp];
	int nc = ovChildrenCount[pp];
	if(ic >= 0 && nc > 0){
	  for(int slot = ic; slot < ic+nc ; slot++){ 
	    ovProcessedFlag[slot] = 0; //flag 2-body overlaps as ready for processing.
	    ovOKtoProcessFlag[slot] = 1;
	  }
	}
	//last atom in tree section updates size of tree section (one over last children)
	if((atom+1) % ATOMS_PER_SECTION == 0 || (atom+1) == NUM_ATOMS){
	  int section = atom/ATOMS_PER_SECTION;
	  int first_atom = section*ATOMS_PER_SECTION;
	  ovAtomTreeSize[section] = ic + nc - ovAtomTreePointer[first_atom];
	}
    }
    atom += get_global_size(0);
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



__kernel void reduceovCountBuffer(int bufferSize, int numBuffers, 
				  __global const int*  restrict ovAtomTreePointer,    //pointers to atom trees
				  __global const int*  restrict ovAtomTreeSize,       //actual sizes
				  __global const int*  restrict ovAtomTreePaddedSize, //actual sizes
				  __global       int*  restrict ovChildrenStartIndex,
				  __global       int*  restrict ovChildrenCount,
				  __global       int*  restrict ovCountBuffer){
  unsigned int atom = get_global_id(0);
  unsigned int id_in_tree = get_local_id(0);
  unsigned int tree = get_group_id(0);
  unsigned int gsize = get_local_size(0);
  int totalSize = bufferSize*numBuffers;
  //used for the "scan" of children counts to get children start indexes
  //ATOMS_PER_SECTION is the workgroup size
  //  __local int children_count[ATOMS_PER_SECTION];
  __local uint temp[2*ATOMS_PER_SECTION];

  while (atom < PADDED_NUM_ATOMS) {
    int sum = 0;
    for (int i = atom; i < totalSize; i += bufferSize) sum += ovCountBuffer[i];
    ovCountBuffer[atom] = sum;
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  //setup children counts
  atom = get_global_id(0);
  while (atom < PADDED_NUM_ATOMS) {
    int pp = ovAtomTreePointer[atom];
    ovChildrenCount[pp] = ovCountBuffer[atom];
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  //setup children pointers
  atom = get_global_id(0);
  while (atom < PADDED_NUM_ATOMS) {
    int pp = ovAtomTreePointer[atom];
    int tree_ptr = tree*ovAtomTreePaddedSize[0];
    unsigned int sum = scan1Exclusive(ovChildrenCount[tree_ptr + id_in_tree], temp, gsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(atom < NUM_ATOMS) ovChildrenStartIndex[tree_ptr + id_in_tree] = tree_ptr + ovAtomTreeSize[tree] + sum;
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  //now reset back the counts so that they can be used as
  //counters by InitOverlapTree()
  atom = get_global_id(0);
  while (atom < PADDED_NUM_ATOMS) {
    int pp = ovAtomTreePointer[atom];
    ovChildrenCount[pp] = 0;
    atom += get_global_size(0);
  }
}


// insertion sort
void sortVolumes2body(
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
  uint atom_ptr = ovAtomTreePointer[get_global_id(0)];
  int size = ovChildrenCount[atom_ptr];
  int offset = ovChildrenStartIndex[atom_ptr];
  if(size > 0 && offset >= 0){
    sortVolumes2body(offset, size,
		     ovVolume,
		     ovVSfp,
		     ovGamma1i,
		     ovG,
		     ovDV1,
		     ovLastAtom);
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
  void ComputeOverlapTree(
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
  const uint tree = get_group_id(0);
  const uint tree_ptr = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
  __local          uint temp[2*OV_WORK_GROUP_SIZE];
  __local volatile uint nprocessed;
  __local volatile uint tree_size;
  __local volatile uint niterations;
  const uint gsize = OV_WORK_GROUP_SIZE;
  uint ov_count = 0;

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
	      COMPUTE_INTERACTION_COUNT
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
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


//this kernel completes the tree with 3-body and higher overlaps avoiding 2 passes over overlap volumes.
//Each workgroup of size OV_WORK_GROUP_SIZE is assigned to a tree section
//Then starting from the top of the tree section, for each slot i to process do:
//1. Retrieve siblings j and process overlaps i<j, count non-zero overlaps
//2. Do a parallel prefix sum (scan) of the array of counts to fill ovChildrenStartIndex[]
//3. Re-process the i<j overlaps and saves them starting at ovChildrenStartIndex[]
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
  void ComputeOverlapTree_1pass(
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
  const uint ntrees = get_global_size(0)/OV_WORK_GROUP_SIZE;
  const uint tree = get_group_id(0);
  const uint tree_ptr = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
  __local          uint temp[2*OV_WORK_GROUP_SIZE];
  __local volatile uint nprocessed;
  __local volatile uint tree_size;
  __local volatile uint niterations;
  const uint gsize = OV_WORK_GROUP_SIZE;

  const uint buffer_offset = tree * (buffer_size/ntrees);

  __local volatile uint n_buffer; //how many overlaps in buffer to process
  __local volatile uint buffer_pos[OV_WORK_GROUP_SIZE]; //where to store in temp. buffers
  __local volatile uint parent1_buffer[OV_WORK_GROUP_SIZE]; //tree slot of "i" overlap
  __local volatile uint level1_buffer[OV_WORK_GROUP_SIZE]; //overlap level of of "i" overlap
  __local volatile real4 posq1_buffer[OV_WORK_GROUP_SIZE]; //position of "i" overlap
  __local volatile real a1_buffer[OV_WORK_GROUP_SIZE]; //a parameter of "i" overlap
  __local volatile real v1_buffer[OV_WORK_GROUP_SIZE]; //volume of "i" overlap
  __local volatile real gamma1_buffer[OV_WORK_GROUP_SIZE]; //gamma parameter of "i" overlap
  __local volatile uint children_count[OV_WORK_GROUP_SIZE]; //number of children

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

      if(n_buffer > 0){ //nothing to process

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

}


/**
 * Initialize tree for rescanning, set Processed, OKtoProcess=1 for leaves and out-of-bound,
 */
__kernel void ResetRescanOverlapTree(
  __global const          int*   restrict ovAtomTreePointer,    //pointers to atom tree sections
  __global const          int*   restrict ovAtomTreeSize,    //pointers to atom tree sections
  __global const          int*   restrict ovAtomTreePaddedSize,    //pointers to atom tree sections
  __global                int*   restrict ovProcessedFlag,
  __global                int*   restrict ovOKtoProcessFlag
				     ){
  const uint local_id = get_local_id(0);
  const uint tree = get_group_id(0);
  const uint tree_ptr = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
  const uint tree_size = ovAtomTreeSize[tree];
  const uint gsize = OV_WORK_GROUP_SIZE;
  const uint endslot = tree_ptr + tree_size;

  for(int slot=tree_ptr+local_id; slot<endslot; slot+=gsize){
    ovProcessedFlag[slot] = (slot < endslot) ? 0 : 1;
  }
  for(int slot=tree_ptr+local_id; slot<endslot; slot+=gsize){
    ovOKtoProcessFlag[slot] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

//this kernel initializes the tree to be processed by RescanOverlapTree()
__kernel void InitRescanOverlapTree(
    __global const int*   restrict ovAtomTreePointer,    //pointers to atom tree sections
    __global const int*   restrict ovChildrenStartIndex,
    __global const int*   restrict ovChildrenCount,
    __global       int*   restrict ovProcessedFlag,
    __global       int*   restrict ovOKtoProcessFlag
){

  int atom = get_global_id(0);
  while( atom < PADDED_NUM_ATOMS){
    if(atom < NUM_ATOMS){
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
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

//this kernel recomputes the overlap volumes of the current tree
//it does not modify the tree in any other way
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
  void RescanOverlapTree(
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
  const uint tree = get_group_id(0);
  const uint tree_ptr = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
  const uint tree_size = ovAtomTreeSize[tree];
  const uint gsize = OV_WORK_GROUP_SIZE;
  __local uint nprocessed;

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
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 }

