#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)
#define Min_GVol (MIN_GVOL)
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
  ATOM_PARAMETER_DATA
} AtomData;

//this kernel initializes the tree with 1-body overlaps.
__kernel void InitOverlapTree_1body(
    __global const int*   restrict ovAtomTreePointer,    //pointers to atom trees
    __global const real4* restrict posq, //atomic positions
    __global const float* restrict radiusParam, //atomic radius
    __global const float* restrict gammaParam, //gamma
    __global const int*   restrict ishydrogenParam, //1=hydrogen atom

    __global       real* restrict GaussianExponent, //atomic Gaussian exponent
    __global       real* restrict GaussianVolume, //atomic Gaussian exponent
    __global       real* restrict AtomicGamma, //atomic Gaussian volume

    __global       int*   restrict ovLevel, //this and below define tree
    __global       real*  restrict ovVolume,
    __global       real*  restrict ovVSfp,
    __global       real*  restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,
    __global       int*   restrict ovLastAtom,
    __global       int*   restrict ovRootIndex,
    __global       int*   restrict ovChildrenStartIndex,
    __global       int*   restrict ovChildrenCount

){
    
  int atom = get_global_id(0);
  if(atom < PADDED_NUM_ATOMS){

    bool out = atom >= NUM_ATOMS;
    bool h = ishydrogenParam[atom] > 0;
    real r = radiusParam[atom];
    real a = KFC/(r*r);
    real v = (out || h) ? 0 : 4.f*PI*powr(r,3)/3.f;
    real g = (out || h) ? 0 : gammaParam[atom];
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

    atom += get_global_size(0);
  }



}

//this kernel initializes the tree with 2-body overlaps.
//It can probably work with any thread group size but is designed for the minimum group size of
//32 threads (1 warp). It locks the trees of group_size atoms, so a larger group size would have trouble
//acquiring sufficient locks.
//
// Note that in this kernel atom1 goes along y and atom2 goes along y to have atom1 < atom2
// Could it interfere with neighbor list/exlusions data structures?
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
    __global       int*  restrict ovChildrenCount
    ){

    const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
    const unsigned int warp = get_global_id(0)/TILE_SIZE;
    const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1);
    const unsigned int tbx = get_local_id(0) - tgx;
    __local AtomData localData[WORK_GROUP_SIZE];

    INIT_VARS
   
    // First loop: process tiles that contain exclusions.


    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
	unsigned int atom1 = y*TILE_SIZE + tgx;

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
	    
	    ACQUIRE_LOCK
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
		      COMPUTE_INTERACTION
		    }


#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif

                SYNC_WARPS;
            }

	    RELEASE_LOCK
	    SYNC_WARPS;
        }
        else { // This is an off-diagonal tile.

            const unsigned int localAtomIndex = get_local_id(0);
            unsigned int j = x*TILE_SIZE + tgx;
            localData[localAtomIndex].posq = posq[j];
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL

	    ACQUIRE_LOCK
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
                        COMPUTE_INTERACTION
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


	    RELEASE_LOCK
            SYNC_WARPS;
        }

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
    __local int atomIndices[WORK_GROUP_SIZE];
    __local volatile int skipTiles[WORK_GROUP_SIZE];
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
        if (includeTile) {
            unsigned int atom1 = y*TILE_SIZE + tgx;
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
	    ACQUIRE_LOCK
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
                            COMPUTE_INTERACTION
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
                            COMPUTE_INTERACTION
                        }


#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);

                    SYNC_WARPS;
                }


		RELEASE_LOCK
                SYNC_WARPS;
            }
        


        }
        pos++;
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

  int atom = get_global_id(0); //initial assignment
  while(atom < PADDED_NUM_ATOMS){    
    unsigned int atom_ptr = ovAtomTreePointer[atom];
    unsigned int size = ovChildrenCount[atom_ptr];
    unsigned int offset = ovChildrenStartIndex[atom_ptr];
    sortVolumes2body(offset, size,
		     ovVolume,
		     ovVSfp,
		     ovGamma1i,
		     ovG,
		     ovDV1,
		     ovLastAtom);
    atom += get_global_size(0);
  }
}

//this kernel completes the tree with 3-body and higher overlaps.
__kernel void ComputeOverlapTree(
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global       int* restrict ovAtomTreeSize,       //actual sizes
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
   __global       int*  restrict ovChildrenCount
   ){


  const unsigned int id = get_global_id(0);
  real4 posq1, posq2;
  int slot, endslot, level, newlevel;
  int index, tree_ptr, tree_size;
  int parent, start, end, this_sibling, atom2;
  real a1, v1, a2, v2;
  real4 delta;
  real r2, a12, deltai, p12, df, gvol, dgvol, dgvolv, ef;
  real4 c12;

  int atom = id; //initial assignment of atom tree, each thread gets a different tree
  while(atom < PADDED_NUM_ATOMS){//NUM_ATOMS){

    tree_ptr = ovAtomTreePointer[atom];
    tree_size = ovAtomTreeSize[atom];

    index =  1;//start with the first 2-body
    while(index < ovAtomTreePaddedSize[atom]){
      slot = tree_ptr + index;
      ovChildrenCount[slot] = 0;
      ovChildrenStartIndex[slot] = -1;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      if(index < tree_size && atom < NUM_ATOMS){
	level = ovLevel[slot];


	posq1.xyz = ovG[slot].xyz;
	a1 = ovG[slot].w;
	v1 = ovVolume[slot];
	real gamma1 = ovGamma1i[slot]; 
	
	parent = ovRootIndex[slot];
	// slot of first and last sibling
	start = ovChildrenStartIndex[parent];
	end  = start+ovChildrenCount[parent];
	
	// the sibling index of this sibling
	this_sibling = slot - start;
	// i<j scan over siblings
	for(int i= start + this_sibling + 1; i<end ; i++){
	  atom2 = ovLastAtom[i];
	  posq2 = posq[atom2];
	  a2 = global_gaussian_exponent[atom2];
	  v2 = global_gaussian_volume[atom2];
	  real gamma2 = global_atomic_gamma[atom2];
	  
	  //Gaussian overlap
	  delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
	  r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	  a12 = a1 + a2;
	  deltai = 1.0f/a12;
	  df = a1*a2*deltai;
	  ef = exp(-df*r2);
	  gvol = (v1*v2/powr(PI/df,1.5f))*ef;
	  dgvol = -2.0f*df*gvol; // (1/r)*dgvol/dr
	  dgvolv = v1 > 0 ? gvol/v1 : 0;
	  c12 = deltai*(a1*posq1 + a2*posq2);
	  //switching function
	  real s, sp;
          if(gvol > VolMinB ){
	    s = 1.0f;
	    sp = 0.0f;
	  }else if(gvol < VolMinA){
	    s = 0.0f;
	    sp = 0.0f;
	  }else{
	    real swd = 1.f/( VolMinB - VolMinA );
	    real swu = (gvol - VolMinA)*swd;
	    real swu2 = swu*swu;
	    real swu3 = swu*swu2;
	    s = swu3*(10.f-15.f*swu+6.f*swu2);
	    sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	   }
	  // switching function end
	  real sfp = sp*gvol + s;
	  gvol = s*gvol;

	  //add child to the tree
	  if(gvol > Min_GVol ){
	    endslot = tree_ptr + tree_size;
	    if(ovChildrenCount[slot] == 0) {//first child
	      atomic_xchg(&ovChildrenStartIndex[slot],endslot); //where the children are stored
	    }
	    atomic_inc(&ovChildrenCount[slot]); //increment children count
	    newlevel = level + 1;
	    atomic_xchg(&ovLevel[endslot], newlevel);
	    atomic_xchg(&ovLastAtom[endslot],atom2);
	    atomic_xchg(&ovRootIndex[endslot], slot);
	    ovVolume[endslot] = gvol;
	    ovVSfp[endslot] = sfp;
	    ovGamma1i[endslot] = gamma1 + gamma2;
	    ovG[endslot] = (real4)(c12.xyz, a12);
	    ovDV1[endslot] = (real4)(-delta.xyz*dgvol,dgvolv);
	    tree_size += 1;
	  }
	}

      }
      index += 1;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    //save tree size
    atomic_xchg(&ovAtomTreeSize[atom],tree_size);

    //next atom
    
    atom += get_global_size(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

}


//this kernel recomputes the overlap volumes of the current tree
__kernel void RescanOverlapTree(
   __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
   __global       int* restrict ovAtomTreeSize,       //actual sizes
   __global const int* restrict ovAtomTreePaddedSize, //padded allocated sizes
   __global       int* restrict ovAtomTreeLock,       //tree locks
   __global const real4* restrict posq, //atomic positions
   __global const real* restrict GaussianExponent, //atomic Gaussian exponent
   __global const real* restrict GaussianVolume, //atomic Gaussian prefactor
   __global const real* restrict AtomicGamma, //atomic Gaussian prefactor
   __global       int*  restrict ovLevel, //this and below define tree
   __global       real* restrict ovVolume,
    __global       real* restrict ovVSfp,
   __global       real* restrict ovGamma1i,
   __global       real4* restrict ovG,
   __global       real4* restrict ovDV1,
   
   __global       int*  restrict ovLastAtom,
   __global       int*  restrict ovRootIndex,
   __global       int*  restrict ovChildrenStartIndex,
   __global       int*  restrict ovChildrenCount
   ){


  const unsigned int id = get_global_id(0);
  real4 posq1, posq2;
  int slot, endslot;
  int index, tree_ptr, tree_size;
  real4 delta;
  real r2, a12, deltai, p12, df, gvol, dgvol, dgvolv, ef;
  real4 c12;

  int atom = id; //initial assignment of atom tree, each thread gets a different tree
  while(atom < PADDED_NUM_ATOMS){//NUM_ATOMS){

    tree_ptr = ovAtomTreePointer[atom];
    tree_size = ovAtomTreeSize[atom];

    index =  0;//start with the first 1-body
    while(index < ovAtomTreePaddedSize[atom]){
      slot = tree_ptr + index;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      if(index < tree_size && atom < NUM_ATOMS){
	posq1.xyz = ovG[slot].xyz;
	real a1 = ovG[slot].w;
	real gamma1 = ovGamma1i[slot]; 
	real v1 = ovVolume[slot];

	// slot of first and last children
	int start = ovChildrenStartIndex[slot];
	int end  = start+ovChildrenCount[slot];
	
	if(start>0){ //-1 is no children
	  // scan over children
	  for(int i= start ; i<end ; i++){
	    int atom2 = ovLastAtom[i];
	    posq2 = posq[atom2];
	    real a2 = GaussianExponent[atom2];
	    real v2 = GaussianVolume[atom2];
	    real gamma2 = AtomicGamma[atom2];
	    
	    //Gaussian overlap
	    delta = (real4) (posq2.xyz - posq1.xyz, 0.0f);
	    r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	    a12 = a1 + a2;
	    deltai = 1.0f/a12;
	    df = a1*a2*deltai;
	    ef = exp(-df*r2);
	    gvol = 0;
	    dgvol = 0;
	    dgvolv = 0;
	    if(v1 > 0 && v2 > 0) {
	      gvol = (v1*v2/powr(PI/df,1.5f))*ef;
	      dgvol = -2.0f*df*gvol; // (1/r)*dgvol/dr
	      dgvolv = gvol/v1;
	    }
	    //switching function
	    real s, sp;
	    if(gvol > VolMinB ){
	      s = 1.0f;
	      sp = 0.0f;
	    }else if(gvol < VolMinA){
	      s = 0.0f;
	      sp = 0.0f;
	    }else{
	      real swd = 1.f/( VolMinB - VolMinA );
	      real swu = (gvol - VolMinA)*swd;
	      real swu2 = swu*swu;
	      real swu3 = swu*swu2;
	      s = swu3*(10.f-15.f*swu+6.f*swu2);
	      sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	    }
	    // switching function end	    
	    real sfp = sp*gvol + s;
	    gvol = s*gvol;

	    c12 = deltai*(a1*posq1 + a2*posq2);
	    ovVolume[i] = gvol;
	    ovVSfp[i] = sfp;
	    ovGamma1i[i] = gamma1 + gamma2;
	    ovG[i] = (real4)(c12.xyz, a12);
	    ovDV1[i] = (real4)(-delta.xyz*dgvol,dgvolv);
	  }	  
	}

      }
      index += 1;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    //next atom
    
    atom += get_global_size(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

}
