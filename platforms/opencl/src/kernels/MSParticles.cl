#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif


/* memory locking functions from http://www.cmsoft.com.br/opencl-tutorial/opencl-c99-atomics/
   by Douglas Coimbra de Andrade.
   occupied = 0 <- lock available
   occupied = 1 <- lock busy

   I think they cause a deadlock on most devices when multiple threads in a warp try to acquire 
   the same semaphor. Okay if only the master thread of the warp acquires the lock. 
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

#define PI (3.14159265359f)

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


//version of scan for a warp


//=====================================================================
inline uint scan1InclusiveWarp(uint idata, __local uint *l_Data, uint local_id, uint size){
  uint pos = 2 * local_id - (local_id & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;
  barrier(CLK_LOCAL_MEM_FENCE);
  //mem_fence(CLK_LOCAL_MEM_FENCE);

  for(uint offset = 1; offset < size; offset <<= 1){
    //mem_fence(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    int t = l_Data[pos] + l_Data[pos - offset];
    //mem_fence(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    l_Data[pos] = t;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //mem_fence(CLK_LOCAL_MEM_FENCE);  
  return l_Data[pos];
}
inline uint scan1ExclusiveWarp(uint idata, __local uint *l_Data, uint local_id, uint size){
  return scan1InclusiveWarp(idata, l_Data, local_id, size) - idata;
}
//===================================================================

__kernel void TestScanWarp(__global uint* restrict input){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local uint temp[2*FORCE_WORK_GROUP_SIZE];

  uint nms1 = input[get_global_id(0)];
  uint sum = scan1ExclusiveWarp(nms1, &(temp[warp_in_group*2*TILE_SIZE]), tgx, TILE_SIZE);
  input[get_global_id(0)] = sum;
}




/* initializes counts */ 
__kernel void MSParticles1Reset(int ntiles, 
				__global int* restrict MScount,
				int size,
				__global real* restrict MSpVol0,
				__global real* restrict MSpVolvdW,
				__global real* restrict MSpVolLarge
				){
  uint id = get_global_id(0);
  while(id < ntiles){
    MScount[id] = 0;
    id += get_global_size(0);
  }
  id = get_global_id(0);
  while(id < size){
    MSpVol0[id] = 0.f;
    MSpVolvdW[id] = 0.f;
    MSpVolLarge[id] = 0.f;
    id += get_global_size(0);
  }
}


typedef struct {
  real4 posq;
  float radius;
  bool isheavy;
} AtomData;



/* Creates MS particles 1  */
__kernel void MSParticles1Store(
			       __global const real4* restrict posq, //atomic positions
			       __global const   int* restrict ishydrogenParam,
			       __global const  real* restrict radiusParam,
			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms,
			       unsigned int maxTiles,
			       __global const ushort2* exclusionTiles,
#else
			       unsigned int numTiles,
#endif
			       //MS particles info
			       int const                 MStile_size,
			       __global   int*  restrict MScount,
			       __global  real*  restrict MSpVol0,
			       __global  real*  restrict MSpVolLarge,
			       __global  real*  restrict MSpVolvdW,  
			       __global  real*  restrict MSpsspLarge,
			       __global  real*  restrict MSpsspvdW,  
			       __global  real4* restrict MSpPos,     
			       __global  int*   restrict MSpParent1, 
			       __global  int*   restrict MSpParent2, 
			       __global  real4* restrict MSpgder,    
			       __global  real4* restrict MSphder,    
			       __global  real*  restrict MSpfms,  
			       __global  real*  restrict MSpG0Large, 
			       __global  real*  restrict MSpG0vdW

){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local AtomData localData[FORCE_WORK_GROUP_SIZE];
  __local uint old_count[NWARPS_IN_GROUP];
  __local uint temp[2*FORCE_WORK_GROUP_SIZE];
  
  real radw = SOLVENT_RADIUS;
  real volw = 4.f*PI*radw*radw*radw/3.f;
  real vol_coeff = 0.17f;

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
       
    uint atom1 = y*TILE_SIZE + tgx;
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real radius1 = radiusParam[atom1];
    bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;
    
    uint j = x*TILE_SIZE + tgx;

    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].radius = radiusParam[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;

    SYNC_WARPS;

    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real r = sqrt(r2);
      real radius2 = localData[localAtom2Index].radius;
      bool isheavy2 = localData[localAtom2Index].isheavy;
      int nms1 = 0;

      
      int atom2 = x*TILE_SIZE + tj;
      
      bool compute = atom1 < NUM_ATOMS && atom2 < NUM_ATOMS;
      compute = compute && r2 < CUTOFF_SQUARED;
      compute = compute && isheavy1 && isheavy2;
      if(x == y) {
	compute = compute && atom1 < atom2;
      }
      real volms;
      real volmsw;
      real dms;
      real4 posms;
      real4 sder;
      real fms;
      real sigmasq;
      uint sum;
      if(compute){
	//MS particle between atom1 and atom2
	real q = sqrt(radius1*radius2)/radw;
	dms = radius1 + radius2 + 0.5f*radw;
	real volms0 = vol_coeff*q*q*volw;
	sigmasq = 0.25f*q*radw*radw;
	volms = volms0*exp(-0.5f*(r-dms)*(r-dms)/sigmasq);
	//store
	if(volms > VOLMINMSA){
	  //switching function
	  real s = 0, sp = 0;
	  if(volms > VOLMINMSB ){
	    s = 1.0f;
	    sp = 0.0f;
	  }else{
	    real swd = 1.f/(VOLMINMSB  - VOLMINMSA );
	    real swu = (volms - VOLMINMSA)*swd;
	    real swu2 = swu*swu;
	    real swu3 = swu*swu2;
	    s = swu3*(10.f-15.f*swu+6.f*swu2);
	    sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	  }
	  // switching function end
	  volmsw = volms*s;
	  sder = s + volms*sp;
	  fms = 0.5f * (1.f + (radius1 - radius2)/r);
	  posms = posq2 * fms + posq1 * (1.f - fms);
	  nms1 = 1;
	}
      }//if(compute)
      //count how many MS particles have been collected in the workgroup at this cycle 
      sum = scan1ExclusiveWarp(nms1, temp, tgx, TILE_SIZE);
      nms[localAtomIndex] = sum;
      SYNC_WARPS;
      //now subtracts sum of number of MS particles from previous warp to get the
      //number of MS particles collected in this warp
      int nsubs = (warp > 0) ? nms[warp*TILE_SIZE - 1] : 0; 
      sum -= nsubs;
      //update count of MS particles in each tile
      if(tgx == TILE_SIZE-1){//master thread in warp
	old_count[warp] = atomic_add(&(MScount[y]),sum+nms1);//sum+nms1 is the inclusive prefix sum
      }
      SYNC_WARPS;
      uint warp_istore = old_count[warp] + sum;
      if(nms1 > 0 && warp_istore < MStile_size){
	uint istore = y*MStile_size + warp_istore;
	MSpVol0[istore] = volmsw;
	MSpPos[istore] = posms;
	MSpParent1[istore] = atom1;
	MSpParent2[istore] = atom2;
	MSpgder[istore] = delta *  sder*(r-dms)*volms/(r*sigmasq);
	MSphder[istore] = delta *  0.5f*(radius1 - radius2)/(r2*r); 
	MSpfms[istore] = fms;
      }

      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
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
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real radius1 = radiusParam[atom1];
    bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;

#ifdef USE_CUTOFF
    uint j = interactingAtoms[pos*TILE_SIZE + tgx];
    atomIndices[get_local_id(0)] = j;
    if(j<PADDED_NUM_ATOMS){
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].radius = radiusParam[j];
      localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
    }
#else
    uint j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].radius = radiusParam[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
#endif
    
    SYNC_WARPS;
    
    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real r = sqrt(r2);
      real radius2 = localData[localAtom2Index].radius;
      bool isheavy2 = localData[localAtom2Index].isheavy;
      int nms1 = 0;
      
#ifdef USE_CUTOFF
      int atom2 = atomIndices[localAtom2Index];
#else
      int atom2 = x*TILE_SIZE + tj;
#endif
      bool compute = atom1 < NUM_ATOMS && atom2 < NUM_ATOMS;
      compute = compute && isheavy1 && isheavy2;
#ifdef USE_CUTOFF
      compute = compute && r2 < CUTOFF_SQUARED;
#else
      //when not using a neighbor list we are getting diagonal tiles here
      if(x == y) compute = compute && atom1 < atom2;
#endif


      real volms;
      real volmsw;
      real dms;
      real4 posms;
      real4 sder;
      real fms;
      real sigmasq;
      uint sum = 0;
      if(compute){
	//MS particle between atom1 and atom2
	real q = sqrt(radius1*radius2)/radw;
	dms = radius1 + radius2 + 0.5f*radw;
	real volms0 = vol_coeff*q*q*volw;
	sigmasq = 0.25f*q*radw*radw;
	volms = volms0*exp(-0.5f*(r-dms)*(r-dms)/sigmasq);
	//store
	if(volms > VOLMINMSA){
	  //switching function
	  real s = 0, sp = 0;
	  if(volms > VOLMINMSB ){
	    s = 1.0f;
	    sp = 0.0f;
	  }else{
	    real swd = 1.f/(VOLMINMSB  - VOLMINMSA );
	    real swu = (volms - VOLMINMSA)*swd;
	    real swu2 = swu*swu;
	    real swu3 = swu*swu2;
	    s = swu3*(10.f-15.f*swu+6.f*swu2);
	    sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	  }
	  // switching function end
	  volmsw = volms*s;
	  sder = s + volms*sp;
	  fms = 0.5f * (1.f + (radius1 - radius2)/r);
	  posms = posq2 * fms + posq1 * (1.f - fms);
	  nms1 = 1;
	}
      }//if(compute)
      //count how many MS particles have been collected in the workgroup at this cycle
#ifdef ONE_WARP_PER_GROUP
      sum = scan1Exclusive(nms1, temp, FORCE_WORK_GROUP_SIZE);
#else
      sum = scan1ExclusiveWarp(nms1, &(temp[warp_in_group*2*TILE_SIZE]), tgx, TILE_SIZE);
#endif

      //nms[localAtomIndex] = sum;
      //SYNC_WARPS;
      //now subtracts sum of number of MS particles from previous warp to get the
      //number of MS particles collected in this warp
      //int nsubs = (warp_in_group > 0) ? nms[warp_in_group*TILE_SIZE - 1] : 0; 
      //sum -= nsubs;
      //update count of MS particles in each tile
      if(tgx == TILE_SIZE-1){//master thread in warp
	//atomic_xchg(&(MScount[y]),sum);
	//atomic_add(&(MScount[y]),nms1);//sum+nms1 is the inclusive prefix sum
      	old_count[warp_in_group] = atomic_add(&(MScount[y]),sum+nms1);//sum+nms1 is the inclusive prefix sum
      }
#ifdef ONE_WARP_PER_GROUP
      SYNC_WARPS;
#else
      //the tile is processed by a single warp (in lockstep) in a group of warps, in lockstep.
      //Can't use a barrier to sync local memory because the other warps will not see it
      mem_fence(CLK_LOCAL_MEM_FENCE);
#endif
      uint warp_istore = old_count[warp_in_group] + sum;
      if(nms1 > 0 && warp_istore < MStile_size){
	uint istore = y*MStile_size + warp_istore;
	MSpVol0[istore] = volmsw;
	MSpPos[istore] = posms;
	MSpParent1[istore] = atom1;
	MSpParent2[istore] = atom2;
	MSpgder[istore] = delta *  sder*(r-dms)*volms/(r*sigmasq);
	MSphder[istore] = delta *  0.5f*(radius1 - radius2)/(r2*r); 
	MSpfms[istore] = fms;
      }
      
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }
    SYNC_WARPS;
    pos++; //new tile	
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


typedef struct {
  real4 g;//position and exponent of atomic gaussian (small radii)
  real v;//volume of atomic gaussian
  real galr; //gaussian exponent (large radii)
  real vlr;//volume of atomic gaussian with large radii
  bool isheavy;
  int parent1;
  int parent2;
} AtomDataG;

//switched overlap function between two gaussian densities 
inline real goverlap(real a1, real v1, real a2, real v2, real r2){
  real a12 = a1 + a2;
  real deltai = 1./a12;
  real df = a1*a2*deltai;
  real ef = exp(-df*r2);
  real dfp = df/PI;
  real gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef;
  real s = 0, sp = 0;
  if(gvol > VOLMINA){
    //switching function
    if(gvol > VOLMINB ){
      s = 1.0f;
      sp = 0.0f;
    }else{
      real swd = 1.f/(VOLMINB  - VOLMINA );
      real swu = (gvol - VOLMINA)*swd;
      real swu2 = swu*swu;
      real swu3 = swu*swu2;
      s = swu3*(10.f-15.f*swu+6.f*swu2);
      sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
    }
  }
  gvol = s*gvol;
  return gvol;
}


/* computes free volumes of MS particles */
__kernel void MSParticles1Vfree(
			       __global const real4* restrict posq, //atomic positions
			       __global const real* restrict GaussianExponent,
			       __global const real* restrict selfVolume,
			       __global const real* restrict GaussianExponentLargeR,
			       __global const real* restrict selfVolumeLargeR,
			       __global const   int* restrict ishydrogenParam,
			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms,
			       unsigned int maxTiles,
			       __global const ushort2* exclusionTiles,
#else
			       unsigned int numTiles,
#endif
			       //MS particles info
			       int const                 MStile_size,
			       __global   int*  restrict MScount,
			       __global  real*  restrict MSpVol0,
			       __global  real*  restrict MSpVolLarge,
			       __global  real*  restrict MSpVolvdW,  
			       __global  real*  restrict MSpsspLarge,
			       __global  real*  restrict MSpsspvdW,  
			       __global  real4* restrict MSpPos,     
			       __global  int*   restrict MSpParent1, 
			       __global  int*   restrict MSpParent2, 
			       __global  real4* restrict MSpgder,    
			       __global  real4* restrict MSphder,    
			       __global  real*  restrict MSpfms,  
			       __global  real*  restrict MSpG0Large, 
			       __global  real*  restrict MSpG0vdW,
			       __global  int*   restrict MSsemaphor

){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local AtomDataG localData[FORCE_WORK_GROUP_SIZE];
  
  real radw = SOLVENT_RADIUS;

  //cutoffs not implemented
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
       
    uint msparticle1 = y*TILE_SIZE + tgx;
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real radius1 = radiusParam[atom1];
    bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;
    
    uint j = x*TILE_SIZE + tgx;

    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].radius = radiusParam[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;

    SYNC_WARPS;

    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real r = sqrt(r2);
      real radius2 = localData[localAtom2Index].radius;
      bool isheavy2 = localData[localAtom2Index].isheavy;
      int nms1 = 0;

      
      int atom2 = x*TILE_SIZE + tj;
      
      bool compute = atom1 < NUM_ATOMS && atom2 < NUM_ATOMS;
      compute = compute && r2 < CUTOFF_SQUARED;
      compute = compute && isheavy1 && isheavy2;
      if(x == y) {
	compute = compute && atom1 < atom2;
      }
      real volms;
      real volmsw;
      real dms;
      real4 posms;
      real4 sder;
      real fms;
      real sigmasq;
      uint sum;
      if(compute){
	//MS particle between atom1 and atom2
	real q = sqrt(radius1*radius2)/radw;
	dms = radius1 + radius2 + 0.5f*radw;
	real volms0 = vol_coeff*q*q*volw;
	sigmasq = 0.25f*q*radw*radw;
	volms = volms0*exp(-0.5f*(r-dms)*(r-dms)/sigmasq);
	//store
	if(volms > VOLMINMSA){
	  //switching function
	  real s = 0, sp = 0;
	  if(volms > VOLMINMSB ){
	    s = 1.0f;
	    sp = 0.0f;
	  }else{
	    real swd = 1.f/(VOLMINMSB  - VOLMINMSA );
	    real swu = (volms - VOLMINMSA)*swd;
	    real swu2 = swu*swu;
	    real swu3 = swu*swu2;
	    s = swu3*(10.f-15.f*swu+6.f*swu2);
	    sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	  }
	  // switching function end
	  volmsw = volms*s;
	  sder = s + volms*sp;
	  fms = 0.5f * (1.f + (radius1 - radius2)/r);
	  posms = posq2 * fms + posq1 * (1.f - fms);
	  nms1 = 1;
	}
      }//if(compute)
      //count how many MS particles have been collected in the workgroup at this cycle 
      sum = scan1Exclusive(nms1, temp, FORCE_WORK_GROUP_SIZE);
      nms[localAtomIndex] = sum;
      barrier(CLK_LOCAL_MEM_FENCE);
      //now subtracts sum of number of MS particles from previous warp to get the
      //number of MS particles collected in this warp
      int nsubs = (warp > 0) ? nms[warp*TILE_SIZE - 1] : 0; 
      sum -= nsubs;
      uint istore = warp*MStile_size + MScount[warp] + sum;
      if(nms1 > 0){
	MSpVol0[istore] = volmsw;
	MSpPos[istore] = posms;
	MSpParent1[istore] = atom1;
	MSpParent2[istore] = atom2;
	MSpgder[istore] = delta *  sder*(r-dms)*volms/(r*sigmasq);
	MSphder[istore] = delta *  0.5f*(radius1 - radius2)/(r2*r); 
	MSpfms[istore] = fms;
      }
      //update count of MS particles in each warp
      if(tgx == TILE_SIZE-1){//master thread in warp
	atomic_add(&(MScount[warp]),sum+nms1);//sum+nms1 is the inclusive prefix sum	  
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
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
#else
    // find x and y coordinates of the tile such that y <= x
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }
#endif

    uint iter = 0;
    uint maxiter = (x==y) ? 1 : 2;
    while(iter < maxiter){
    if(iter == 1) {
      //reverse x and y
      int z = x;
      x = y;
      y = z;
    }
    //load atom data for this x,y tile
#ifdef USE_CUTOFF
    uint j = interactingAtoms[pos*TILE_SIZE + tgx];
    atomIndices[get_local_id(0)] = j;
    if(j<PADDED_NUM_ATOMS){
      localData[localAtomIndex].g = posq[j];
      localData[localAtomIndex].g.w = GaussianExponent[j];
      localData[localAtomIndex].v = selfVolume[j];
      localData[localAtomIndex].galr = GaussianExponentLargeR[j];
      localData[localAtomIndex].vlr = selfVolumeLargeR[j];
      localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
    }
#else
    uint j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].g = posq[j];
    localData[localAtomIndex].g.w = GaussianExponent[j];
    localData[localAtomIndex].v = selfVolume[j];
    localData[localAtomIndex].galr = GaussianExponentLargeR[j];
    localData[localAtomIndex].vlr = selfVolumeLargeR[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
#endif
    
    //number of blocks of non-zero MS particles in this tile 
    uint nmsblock = MScount[y]/TILE_SIZE + 1;
    //stripes assignment of blocks across warps to minimize chance of collisions
    uint imsblock = warp % nmsblock;
    for(int mj = 0; mj < nmsblock; mj++){
      uint msid = imsblock*TILE_SIZE + tgx;
      uint msparticle1 = y*MStile_size + msid;

      // Load MS atom data for this x,y tile.
      real4 posq1 = MSpPos[msparticle1];
      real v1 = MSpVol0[msparticle1];
      real a1 = KFC/(radw*radw);
      int  parent1 = MSpParent1[msparticle1];
      int  parent2 = MSpParent2[msparticle1];
      real msvol1 = 0.f;
      real msvol1lr = 0.f;
	
      SYNC_WARPS;
	
      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].g;
	real a2 = localData[localAtom2Index].g.w;
	real v2 = localData[localAtom2Index].v;
	real a2lr = localData[localAtom2Index].galr;
	real v2lr = localData[localAtom2Index].vlr;
	bool isheavy2 = localData[localAtom2Index].isheavy;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real r = sqrt(r2);
	
#ifdef USE_CUTOFF
	int atom2 = atomIndices[localAtom2Index];
#else
	int atom2 = x*TILE_SIZE + tj;
#endif
	bool compute = msid < MScount[y] && atom2 < NUM_ATOMS ;
	compute = compute && isheavy2 && atom2 != parent1 && atom2 != parent2; 
#ifdef USE_CUTOFF
	compute = compute && r2 < CUTOFF_SQUARED;
#endif
	if(compute){
	  msvol1   += goverlap(a1, v1, a2,   v2,   r2);
	  msvol1lr += goverlap(a1, v1, a2lr, v2lr, r2);
	}
	  
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }
      //updates MS volumes
      if(tgx == TILE_SIZE-1){
	GetSemaphor(&(MSsemaphor[msparticle1]));
      }
      SYNC_WARPS;
      MSpVolvdW[msparticle1] += msvol1;
      MSpVolLarge[msparticle1] += msvol1lr;
      SYNC_WARPS;
      if(tgx == TILE_SIZE-1){
	ReleaseSemaphor(&(MSsemaphor[msparticle1]));
      }
      SYNC_WARPS;
      
      //new block of MS particles
      imsblock = (imsblock + 1) % nmsblock;
    }

    iter++; //symmetric tile
    SYNC_WARPS;
    }
    SYNC_WARPS;
    pos++; //new tile	
  }


#ifdef NOTNOW
  uint ntiles = PADDED_NUM_ATOMS/TILE_SIZE;
  uint n2dtiles = ntiles*ntiles;
  int pos = (int) (warp*(long)n2dtiles/totalWarps);
  int end = (int) ((warp+1)*(long)n2dtiles/totalWarps);
  while (pos < end) {
    int y = pos/ntiles;
    int x = pos % ntiles;

    //load atom data for this x,y tile
    uint j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].g = posq[j];
    localData[localAtomIndex].g.w = global_gaussian_exponent[j];
    localData[localAtomIndex].v = selfVolume[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
    
    //number of blocks of non-zero MS particles in this tile 
    uint nmsblock = MScount[y]/TILE_SIZE + 1;
    //straddle assignment of blocks across warps to minimize chance of collisions
    uint imsblock = warp % nmsblock;
    for(int mj = 0; mj < nmsblock; mj++){
      uint msid = imsblock*TILE_SIZE + tgx;
      uint msparticle1 = y*MStile_size + msid;

      // Load MS atom data for this x,y tile.
      real4 posq1 = MSpPos[msparticle1];
      real v1 = MSpVol0[msparticle1];
      real a1 = KFC/(radw*radw);
      int  parent1 = MSpParent1[msparticle1];
      int  parent2 = MSpParent2[msparticle1];
      real msvol1 = 0.f;
	
      SYNC_WARPS;
	
      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].g;
	real a2 = localData[localAtom2Index].g.w;
	real v2 = localData[localAtom2Index].v;
	bool isheavy2 = localData[localAtom2Index].isheavy;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real r = sqrt(r2);
	real gvol = 0.f;
	
	int atom2 = x*TILE_SIZE + tj;

	bool compute = msid < MScount[y] && atom2 < NUM_ATOMS ;
	compute = compute && isheavy2 && atom2 != parent1 && atom2 != parent2; 

	if(compute){	    
	  real a12 = a1 + a2;
	  real deltai = 1./a12;
	  real df = a1*a2*deltai;
	  real ef = exp(-df*r2);
	  real dfp = df/PI;
	  gvol = v1*v2*dfp*dfp*rsqrt(dfp)*ef;
	  real s = 0, sp = 0;
	  if(gvol > VOLMINA){
	    //switching function
	    if(gvol > VOLMINB ){
	      s = 1.0f;
	      sp = 0.0f;
	    }else{
	      real swd = 1.f/(VOLMINB  - VOLMINA );
	      real swu = (gvol - VOLMINA)*swd;
	      real swu2 = swu*swu;
	      real swu3 = swu*swu2;
	      s = swu3*(10.f-15.f*swu+6.f*swu2);
	      sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
	    }
	  }
	  gvol = s*gvol;
	}
	msvol1 += gvol;
	  
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }
      //updates MS volumes
      if(tgx == TILE_SIZE-1){
	GetSemaphor(&(MSsemaphor[msparticle1]));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      MSpVolvdW[msparticle1] += msvol1;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(tgx == TILE_SIZE-1){
	ReleaseSemaphor(&(MSsemaphor[msparticle1]));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //new block of MS particles
      imsblock = (imsblock + 1) % nmsblock;
    }

    SYNC_WARPS;
    pos++; //new tile	
  }
#endif
  
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

/* compute switched volumes of MS particles */ 
__kernel void MSParticles1Vols(int const                 ntiles, 
			       int const                 MStile_size,
			       __global const  int*  restrict MScount,
			       __global const real*  restrict MSpVol0,
			       __global       real*  restrict MSpVolvdW,
			       __global       real*  restrict MSpVolLarge
			       ){
  uint id = get_global_id(0);
  while(id < ntiles*MStile_size){
    //from vdw radii
    real gvol  = MSpVol0[id] - MSpVolvdW[id];
    if(gvol > VOLMINMSA){
      //switching function
      real s = 0, sp = 0;
      if(gvol > VOLMINMSB ){
	s = 1.0f;
	sp = 0.0f;
      }else{
	real swd = 1.f/( VOLMINMSB - VOLMINMSA );
	real swu = (gvol - VOLMINMSA)*swd;
	real swu2 = swu*swu;
	real swu3 = swu*swu2;
	s = swu3*(10.f-15.f*swu+6.f*swu2);
	sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
      }
      // switching function end
      gvol = s*gvol;
    }else{
      gvol = 0.f;
    }
    MSpVolvdW[id] = gvol;
    
    //from large radii
    gvol = MSpVol0[id] - MSpVolLarge[id];
    if(gvol > VOLMINMSA){
      //switching function
      real s = 0, sp = 0;
      if(gvol > VOLMINMSB ){
	s = 1.0f;
	sp = 0.0f;
      }else{
	real swd = 1.f/( VOLMINMSB - VOLMINMSA );
	real swu = (gvol - VOLMINMSA)*swd;
	real swu2 = swu*swu;
	real swu3 = swu*swu2;
	s = swu3*(10.f-15.f*swu+6.f*swu2);
	sp = swd*30.f*swu2*(1.f - 2.f*swu + swu2);
      }
      // switching function end
      gvol = s*gvol;
    }else{
      gvol = 0.f;
    }
    MSpVolLarge[id] = gvol;
      
    id += get_global_size(0);
  }
}

/* counts number of non-zero vdw volume MS particles in each tile */ 
__kernel void MSParticles2Count(int const                 ntiles, 
				int const                 MStile_size,
				__global       int*  restrict MScount,
				__global       int*  restrict MSptr,			       
				__global const real*  restrict MSpVolvdW
				){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local uint temp[2*FORCE_WORK_GROUP_SIZE];
  __local int ptr[FORCE_WORK_GROUP_SIZE];
  const uint nsections = MStile_size/TILE_SIZE;
  
  int pos = (int) (warp*(long)ntiles/totalWarps);
  int end = (int) ((warp+1)*(long)ntiles/totalWarps);
  while(pos < end){
    if(tgx == TILE_SIZE - 1) MScount[pos] = 0;
    uint section = 0;
    while(section < nsections){
      uint nms1 = 0;
      ptr[get_local_id(0)] = -1;
      uint j = pos*MStile_size + section*TILE_SIZE + tgx;
      if(MSpVolvdW[j] > FLT_MIN) {
	nms1 += 1;
	ptr[get_local_id(0)] = j;
      }
#ifdef ONE_WARP_PER_GROUP
      uint sum = scan1Inclusive(nms1, temp, FORCE_WORK_GROUP_SIZE);
#else
      uint sum = scan1InclusiveWarp(nms1, &(temp[warp_in_group*2*TILE_SIZE]), tgx, TILE_SIZE);
#endif
      if(tgx == TILE_SIZE - 1) {
	int idx = 0;
	for(int k=0; k<TILE_SIZE; k++){
	  int p = warp_in_group*TILE_SIZE + k;
	  if(ptr[p] >= 0){
	    MSptr[pos*MStile_size + MScount[pos] + idx++] = ptr[p];
	  }
	}
	MScount[pos] += sum;
      }
      SYNC_WARPS;
      section++; //new section
    }
    SYNC_WARPS;
    pos++;//new tile
  }
}

__kernel void MSResetTreeCount(unsigned const int num_trees,
			       __global       int*   restrict ovNumAtomsInTree,
			       __global       int*   restrict ovAtomTreeSize
			       ){
  uint tree = get_global_id(0);
  while(tree<num_trees){
    ovNumAtomsInTree[tree] = 0;
    ovAtomTreeSize[tree] = 0;
    tree += get_global_size(0);
  }
}


//this kernel initializes the MS overalp tree with 1-body particles
__kernel void MSInitOverlapTree_1body(
    int const                 ntiles, 
    int const                 MStile_size,
    __global const int*    restrict MScount,
    __global const int*    restrict MSptr,			       
    __global const real*   restrict MSpVol,
    __global const real4*  restrict MSpPos,
    __global       real*   restrict MSpGaussExponent,
             const float            ms_gamma_value,
    __global       real*   restrict MSpGamma,
    

    
    unsigned const int              num_trees,
    unsigned const int              reset_tree_size,
    __global       int*   restrict ovTreePointer, 
    __global       int*   restrict ovNumAtomsInTree,
    __global       int*   restrict ovFirstAtom, 
    __global       int*   restrict ovAtomTreeSize,    //sizes of tree sections
    __global       int*   restrict NIterations,      
    __global const int*   restrict ovAtomTreePaddedSize, 
    __global       int*   restrict ovAtomTreePointer,    //pointers to atoms in tree
    __global       int*   restrict ovLevel, //this and below define tree
    __global       real*  restrict ovVolume,
    __global       real*  restrict ovVsp,
    __global       real*  restrict ovVSfp,
    __global       real*  restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,
    __global       int*   restrict ovLastAtom,
    __global       int*   restrict ovRootIndex,
    __global       int*   restrict ovChildrenStartIndex,
    __global volatile int*   restrict ovChildrenCount
){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  const uint nms_slots = ntiles*MStile_size;
  const real radw = SOLVENT_RADIUS;
  
  //    if(id == 0) ovNumAtomsInTree[section] = 0; need to reset tree
  int pos = (int) (warp*(long)ntiles/totalWarps);
  int end = (int) ((warp+1)*(long)ntiles/totalWarps);
  while(pos < end && pos < ntiles){

    //load ms atoms in this tile onto one of the trees
    int ims = tgx;
    while(ims < MScount[pos] ){
      int msatom = MSptr[pos*MStile_size + ims];
      int tree = (msatom*num_trees)/nms_slots;
      int old_count = atomic_inc(&(ovAtomTreeSize[tree]));
      atomic_inc(&(ovNumAtomsInTree[tree]));
      int slot = ovTreePointer[tree] + old_count;
      real a = KFC/(radw*radw);
      real v = MSpVol[msatom];
      real g = ms_gamma_value;
      MSpGaussExponent[msatom] = a;
      MSpGamma[msatom] = g;
      ovAtomTreePointer[msatom] = slot;
      if(slot - ovTreePointer[tree] < ovAtomTreePaddedSize[tree]){
	ovLevel[slot] = 1;
	ovVolume[slot] = v;
	ovVsp[slot] = 1;
	ovVSfp[slot] = 1;
	ovGamma1i[slot] = g;
	ovG[slot] = (real4)(MSpPos[msatom].xyz,a);
	ovDV1[slot] = (real4)0.f;
	ovLastAtom[slot] = msatom;
      }
      ims += TILE_SIZE;
    }

    pos++;//new tile
  }
}

#ifdef SUPPORTS_64_BIT_ATOMICS
//initializes accumulators for atomic MS energies
__kernel void MSinitEnergyBuffer(__global  long* restrict MSEnergyBuffer_long){
  uint iatom = get_global_id(0);
  while(iatom < PADDED_NUM_ATOMS){
    MSEnergyBuffer_long[iatom] = 0;
    iatom += get_global_size(0);
  }
}

//transfer MS energies to OpenMM's energy buffer
__kernel void MSupdateEnergyBuffer(__global  long* restrict MSEnergyBuffer_long,
				   __global mixed* restrict energyBuffer){
  real scale = 1/(real) 0x100000000;
  uint iatom = get_global_id(0);
  while(iatom < PADDED_NUM_ATOMS){
    energyBuffer[iatom] = scale*MSEnergyBuffer_long[iatom];
    iatom += get_global_size(0);
  }
}
#endif
			       
//update energy and force buffers with MS energies and forces 
__kernel void MSupdateSelfVolumesForces(int update_energy,
					int const             padded_num_atoms,
					int const             ntiles, 
					int const             MStile_size,
					__global const int*   restrict MScount,
					__global const int*   restrict MSptr,
					__global const int*   restrict ovAtomTreePointer,
					__global const real*  restrict ovVolEnergy,
					__global const real4* restrict grad, //gradient wrt to atom positions and volume
				        __global const real4* restrict posq, //atomic positions
					__global  int*   restrict MSpParent1, 
					__global  int*   restrict MSpParent2,
					__global  int*   restrict AtomSemaphor,
					__global  real4* restrict MSphder,
					__global  real*  restrict MSpfms,				
#ifdef SUPPORTS_64_BIT_ATOMICS
				        __global long*   restrict forceBuffers,
					__global long*   restrict MSEnergyBuffer_long
#else
					__global real4*  restrict forceBuffers,
					__global mixed*  restrict energyBuffer
#endif
){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int id = get_global_id(0);

  int pos = (int) (warp*(long)ntiles/totalWarps);
  int end = (int) ((warp+1)*(long)ntiles/totalWarps);
  while(pos < end && pos < ntiles){

    //load ms atoms in this tile onto one of the trees
    int ims = tgx;
    while(ims < MScount[pos] ){
      int msatom = MSptr[pos*MStile_size + ims];
      int parent1 = MSpParent1[msatom];
      int parent2 = MSpParent2[msatom];
      real4 dist = (real4) (posq[parent2].xyz - posq[parent1].xyz, 0);
      real4 hder = MSphder[msatom];
      real fms = MSpfms[msatom];
      real gms = 1. - fms;
      real evprod = -dot(grad[msatom],dist);
      real4 force1 = evprod * hder - gms * grad[msatom];
      real4 force2 = evprod * hder - fms * grad[msatom];
#ifdef SUPPORTS_64_BIT_ATOMICS
      atom_add(&forceBuffers[parent1                     ], (long)(force1.x*0x100000000));
      atom_add(&forceBuffers[parent1 +   padded_num_atoms], (long)(force1.y*0x100000000));
      atom_add(&forceBuffers[parent1 + 2*padded_num_atoms], (long)(force1.z*0x100000000));
      atom_add(&forceBuffers[parent2                     ], (long)(force2.x*0x100000000));
      atom_add(&forceBuffers[parent2 +   padded_num_atoms], (long)(force2.y*0x100000000));
      atom_add(&forceBuffers[parent2 + 2*padded_num_atoms], (long)(force2.z*0x100000000));
      if(update_energy > 0){
	// arbitrarily assigns energy to first parent
	uint slot = ovAtomTreePointer[msatom];
	atom_add(&MSEnergyBuffer_long[parent1], (long)(ovVolEnergy[slot]*0x100000000));
      }
#else
      //in most cases the semaphor will deadlock for a synchronized warp of size greater than 1
      GetSemaphor(&(AtomSemaphor[parent1]));
      GetSemaphor(&(AtomSemaphor[parent2]));
      forceBuffers[parent1].xyz += force1.xyz;
      forceBuffers[parent2].xyz += force2.xyz;
      if(update_energy > 0){
	uint slot = ovAtomTreePointer[msatom];
	energyBuffer[parent1] += ovVolEnergy[slot];
      }
      ReleaseSemaphor(&(AtomSemaphor[parent1]));
      ReleaseSemaphor(&(AtomSemaphor[parent2]));
#endif
      ims += TILE_SIZE;
    }
    pos++;//new tile
  }
}

//same as MSInit 1-body but for a rescan
__kernel void MSInitRescanOverlapTree_1body(
    int const                 ntiles, 
    int const                 MStile_size,
    __global const int*    restrict MScount,
    __global const int*    restrict MSptr,			       
    __global const real*   restrict MSpVol,
    __global const real4*  restrict MSpPos,
    __global       real*   restrict MSpGaussExponent,
             const float            ms_gamma_value,
    __global       real*   restrict MSpGamma,
    

    
    unsigned const int              num_trees,
    unsigned const int              reset_tree_size,
    __global       int*   restrict ovTreePointer, 
    __global       int*   restrict ovNumAtomsInTree,
    __global       int*   restrict ovFirstAtom, 
    __global       int*   restrict ovAtomTreeSize,    //sizes of tree sections
    __global       int*   restrict NIterations,      
    __global const int*   restrict ovAtomTreePaddedSize, 
    __global       int*   restrict ovAtomTreePointer,    //pointers to atoms in tree
    __global       int*   restrict ovLevel, //this and below define tree
    __global       real*  restrict ovVolume,
    __global       real*  restrict ovVsp,
    __global       real*  restrict ovVSfp,
    __global       real*  restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,
    __global       int*   restrict ovLastAtom,
    __global       int*   restrict ovRootIndex,
    __global       int*   restrict ovChildrenStartIndex,
    __global volatile int*   restrict ovChildrenCount
){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int warp_in_group = get_local_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //id in warp
  const unsigned int tbx = get_local_id(0) - tgx;           //start of warp
  const unsigned int localAtomIndex = get_local_id(0);
  const uint nms_slots = ntiles*MStile_size;
  const real radw = SOLVENT_RADIUS;
  
  //    if(id == 0) ovNumAtomsInTree[section] = 0; need to reset tree
  int pos = (int) (warp*(long)ntiles/totalWarps);
  int end = (int) ((warp+1)*(long)ntiles/totalWarps);
  while(pos < end && pos < ntiles){

    //load ms atoms in this tile onto one of the trees
    int ims = tgx;
    while(ims < MScount[pos] ){
      int msatom = MSptr[pos*MStile_size + ims];
      int tree = (msatom*num_trees)/nms_slots;
      int slot = ovAtomTreePointer[msatom];
      real a = KFC/(radw*radw);
      real v = MSpVol[msatom];
      real g = ms_gamma_value;
      MSpGaussExponent[msatom] = a;
      MSpGamma[msatom] = g;
      if(slot - ovTreePointer[tree] < ovAtomTreePaddedSize[tree]){
	ovLevel[slot] = 1;
	ovVolume[slot] = v;
	ovVsp[slot] = 1;
	ovVSfp[slot] = 1;
	ovGamma1i[slot] = g;
	ovG[slot] = (real4)(MSpPos[msatom].xyz,a);
	ovDV1[slot] = (real4)0.f;
	ovLastAtom[slot] = msatom;
      }
      ims += TILE_SIZE;
    }

    pos++;//new tile
  }
}



#define Min_GVol (MIN_GVOL)
#define VolMin0 (VOLMIN0)
#define VolMinA (VOLMINA)
#define VolMinB (VOLMINB)

typedef struct {
  real4 posq;
  int   ov_count;
  ATOM_PARAMETER_DATA
} AtomDataOv;


//this kernel counts the no. of 2-body overlaps for each MS particle, stores in ovChildrenCount
__kernel void MSInitOverlapTreeCount(
    int const                 ntiles, 
    int const                 MStile_size,
    __global const int*    restrict MScount,
    __global const int*    restrict MSptr,			       
				       
    __global const int* restrict ovAtomTreePointer,    //pointers to atom trees
    __global const real4* restrict posq, //atomic positions //MSpPos
    __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent //MSpGaussExponent
    __global const real* restrict global_gaussian_volume, //atomic Gaussian volume //MSpVolvdW
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
  __local AtomDataOv localData[FORCE_WORK_GROUP_SIZE];
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
    //number of blocks of non-zero MS particles in this tile 
    uint nmsblock1 = MScount[y]/TILE_SIZE + 1;
    uint imsblock1 = 0;
    for(int mj1 = 0; mj1 < nmsblock1; mj1++){
      uint msid1 = imsblock1*TILE_SIZE + tgx;
      uint atom1 = 0;
      int tree_pointer1 = 0;
      if(msid1 < MScount[y]){
	atom1 = MSptr[y*MStile_size + msid1];
	tree_pointer1 = ovAtomTreePointer[atom1];
      }
      // Load MS atom data for this tile.
#ifndef USE_CUTOFF
      //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
      int parent_slot = tree_pointer1;
#endif
      real4 posq1 = posq[atom1];
      real a1 = global_gaussian_exponent[atom1];
      real v1 = global_gaussian_volume[atom1];

#ifdef USE_CUTOFF
      //must figure out x from interactingAtoms[]
      /*
      uint j = interactingAtoms[pos*TILE_SIZE + tgx];
      atomIndices[get_local_id(0)] = j;
      if(j<PADDED_NUM_ATOMS){
	localData[localAtomIndex].posq = posq[j];
	localData[localAtomIndex].g.w = global_gaussian_exponent[j];
	localData[localAtomIndex].v = global_gaussian_volume[j];
	localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
      }
      */
#endif      

      uint nmsblock2 = MScount[x]/TILE_SIZE + 1;
      uint imsblock2 = 0;
      for(int mj2 = 0; mj2 < nmsblock2; mj2++){
	uint msid2 = imsblock2*TILE_SIZE + tgx;
	uint j = (msid2 < MScount[x]) ? MSptr[x*MStile_size + msid2] : 0;
	
	localData[localAtomIndex].posq = posq[j];
	localData[localAtomIndex].g.w = global_gaussian_exponent[j];
	localData[localAtomIndex].v = global_gaussian_volume[j];

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
	  int atom2 = atomIndices[localAtom2Index]; //?
	  int tree_pointer2 =  localData[localAtom2Index].tree_pointer;//??
#else
	  msid2 = imsblock2*TILE_SIZE + tj;
	  int atom2 = (msid2 < MScount[x]) ? MSptr[x*MStile_size + msid2] : 0;
#endif
	  bool compute = msid1 < MScount[y] &&  msid2 < MScount[x];
#ifdef USE_CUTOFF
	  compute = compute && r2 < CUTOFF_SQUARED;
#else
	  //when not using a neighbor list we are getting diagonal tiles here
	  if(x == y) compute = compute && atom1 < atom2;
#endif
	  if (compute){
#ifdef USE_CUTOFF
	    //the parent is taken as the atom with the smaller index
	    int parent_slot = (atom1 < atom2) ? tree_pointer1 : tree_pointer2;
#endif
	    COMPUTE_INTERACTION_COUNT
	   }
	  tj = (tj + 1) & (TILE_SIZE - 1);
	  SYNC_WARPS;
	}

	imsblock2++;
	SYNC_WARPS;
      }//mj2
      
      imsblock1++;
      SYNC_WARPS;
    }//mj1

    SYNC_WARPS;
    pos++;
  }       

}



//this kernel counts the no. of 2-body overlaps for each atom, stores in ovChildrenCount
__kernel void MSInitOverlapTree(
    int const                 ntiles, 
    int const                 MStile_size,
    __global const int*    restrict MScount,
    __global const int*    restrict MSptr,		
				
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
    __global       real* restrict ovVsp,
    __global       real* restrict ovVSfp,
    __global       real* restrict ovGamma1i,
    __global       real4* restrict ovG,
    __global       real4* restrict ovDV1,

    __global       int*  restrict ovLastAtom,
    __global       int*  restrict ovRootIndex,
    __global       int*  restrict ovChildrenStartIndex,
    __global       int*  restrict ovChildrenCount,
    __global       int*  restrict ovChildrenCountTop,
    __global       int*  restrict ovChildrenCountBottom,
    __global       int*  restrict PanicButton
){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  __local AtomDataOv localData[FORCE_WORK_GROUP_SIZE];
  const unsigned int localAtomIndex = get_local_id(0);

  INIT_VARS

  if(PanicButton[0] > 0) return;
    
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

    //number of blocks of non-zero MS particles in this tile 
    uint nmsblock1 = MScount[y]/TILE_SIZE + 1;
    uint imsblock1 = 0;
    for(int mj1 = 0; mj1 < nmsblock1; mj1++){
      uint msid1 = imsblock1*TILE_SIZE + tgx;
      uint atom1 = 0;
      int tree_pointer1 = 0;
      if(msid1 < MScount[y]){
	atom1 = MSptr[y*MStile_size + msid1];
	tree_pointer1 = ovAtomTreePointer[atom1];
      }
      // Load MS atom data for this tile.
#ifndef USE_CUTOFF
      //the parent is taken as the atom with the smaller index: w/o cutoffs atom1 < atom2 because y<x
      int parent_slot = tree_pointer1;
      int parent_children_start = ovChildrenStartIndex[parent_slot];
#endif
      real4 posq1 = posq[atom1];
      LOAD_ATOM1_PARAMETERS

#ifdef USE_CUTOFF
      //must figure out x from interactingAtoms[]
      /*
      uint j = interactingAtoms[pos*TILE_SIZE + tgx];
      atomIndices[get_local_id(0)] = j;
      if(j<PADDED_NUM_ATOMS){
	localData[localAtomIndex].posq = posq[j];
	localData[localAtomIndex].tree_pointer = ovAtomTreePointer[j];
	LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
      }
      */
#endif

      uint nmsblock2 = MScount[x]/TILE_SIZE + 1;
      uint imsblock2 = 0;
      for(int mj2 = 0; mj2 < nmsblock2; mj2++){
	uint msid2 = imsblock2*TILE_SIZE + tgx;
	uint j = (msid2 < MScount[x]) ? MSptr[x*MStile_size + msid2] : 0;
	
	localData[localAtomIndex].posq = posq[j];
        LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
    
        SYNC_WARPS;

	unsigned int tj = tgx;
	for (j = 0; j < TILE_SIZE; j++) {
      
	  int localAtom2Index = tbx+tj;
	  real4 posq2 = localData[localAtom2Index].posq;
	  real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	  real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	  LOAD_ATOM2_PARAMETERS
#ifdef USE_CUTOFF
	  int atom2 = atomIndices[localAtom2Index];//?
	  int tree_pointer2 = localData[localAtom2Index].tree_pointer;//?
#else
	  uint msid2 = imsblock2*TILE_SIZE + tj;
	  int atom2 = (msid2 < MScount[x]) ? MSptr[x*MStile_size + msid2] : 0;
#endif
	  bool compute = msid1 < MScount[y] && msid2 < MScount[x];
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
	    int parent_slot = (ordered) ? tree_pointer1 : tree_pointer2;
	    int parent_children_start = ovChildrenStartIndex[parent_slot];
	    int child_atom = (ordered) ? atom2 : atom1 ;
	    if(!ordered) delta = -delta; //parent and child are reversed (atom2>atom1)
#else
	    int child_atom = atom2;	
#endif
	    COMPUTE_INTERACTION_STORE1
	      }
	  tj = (tj + 1) & (TILE_SIZE - 1);
	  SYNC_WARPS;
	}
	
	imsblock2++;
	SYNC_WARPS;
      }//mj2
	
      imsblock1++;
      SYNC_WARPS;
    }//mj1
	
    SYNC_WARPS;
    pos++;
  }
  
}



/**
 * Reduce the self volumes of the MS particles
 */
/*
__kernel void MSreduceSelfVolumes_buffer(int bufferSize, int numBuffers, 
				       __global const int*   restrict ovAtomTreePointer,
				       __global const real4* restrict ovAtomBuffer,
				       //self volumes
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global       long*  restrict selfVolumeBuffer_long,
#endif
				       __global       real*  restrict selfVolumeBuffer,
				       __global       real*  restrict selfVolume
){
  uint id = get_global_id(0);
  int totalSize = bufferSize*numBuffers;
#ifdef SUPPORTS_64_BIT_ATOMICS
  real scale = 1/(real) 0x100000000;
#endif
  
  //accumulate self volumes
  uint atom = id;
  while (atom < bufferSize) {  
#ifdef SUPPORTS_64_BIT_ATOMICS
   // copy self volumes from long energy buffer to regular one
    selfVolume[atom] = scale*selfVolumeBuffer_long[atom];
#else
    real sum = 0;
    for (int i = atom; i < totalSize; i += bufferSize) sum += selfVolumeBuffer[i];
    selfVolume[atom] = sum;
#endif
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
}
*/

/**
 * Reduce the self volumes of the MS particles
 */
__kernel void MSaddSelfVolumes(
					 int const                 ntiles, 
					 int const                 MStile_size,
					 __global const int*    restrict MScount,
					 __global const int*    restrict MSptr,
					 __global const int*    restrict MSpParent1,
					 __global const int*    restrict MSpParent2,
					 __global const real*   restrict MSpSelfVolume,
					 //atomic self volumes buffers
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global       long*  restrict selfVolumeBuffer_long,
#endif
				       __global       real*  restrict selfVolumeBuffer,
				        __global       int*  restrict Semaphor,
				       __global       real*  restrict selfVolume){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  const unsigned int localAtomIndex = get_local_id(0);

  int pos = (int) (warp*(long)ntiles/totalWarps);
  int end = (int) ((warp+1)*(long)ntiles/totalWarps);
  while(pos < end && pos < ntiles){
    int ims = tgx;
    while(ims < MScount[pos] ){
      int msatom = MSptr[pos*MStile_size + ims];
      int parent1 = MSpParent1[msatom];
      int parent2 = MSpParent2[msatom];
      real selfv = 0.5f*MSpSelfVolume[msatom];
#ifdef SUPPORTS_64_BIT_ATOMICS
      atom_add(&selfVolumeBuffer_long[parent1], (long)(selfv*0x100000000));
      atom_add(&selfVolumeBuffer_long[parent2], (long)(selfv*0x100000000));
#else
      //in most cases the semaphor will deadlock for a synchronized warp of size greater than 1
      GetSemaphor(&(Semaphor[parent1]));
      selfVolume[parent1] += selfv;
      ReleaseSemaphor(&(Semaphor[parent1]));
      GetSemaphor(&(Semaphor[parent2]));
      selfVolume[parent2] += selfv;
      ReleaseSemaphor(&(Semaphor[parent2]));
#endif
      ims += TILE_SIZE;
    }

    pos++;
    SYNC_WARPS;
  }
  
}


__kernel void MSaddSelfVolumesFromLong(
					 int const                 padded_num_atoms, 
				       __global       long*  restrict selfVolumeBuffer_long,
				       __global       real*  restrict selfVolume){
#ifdef SUPPORTS_64_BIT_ATOMICS
  real scale = 1/(real) 0x100000000;
  uint atom = get_global_id(0);
  while (atom < padded_num_atoms) {  
    selfVolume[atom] = scale*selfVolumeBuffer_long[atom];
    atom += get_global_size(0);
  }
#endif
}
