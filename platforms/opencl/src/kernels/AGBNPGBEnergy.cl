#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif



//initializes accumulators 
__kernel void initGBEnergy(unsigned const int             bufferSize,
			   unsigned const int             numBuffers,
#ifdef SUPPORTS_64_BIT_ATOMICS
			    __global        long* restrict GBEnergyBuffer_long,
#endif
			   __global        real* restrict GBEnergyBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
			   __global        long* restrict GBDerYBuffer_long,
#endif
			   __global        real* restrict GBDerYBuffer
			   ){
  
#ifdef SUPPORTS_64_BIT_ATOMICS
  //reset "long" GB energy buffer
  uint atom = get_global_id(0);
  while(atom < PADDED_NUM_ATOMS){
    GBEnergyBuffer_long[atom] = 0;
    atom += get_global_size(0);
  }
  atom = get_global_id(0);
  while(atom < PADDED_NUM_ATOMS){
    GBDerYBuffer_long[atom] = 0;
    atom += get_global_size(0);
  }  
#else
  // resets traditional thread group accumulation buffer
  uint id = get_global_id(0);
  while(id < bufferSize*numBuffers){
    GBEnergyBuffer[id] = 0;
    id += get_global_size(0);
  }
  id = get_global_id(0);
  while(id < bufferSize*numBuffers){
    GBDerYBuffer[id] = 0;
    id += get_global_size(0);
  }  
#endif
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


typedef struct {
  real4 posq;
  real bornr;
  real4 force;
  real gbdery;
} AtomData;

#define PI (3.14159265359f)

__kernel void GBPairEnergy(
			       __global const real4* restrict posq, //atomic positions
			       __global const  real* restrict chargeParam, //atomic radius
			       __global const  real* restrict BornRadius,

			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, __global const int* restrict interactingAtoms,
			       unsigned int maxTiles,
			       __global const ushort2* exclusionTiles,
#else
			       unsigned int numTiles,
#endif

#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global long* restrict GBEnergyBuffer_long,
#endif
			       __global real* restrict GBEnergyBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global long* restrict GBDerYBuffer_long,
#endif
			       __global real* restrict GBDerYBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
			     __global long*   restrict forceBuffers
#else
			     __global real4* restrict forceBuffers
#endif
){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local AtomData localData[FORCE_WORK_GROUP_SIZE];
  
  real dielectric_factor = AGBNP_DIELECTRIC_FACTOR;
  real pt25 = 0.25;


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
    real charge1 = chargeParam[atom1];
    real bornr1 = BornRadius[atom1]; 
    real egb1 = 0;
    real4 force1 = 0;
    real gbdery1 = 0;
    
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].posq.w = chargeParam[j];
    localData[localAtomIndex].bornr = BornRadius[j];
    localData[localAtomIndex].force = 0;
    localData[localAtomIndex].gbdery = 0;
    
    SYNC_WARPS;

    if(y==x){//diagonal tile, need to check for atom1<atom2

      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real charge2 = posq2.w;
	real bornr2 = localData[localAtom2Index].bornr;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real4 force2 = 0;
	real gbdery2 = 0;
	
	int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2 && r2 < CUTOFF_SQUARED && atom1 != atom2) {
	  
	  real qqf = charge1*charge2;
	  real qq = dielectric_factor*qqf;
	  real bb = bornr1*bornr2;
	  real etij = exp(-pt25*r2/bb);
	  real fgb = rsqrt(r2 + bb*etij);
	  egb1 += 2.*qq*fgb;
	  real fgb3 = fgb*fgb*fgb;
	  real mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	  real4 g = delta * mw;
	  force1 += g;
	  force2 -= g;
	  real ytij = qqf*(bb+pt25*r2)*etij*fgb3;
	  gbdery1 += ytij;
	  gbdery2 += ytij;
	}
	
	localData[localAtom2Index].force += force2;
	localData[localAtom2Index].gbdery += gbdery2;
	
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

    }else{//off-diagonal tile: atom1 and atom2 do not have to be in any particular order

      unsigned int tj = tgx;
      for (j = 0; j < TILE_SIZE; j++) {
	
	int localAtom2Index = tbx+tj;
	real4 posq2 = localData[localAtom2Index].posq;
	real charge2 = posq2.w;
	real bornr2 = localData[localAtom2Index].bornr;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real4 force2 = 0;
	real gbdery2 = 0;
	
	int atom2 = x*TILE_SIZE+tj;
	
	if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && r2 < CUTOFF_SQUARED) {
	  
	  real qqf = charge1*charge2;
	  real qq = dielectric_factor*qqf;
	  real bb = bornr1*bornr2;
	  real etij = exp(-pt25*r2/bb);
	  real fgb = rsqrt(r2 + bb*etij);
	  egb1 += 2.*qq*fgb;
	  real fgb3 = fgb*fgb*fgb;
	  real mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	  real4 g = delta * mw;
	  force1 += g;
	  force2 -= g;
	  real ytij = qqf*(bb+pt25*r2)*etij*fgb3;
	  gbdery1 += ytij;
	  gbdery2 += ytij;
	}
	
	localData[localAtom2Index].force += force2;
	localData[localAtom2Index].gbdery += gbdery2;
	
	tj = (tj + 1) & (TILE_SIZE - 1);
	SYNC_WARPS;
      }

    }
    SYNC_WARPS;

    //update force, energy, etc. buffers for this tile
    unsigned int atom2 = x*TILE_SIZE + tgx;
#ifdef SUPPORTS_64_BIT_ATOMICS
    atom_add(&GBEnergyBuffer_long[atom1], (long) (egb1*0x100000000));
    atom_add(&GBDerYBuffer_long[atom1], (long) (gbdery1*0x100000000));
    atom_add(&forceBuffers[atom1], (long) (force1.x*0x100000000));
    atom_add(&forceBuffers[atom1+PADDED_NUM_ATOMS], (long) (force1.y*0x100000000));
    atom_add(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (long) (force1.z*0x100000000));
    if(atom2 < PADDED_NUM_ATOMS) {
       atom_add(&GBDerYBuffer_long[atom2], (long) (localData[get_local_id(0)].gbdery*0x100000000));
       atom_add(&forceBuffers[atom2], (long) (localData[get_local_id(0)].force.x*0x100000000));
       atom_add(&forceBuffers[atom2+PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.y*0x100000000));
       atom_add(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.z*0x100000000));
    }
#else
    unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
    unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;
    GBEnergyBuffer[offset1] += egb1;
    GBDerYBuffer[offset1] += gbdery1;
    forceBuffers[offset1] += force1;
    if(atom2 < PADDED_NUM_ATOMS) {
      GBDerYBuffer[offset2] += localData[get_local_id(0)].gbdery;
      forceBuffers[offset2] += localData[get_local_id(0)].force;
    }
#endif
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
    real charge1 = chargeParam[atom1];
    real bornr1 = BornRadius[atom1]; 
    real egb1 = 0;
    real4 force1 = 0;
    real gbdery1 = 0;
    
#ifdef USE_CUTOFF
    uint j = interactingAtoms[pos*TILE_SIZE + tgx];
    atomIndices[get_local_id(0)] = j;
    if(j<PADDED_NUM_ATOMS){
      localData[localAtomIndex].posq = posq[j];
      localData[localAtomIndex].posq.w = chargeParam[j];
      localData[localAtomIndex].bornr = BornRadius[j];
    }
#else
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].posq.w = chargeParam[j];
    localData[localAtomIndex].bornr = BornRadius[j];
#endif
    localData[localAtomIndex].force = 0;
    localData[localAtomIndex].gbdery = 0;
    SYNC_WARPS;
    
    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real charge2 = posq2.w;
      real bornr2 = localData[localAtom2Index].bornr;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real4 force2 = 0;
      real gbdery2 = 0;

#ifdef USE_CUTOFF
      int atom2 = atomIndices[localAtom2Index];
#else
      int atom2 = x*TILE_SIZE+tj;
#endif
      bool compute = atom1 < NUM_ATOMS && atom2 < NUM_ATOMS;
#ifdef USE_CUTOFF
      compute = compute && r2 < CUTOFF_SQUARED;
#else
      //when not using a neighbor list we are getting diagonal tiles here
      if(x == y) compute = compute && atom1 < atom2;
#endif
      if (compute) {

	real qqf = charge1*charge2;
	real qq = dielectric_factor*qqf;
	real bb = bornr1*bornr2;
	real etij = exp(-pt25*r2/bb);
	real fgb = rsqrt(r2 + bb*etij);
	egb1 += 2.*qq*fgb;
	real fgb3 = fgb*fgb*fgb;
	real mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	real4 g = delta * mw;
	force1 += g;
	force2 -= g;
	real ytij = qqf*(bb+pt25*r2)*etij*fgb3;
	gbdery1 += ytij;
	gbdery2 += ytij;
      }

      localData[localAtom2Index].force += force2;
      localData[localAtom2Index].gbdery += gbdery2;
      
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }
    SYNC_WARPS;
    
    //update force, energy, etc. buffers for this tile
#ifdef USE_CUTOFF
   uint atom2 = atomIndices[get_local_id(0)];
#else
   uint atom2 = x*TILE_SIZE + tgx;
#endif
#ifdef SUPPORTS_64_BIT_ATOMICS
    atom_add(&GBEnergyBuffer_long[atom1], (long) (egb1*0x100000000));
    atom_add(&GBDerYBuffer_long[atom1], (long) (gbdery1*0x100000000));
    atom_add(&forceBuffers[atom1], (long) (force1.x*0x100000000));
    atom_add(&forceBuffers[atom1+PADDED_NUM_ATOMS], (long) (force1.y*0x100000000));
    atom_add(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (long) (force1.z*0x100000000));
    if(atom2 < PADDED_NUM_ATOMS) {
       atom_add(&GBDerYBuffer_long[atom2], (long) (localData[get_local_id(0)].gbdery*0x100000000));
       atom_add(&forceBuffers[atom2], (long) (localData[get_local_id(0)].force.x*0x100000000));
       atom_add(&forceBuffers[atom2+PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.y*0x100000000));
       atom_add(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.z*0x100000000));
    }
#else
    unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
    unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;
    GBEnergyBuffer[offset1] += egb1;
    GBDerYBuffer[offset1] += gbdery1;
    forceBuffers[offset1] += force1;
    if(atom2 < PADDED_NUM_ATOMS) {
      GBDerYBuffer[offset2] += localData[get_local_id(0)].gbdery;
      forceBuffers[offset2] += localData[get_local_id(0)].force;
    }
#endif

    SYNC_WARPS;
    
    pos++; //new tile	
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

#ifndef SUPPORTS_64_BIT_ATOMICS
// version of GBPairEnergy optimized for CPU devices
//  executed by 1 CPU core, instead of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and process them 
__kernel __attribute__((reqd_work_group_size(1,1,1)))
void GBPairEnergy_cpu(
			       __global const real4* restrict posq, //atomic positions
			       __global const  real* restrict chargeParam, //atomic radius
			       __global const  real* restrict BornRadius,

			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
			       unsigned int maxTiles,
#else
			       unsigned int numTiles,
#endif


			       __global real* restrict GBEnergyBuffer,
			       __global real* restrict GBDerYBuffer,
			       __global real4* restrict forceBuffers
){

  uint id = get_global_id(0);
  uint ncores = get_global_size(0);
  __local AtomData localData[TILE_SIZE];

  real dielectric_factor = AGBNP_DIELECTRIC_FACTOR;
  real pt25 = 0.25;

  uint warp = id;
  uint totalWarps = ncores; 
#ifdef USE_CUTOFF
  unsigned int numTiles = interactionCount[0];
  int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
  int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
#else
  int pos = (int) (warp*(long)numTiles/totalWarps);
  int end = (int) ((warp+1)*(long)numTiles/totalWarps);
#endif
  
  while (pos < end) {
    // Extract the coordinates of this tile.
    int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
    int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
      y += (x < y ? -1 : 1);
      x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
    }

    // Load the data for this tile in local memory
    for (int j = 0; j < TILE_SIZE; j++) {
      unsigned int atom2 = x*TILE_SIZE + j;
      localData[j].posq = posq[atom2];
      localData[j].posq.w = chargeParam[atom2];
      localData[j].bornr = BornRadius[atom2];
      localData[j].force = 0;
      localData[j].gbdery = 0;
    }


    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // Load atom1 data for this tile.
      real4 posq1 = posq[atom1];
      real charge1 = chargeParam[atom1];
      real bornr1 = BornRadius[atom1];
      real egb1 = 0;
      real4 force1 = 0;
      real gbdery1 = 0;
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	uint atom2 = x*TILE_SIZE+j;
	
	//load atom2 parameters
	real4 posq2 = localData[j].posq;
	real charge2 = posq2.w;
	real bornr2 = localData[j].bornr;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real4 force2 = 0;
	real gbdery2 = 0;
	
	if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {

	  real qqf = charge1*charge2;
	  real qq = dielectric_factor*qqf;
	  real bb = bornr1*bornr2;
	  real etij = exp(-pt25*r2/bb);
	  real fgb = rsqrt(r2 + bb*etij);
	  egb1 += 2.*qq*fgb;
	  real fgb3 = fgb*fgb*fgb;
	  real mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	  real4 g = delta * mw;
	  force1 += g;
	  force2 -= g;
	  real ytij = qqf*(bb+pt25*r2)*etij*fgb3;
	  gbdery1 += ytij;
	  gbdery2 += ytij;

	}

	localData[j].force += force2;
	localData[j].gbdery += gbdery2;
	
      }

      //update buffers for atoms in set 1
      if(atom1 < NUM_ATOMS){
	unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
      	GBEnergyBuffer[offset1] += egb1;
      	GBDerYBuffer[offset1] += gbdery1;
	real4 f = forceBuffers[offset1];
	f.x += force1.x;
	f.y += force1.y;
	f.z += force1.z;
	forceBuffers[offset1] = f;
      }
      
    }

    //update buffers for atoms in set 2
    for (unsigned int j = 0; j < TILE_SIZE; j++) {
      uint atom2 = x*TILE_SIZE+j;
      if(atom2 < NUM_ATOMS){
    	unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;	    
    	GBDerYBuffer[offset2] += localData[j].gbdery;
	real4 f = forceBuffers[offset2];
	f.x += localData[j].force.x;
	f.y += localData[j].force.y;
	f.z += localData[j].force.z;
	forceBuffers[offset2] = f;
      }
    }

    pos++; //new tile
  }
}
#endif /* SUPPORTS_64_BIT_ATOMICS */


__kernel void reduceGBEnergy(unsigned const int bufferSize, unsigned const int numBuffers, 
#ifdef SUPPORTS_64_BIT_ATOMICS
			      __global const long* restrict GBEnergyBuffer_long,
#endif
			      __global        real* restrict GBEnergyBuffer,

			     __global const  real* restrict chargeParam,
			     __global const  real* restrict BornRadius,
			     __global const  real* restrict invBornRadius_fp,

#ifdef SUPPORTS_64_BIT_ATOMICS
			      __global const long* restrict GBDerYBuffer_long,
#endif
			      __global const real* restrict GBDerYBuffer,

			      __global       real* restrict GBDerY,
			      __global       real* restrict GBDerBrU,
			     
                              __global mixed*  restrict energyBuffer
){
  uint id = get_global_id(0);

  int totalSize = bufferSize*numBuffers;
  real scale = 1/(real) 0x100000000;

  real dielectric_factor = AGBNP_DIELECTRIC_FACTOR;

  //
  //GB energy
  //
  uint atom = id;
#ifdef SUPPORTS_64_BIT_ATOMICS
  while (atom < PADDED_NUM_ATOMS) {  
    GBEnergyBuffer[atom] = scale*GBEnergyBuffer_long[atom];
    atom += get_global_size(0);
  }
#else
  while (atom < NUM_ATOMS) {
    real sum = 0;
    for (int i = atom; i < totalSize; i += bufferSize) sum += GBEnergyBuffer[i];
    GBEnergyBuffer[atom] = sum;
    atom += get_global_size(0);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
  //add self energy
  atom = id;
  while (atom < NUM_ATOMS) {
    real qq = dielectric_factor*chargeParam[atom]*chargeParam[atom];
    GBEnergyBuffer[atom] += qq/BornRadius[atom];
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  //copy to OpenMM's energy buffer
  atom = id;
  while (atom < NUM_ATOMS) {
    energyBuffer[atom] += GBEnergyBuffer[atom];
    atom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  //
  //GB Y and BrU variables (for 2nd component of GB forces)
  //
  atom = id;
#ifdef SUPPORTS_64_BIT_ATOMICS
  while (atom < NUM_ATOMS) {  
    GBDerY[atom] = scale*GBDerYBuffer_long[atom];
    atom += get_global_size(0);
  }
#else
  while (atom < NUM_ATOMS) {
    real sum = 0;
    for (int i = atom; i < totalSize; i += bufferSize) sum += GBDerYBuffer[i];
    GBDerY[atom] = sum;
    atom += get_global_size(0);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  real fac = -dielectric_factor/(4.*PI);
  atom = id;
  while (atom < NUM_ATOMS) {  
    GBDerBrU[atom] = fac*(chargeParam[atom]*chargeParam[atom]+GBDerY[atom]*BornRadius[atom])*invBornRadius_fp[atom];
    atom += get_global_size(0);
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
}
