#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif


#define PI (3.14159265359f)


/**
 * returns the offset into the I4 lookup table based on
 * radius of atom i (the one being descreened) and the
 * radius of atom j (the one that descreens atom i) 
 */
inline int i4_table_id(real Ri, real Rj,
		       uint hsize, uint hmask, uint hjump,
		       __global const  int* restrict values){
  uint key = ((Ri-AGBNP_RADIUS_INCREMENT)/Rj)*AGBNP_RADIUS_PRECISION;
  // search key in hash table and return id if found, 
  // or -1 if not found
  uint k = (key & hmask);
  uint ntries = 0;
  while(values[k] >= 0 &&  values[k] != key && ntries < hsize){
    k = ( (k+hjump) & hmask);
    ntries += 1;
  }
  if(values[k] < 0 || ntries >= hsize) return -1;
  return k;
}



__kernel void testHash(uint hsize, uint hmask, uint hjump, 
		  __global const  int* restrict values,
		  uint num_radii,
		  float Ri,
		  __global const float* restrict radii,
		  __global        int* restrict ids){
  uint j =  get_global_id(0);
  while (j < num_radii) {
    ids[j] = i4_table_id(Ri, radii[j], 
			 hsize, hmask, hjump, values);
    j += get_global_size(0);
  }
}

// value and derivative of function from spline lookup table
// Corresponds to SplineFitter::evaluateSplineDerivative in OpenMM's SplineFitter.cpp
// except that it supports only regular x grids specified by n, xmin and xmax
inline real2 lookup_table(real t, 
			  int n, real xmin, real xmax,
			 __global const real* restrict y,
			 __global const real* restrict y2){

  real2 res;
  res.x = 0;
  res.y = 0;
  if(t > xmin && t < xmax ) {
    real dx = (xmax - xmin)/(n-1);
    real invdx = 1.0/dx;
    int ix = (t-xmin)*invdx;
    real xlow = ix*dx;
    real xupp = (ix+1)*dx; 
    real a = (xupp - t)*invdx;
    real b = 1.0 - a;
    
    real ylow = y[ix];
    real yupp = y[ix+1];
    real y2low = y2[ix];
    real y2upp = y2[ix+1];
    
    res.x = a*ylow+b*yupp + ( (a*a*a-a)*y2low + (b*b*b-b)*y2upp ) *dx*dx/6.0;
    res.y = invdx*(yupp-ylow) + ( (1.0-3.0*a*a)*y2low + (3.0*b*b-1.0)*y2upp ) * dx/6.0;
  }

  return res;
}


__kernel void testLookup(int table_size, real xmin, real xmax,
			 __global const real* restrict y,
                         __global const real* restrict y2,
			 int itable, int num_values,
			 //__global const real* restrict x,
			 __global       real* restrict f,
			 __global       real* restrict derf){
  int table_offset = itable*table_size;
  real dx = 0.03;
  int i = get_global_id(0);
  while(i<num_values){
    real t = xmin + i*dx;
    real2 res = lookup_table(t,
    			     table_size, xmin, xmax,
    			     &(y[table_offset]), &(y2[table_offset]));
    f[i] = res.x;
    derf[i] = res.y;
    i += get_global_size(0);
  }
}

/* a switching function for the inverse born radius (beta)
   so that if beta is negative -> beta' = minbeta
   res.x = filtered inverse born radius
   res.y = derivative of filter function
*/ 		 
inline real2 agbnp_swf_invbr(real invBr){
  /* the maximum born radius is max reach of Q4 lookup table */
  const real  a  = 1./AGBNP_I4LOOKUP_MAXA;
  const real  a2 = 1./(AGBNP_I4LOOKUP_MAXA*AGBNP_I4LOOKUP_MAXA);
  real2 res;

  if(invBr<0.0){
    res.x = a;
    res.y = 0.0;
  }else{
    res.x = sqrt(a2 + invBr*invBr);
    res.y  = invBr/res.x;
  }
  return res;
} 


//initializes accumulators 
__kernel void initBornRadii(unsigned const int             bufferSize,
			    unsigned const int             numBuffers,
#ifdef SUPPORTS_64_BIT_ATOMICS
			    __global        long* restrict invBornRadiusBuffer_long,
#endif
			    __global        real* restrict invBornRadiusBuffer,
			    __global const  real* restrict radiusParam, //atomic radius
			    __global const  real* restrict selfVolume,
			    __global        real* restrict volScalingFactor,
			    __global        real* restrict invBornRadius){
  
#ifndef SUPPORTS_64_BIT_ATOMICS
  // resets traditional thread group accumulation buffer
  uint id = get_global_id(0);
  while(id < bufferSize*numBuffers){
    invBornRadiusBuffer[id] = 0;
    id += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif


  //resets atom-level arrays and "long" accumulation buffer
  uint iatom = get_global_id(0);
  while(iatom < PADDED_NUM_ATOMS){
    real rad = radiusParam[iatom];
    real vol = (4./3.)*PI*rad*rad*rad;
    volScalingFactor[iatom] = selfVolume[iatom]/vol;
    invBornRadius[iatom] = 0;
#ifdef SUPPORTS_64_BIT_ATOMICS
    invBornRadiusBuffer_long[iatom] = 0;
#endif
    iatom += get_global_size(0);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
}


typedef struct {
  real4 posq;
  real rad;
  real scaling_factor;
  bool isheavy;
  real invbr; //inverse Born radius accumulator 
} AtomData;

/* Computes inverse Born radii */
__kernel void inverseBornRadii(
			       __global const real4* restrict posq, //atomic positions
			       __global const  real* restrict radiusParam, //atomic radius
			       __global const  real* restrict volScalingFactor,
			       __global const   int* restrict ishydrogenParam,

			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
			       unsigned int maxTiles,
#else
			       unsigned int numTiles,
#endif
			       //dictionary of table indexes
			       uint hsize, uint hmask, uint hjump,
			       __global const  int* restrict hvalues,

			       //spline lookup data
			       int table_size, real rmin, real rmax,
			       __global const real* restrict hy,
			       __global const real* restrict hy2,


#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global long* restrict invBornRadiusBuffer_long,
#endif
			       __global real* restrict invBornRadiusBuffer
){

  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;
  const unsigned int tgx = get_local_id(0) & (TILE_SIZE-1); //warp id in group
  const unsigned int tbx = get_local_id(0) - tgx;           //id in warp
  const unsigned int localAtomIndex = get_local_id(0);
  __local AtomData localData[FORCE_WORK_GROUP_SIZE];
  
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
    
    unsigned int atom1 = y*TILE_SIZE + tgx;
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real rad1 = radiusParam[atom1]; 
    real scaling_factor1 = volScalingFactor[atom1];
    bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;
    real invbr1 = 0;
    
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].rad = radiusParam[j];
    localData[localAtomIndex].scaling_factor = volScalingFactor[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
    localData[localAtomIndex].invbr = 0;
    
    SYNC_WARPS;
    
    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real r = sqrt(r2);
      real rad2 = localData[localAtom2Index].rad;
      real scaling_factor2 = localData[localAtom2Index].scaling_factor;
      bool isheavy2 = localData[localAtom2Index].isheavy;
      real invbr2 = 0;
      
      int atom2 = x*TILE_SIZE+tj;
      
      if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {
	
	if(isheavy2) {
	  // atom2 descreens atom1
	  int table_id = i4_table_id(rad1, rad2, 
	  			     hsize, hmask, hjump, hvalues);
	  int table_offset = table_size * table_id;
	  real2 res = lookup_table(r,
	  			   table_size, rmin, rmax,
	  			   &(hy[table_offset]), &(hy2[table_offset]));
	  invbr1 += scaling_factor2*res.x;
	}
	if(isheavy1){
	  // atom1 descreens atom2
	  int table_id = i4_table_id(rad2, rad1, 
				     hsize, hmask, hjump, hvalues);
	  int table_offset = table_size * table_id;
	  real2 res = lookup_table(r,
	  			   table_size, rmin, rmax,
	  			   &(hy[table_offset]), &(hy2[table_offset]));
	  invbr2 += scaling_factor1*res.x;
	}
      }
      
      localData[localAtom2Index].invbr += invbr2;
      
      
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }
    SYNC_WARPS;
    
    //update inverse Born radii buffers
    unsigned int atom2 = x*TILE_SIZE + tgx;
#ifdef SUPPORTS_64_BIT_ATOMICS
    atom_add(&invBornRadiusBuffer_long[atom1], (long) (invbr1*0x100000000));
    if (atom2 < PADDED_NUM_ATOMS) {
      atom_add(&invBornRadiusBuffer_long[atom2], (long) (localData[get_local_id(0)].invbr*0x100000000));
    }
#else
    unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
    unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;
    invBornRadiusBuffer[offset1] += invbr1;
    if (atom2 < PADDED_NUM_ATOMS)
      invBornRadiusBuffer[offset2] += localData[get_local_id(0)].invbr;
#endif

    SYNC_WARPS;
    
    pos++; //new tile	
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


#ifndef SUPPORTS_64_BIT_ATOMICS
// version of inversBornRadii optimized for CPU devices
//  executed by 1 CPU core, instead of 32 as in the GPU-optimized version, loads a TILE_SIZE of interactions
//  and process them 
__kernel __attribute__((reqd_work_group_size(1,1,1)))
           void inverseBornRadii_cpu(
			       __global const real4* restrict posq, //atomic positions
			       __global const  real* restrict radiusParam, //atomic radius
			       __global const  real* restrict volScalingFactor,
			       __global const   int* restrict ishydrogenParam,

			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
			       unsigned int maxTiles,
#else
			       unsigned int numTiles,
#endif
			       //dictionary of table indexes
			       uint hsize, uint hmask, uint hjump,
			       __global const  int* restrict hvalues,

			       //spline lookup data
			       int table_size, real rmin, real rmax,
			       __global const real* restrict hy,
			       __global const real* restrict hy2,
			       __global real* restrict invBornRadiusBuffer
){
  uint id = get_global_id(0);
  uint ncores = get_global_size(0);
  __local AtomData localData[TILE_SIZE];
  
  //  real pifac = -1./(4.*PI);

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
      localData[j].rad = radiusParam[atom2];
      localData[j].scaling_factor = volScalingFactor[atom2];
      localData[j].isheavy = ishydrogenParam[atom2] > 0 ? 0 : 1;
      localData[j].invbr = 0;
    }

    for (unsigned int tgx = 0; tgx < TILE_SIZE; tgx++) {
      uint atom1 = y*TILE_SIZE+tgx;
      
      // Load atom1 data for this tile.
      real4 posq1 = posq[atom1];
      real rad1 = radiusParam[atom1]; 
      real scaling_factor1 = volScalingFactor[atom1];
      bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;
      real invbr1 = 0;
      
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	uint atom2 = x*TILE_SIZE+j;
	
	//load atom2 parameters
	real4 posq2 = localData[j].posq;
	real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
	real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
	real r = sqrt(r2);
	real rad2 = localData[j].rad;
	real scaling_factor2 = localData[j].scaling_factor;
	bool isheavy2 = localData[j].isheavy;
	real invbr2 = 0;
	
	if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {
	  
	  if(isheavy2) {
	    // atom2 descreens atom1
	    int table_id = i4_table_id(rad1, rad2, 
				       hsize, hmask, hjump, hvalues);
	    if(table_id >= 0){
	      int table_offset = table_size * table_id;
	      real2 res = lookup_table(r,
				       table_size, rmin, rmax,
				       &(hy[table_offset]), &(hy2[table_offset]));
	      invbr1 += pifac*scaling_factor2*res.x;
	    }
	  }
	  if(isheavy1){
	    // atom1 descreens atom2
	    int table_id = i4_table_id(rad2, rad1, 
				       hsize, hmask, hjump, hvalues);
	    if(table_id >= 0){
	      int table_offset = table_size * table_id;
	      real2 res = lookup_table(r,
				       table_size, rmin, rmax,
				       &(hy[table_offset]), &(hy2[table_offset]));
	      invbr2 += pifac*scaling_factor1*res.x;
	    }
	  }
	}
	localData[j].invbr += invbr2;
      }



      //updates buffers
      if(atom1 < NUM_ATOMS){
	unsigned int offset1 = atom1 + id*PADDED_NUM_ATOMS;
	invBornRadiusBuffer[offset1] += invbr1;
      }
      for (unsigned int j = 0; j < TILE_SIZE; j++) {
	uint atom2 = x*TILE_SIZE+j;
	if(atom2 < NUM_ATOMS){
	  unsigned int offset2 = atom2 + id*PADDED_NUM_ATOMS;	    
	  invBornRadiusBuffer[offset2] = localData[j].invbr;
	}
      }
      
    }     
    pos++; //new tile	
  }
}
#endif


__kernel void reduceBornRadii(unsigned const int bufferSize, unsigned const int numBuffers, 
#ifdef SUPPORTS_64_BIT_ATOMICS
			      __global const long* restrict invBornRadiusBuffer_long,
#endif
			      __global const  real* restrict invBornRadiusBuffer,
			      __global const  real* restrict radiusParam, //atomic radius
			      __global        real* restrict invBornRadius,
			      __global        real* restrict invBornRadius_fp,
			      __global        real* restrict BornRadius){
  uint id = get_global_id(0);
  real pifac = -1./(4.*PI);
  int totalSize = bufferSize*numBuffers;
  real scale = 1/(real) 0x100000000;

 uint atom = id;
 while (atom < PADDED_NUM_ATOMS) {  
#ifdef SUPPORTS_64_BIT_ATOMICS
   real b1 = scale*invBornRadiusBuffer_long[atom];
#else
   real b1 = 0;
   for (int i = atom; i < totalSize; i += bufferSize) b1 += invBornRadiusBuffer[i];
#endif
   //scale and add 1/Rvdw
   invBornRadius[atom] = (1./(radiusParam[atom]-AGBNP_RADIUS_INCREMENT)) + pifac*b1;
   real2 res = agbnp_swf_invbr(invBornRadius[atom]);
   BornRadius[atom] = 1./res.x;
   invBornRadius_fp[atom] = res.y;
   atom += get_global_size(0);
 }
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
}


//van der Waals energy
__kernel void VdWEnergy(__global const  real* restrict alphaParam, //VdW alpha parameter
			__global const  real* restrict BornRadius,
			__global const  real* restrict invBornRadius_fp,
			__global        real* restrict VdWDerBrW,
			__global        mixed* restrict energyBuffer,
			__global        real* restrict testBuffer
			){
 uint id = get_global_id(0);
 real pifac = 1./(4.*PI);
 uint atom;
 
 atom = id;
 while (atom < NUM_ATOMS) {
   testBuffer[atom] = 0;
   atom += get_global_size(0);
 }
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
 
 atom = id;
 while (atom < NUM_ATOMS) {
   real rbr = BornRadius[atom]+AGBNP_HB_RADIUS;
   real rbr3 = rbr*rbr*rbr;
   real rbr4 = rbr3*rbr;
   real bb = BornRadius[atom]*BornRadius[atom];
   real evdw = alphaParam[atom]/rbr3;
   VdWDerBrW[atom] = -pifac*3.0*alphaParam[atom]*bb*invBornRadius_fp[atom]/rbr4;
   testBuffer[atom] += evdw;
   energyBuffer[atom] += evdw;
   atom += get_global_size(0);
 }
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

//initializes accumulators for derivatives
__kernel void initVdWGBDerBorn(unsigned const int             bufferSize,
			       unsigned const int             numBuffers,
#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global        long* restrict VdWDerWBuffer_long,
#endif
			       __global        real* restrict VdWDerWBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global        long* restrict GBDerUBuffer_long,
#endif
			       __global        real* restrict GBDerUBuffer		    
){
  
#ifdef SUPPORTS_64_BIT_ATOMICS
  uint iatom = get_global_id(0);
  while(iatom < PADDED_NUM_ATOMS){
    VdWDerWBuffer_long[iatom] = 0;
    GBDerUBuffer_long[iatom] = 0;
    iatom += get_global_size(0);
  }
#else
  // resets traditional thread group accumulation buffer
  uint id = get_global_id(0);
  while(id < bufferSize*numBuffers){
    VdWDerWBuffer[id] = 0;
    GBDerUBuffer[id] = 0;
    id += get_global_size(0);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}



typedef struct {
  real4 posq;
  real rad;
  real scaling_factor;
  bool isheavy;
  real brw;
  real bru;
  real derw;
  real deru;
  real4 force;
} AtomDataWU;

/* Computes U and W intermediate parameters for derivatives calculation */
__kernel void VdWGBDerBorn(
			       __global const real4* restrict posq, //atomic positions
			       __global const  real* restrict radiusParam, //atomic radius
			       __global const  real* restrict volScalingFactor,
			       __global const   int* restrict ishydrogenParam,

			       //neighbor list information
#ifdef USE_CUTOFF
			       __global const int* restrict tiles, __global const unsigned int* restrict interactionCount, 
			       unsigned int maxTiles,
#else
			       unsigned int numTiles,
#endif
			       //dictionary of table indexes
			       uint hsize, uint hmask, uint hjump,
			       __global const  int* restrict hvalues,

			       //spline lookup data
			       int table_size, real rmin, real rmax,
			       __global const real* restrict hy,
			       __global const real* restrict hy2,

			       __global const real* restrict VdWDerBrW,
			       __global const real* restrict GBDerBrU,
#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global        long* restrict VdWDerWBuffer_long,
#endif
			       __global        real* restrict VdWDerWBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
			       __global        long* restrict GBDerUBuffer_long,
#endif
			       __global        real* restrict GBDerUBuffer,		    
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
  __local AtomDataWU localData[FORCE_WORK_GROUP_SIZE];
  
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
    
    unsigned int atom1 = y*TILE_SIZE + tgx;
    
    // Load atom data for this tile.
    real4 posq1 = posq[atom1];
    real rad1 = radiusParam[atom1];
    real scaling_factor1 = volScalingFactor[atom1];
    bool isheavy1 = ishydrogenParam[atom1] > 0 ? 0 : 1;
    real brw1 = VdWDerBrW[atom1];
    real bru1 = GBDerBrU[atom1];
    real derw1 = 0;
    real deru1 = 0;
    real4 force1 = 0;
    
    unsigned int j = x*TILE_SIZE + tgx;
    localData[localAtomIndex].posq = posq[j];
    localData[localAtomIndex].rad = radiusParam[j];
    localData[localAtomIndex].scaling_factor = volScalingFactor[j];
    localData[localAtomIndex].isheavy = ishydrogenParam[j] > 0 ? 0 : 1;
    localData[localAtomIndex].brw = VdWDerBrW[j];
    localData[localAtomIndex].bru = GBDerBrU[j];
    localData[localAtomIndex].derw = 0;
    localData[localAtomIndex].deru = 0;
    localData[localAtomIndex].force = 0;
    
    SYNC_WARPS;
    
    unsigned int tj = tgx;
    for (j = 0; j < TILE_SIZE; j++) {
      
      int localAtom2Index = tbx+tj;
      real4 posq2 = localData[localAtom2Index].posq;
      real4 delta = (real4) (posq2.xyz - posq1.xyz, 0);
      real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
      real r = sqrt(r2);
      real rad2 = localData[localAtom2Index].rad;
      real scaling_factor2 = localData[localAtom2Index].scaling_factor;
      bool isheavy2 = localData[localAtom2Index].isheavy;
      real brw2 = localData[localAtom2Index].brw;
      real bru2 = localData[localAtom2Index].bru;
      real derw2 = 0;
      real deru2 = 0;
      real4 force2 = 0;
      
      int atom2 = x*TILE_SIZE+tj;
      
      if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 < atom2) {
	
	if(isheavy2) {
	  // atom2 descreens atom1
	  int table_id = i4_table_id(rad1, rad2, 
	  			     hsize, hmask, hjump, hvalues);
	  int table_offset = table_size * table_id;
	  real2 res = lookup_table(r,
	  			   table_size, rmin, rmax,
	  			   &(hy[table_offset]), &(hy2[table_offset]));
	  derw2 += brw1*res.x;
	  deru2 += bru1*res.x;
	  //force of GB+VdW energies due to variations of Born radii 
	  real4 w = delta * (bru1+brw1)*scaling_factor2*res.y/r;
	  force1 += w; 
	  force2 -= w;
	}
	if(isheavy1){
	  // atom1 descreens atom2
	  int table_id = i4_table_id(rad2, rad1, 
				     hsize, hmask, hjump, hvalues);
	  int table_offset = table_size * table_id;
	  real2 res = lookup_table(r,
	  			   table_size, rmin, rmax,
	  			   &(hy[table_offset]), &(hy2[table_offset]));
	  derw1 += brw2*res.x;
	  deru1 += bru2*res.x;
	  //force of GB+VdW energies due to variations of Born radii 
	  real4 w = - delta * (bru2+brw2)*scaling_factor1*res.y/r; //-delta because we are inverting i with j
	  force2 += w;
	  force1 -= w;
	}
      }
      
      localData[localAtom2Index].derw += derw2;
      localData[localAtom2Index].deru += deru2;
      localData[localAtom2Index].force += force2;

      
      tj = (tj + 1) & (TILE_SIZE - 1);
      SYNC_WARPS;
    }
    SYNC_WARPS;
    
    //update W and U buffers
    unsigned int atom2 = x*TILE_SIZE + tgx;
#ifdef SUPPORTS_64_BIT_ATOMICS
    atom_add(&VdWDerWBuffer_long[atom1], (long) (derw1*0x100000000));
    atom_add(&GBDerUBuffer_long[atom1], (long) (deru1*0x100000000));
    atom_add(&forceBuffers[atom1], (long) (force1.x*0x100000000));
    atom_add(&forceBuffers[atom1+PADDED_NUM_ATOMS], (long) (force1.y*0x100000000));
    atom_add(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (long) (force1.z*0x100000000));  
    if (atom2 < PADDED_NUM_ATOMS) {
      atom_add(&VdWDerWBuffer_long[atom2], (long) (localData[get_local_id(0)].derw*0x100000000));
      atom_add(&GBDerUBuffer_long[atom2], (long) (localData[get_local_id(0)].deru*0x100000000));
      atom_add(&forceBuffers[atom2], (long) (localData[get_local_id(0)].force.x*0x100000000));
      atom_add(&forceBuffers[atom2+PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.y*0x100000000));
      atom_add(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], (long) (localData[get_local_id(0)].force.z*0x100000000));
    }
#else
    unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
    unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;
    VdWDerWBuffer[offset1] += derw1;
    GBDerUBuffer[offset1] += deru1;
    forceBuffers[offset1] += force1;
    if (atom2 < PADDED_NUM_ATOMS){
      VdWDerWBuffer[offset2] += localData[get_local_id(0)].derw;
      GBDerUBuffer[offset2] += localData[get_local_id(0)].deru;
      forceBuffers[offset2] += localData[get_local_id(0)].force;
    }
#endif

    SYNC_WARPS;
    
    pos++; //new tile	
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


__kernel void reduceVdWGBDerBorn(unsigned const int bufferSize, unsigned const int numBuffers, 
#ifdef SUPPORTS_64_BIT_ATOMICS
				 __global const long* restrict VdWDerWBuffer_long,
#endif
				 __global const  real* restrict VdWDerWBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
				 __global const long* restrict GBDerUBuffer_long,
#endif
				 __global const  real* restrict GBDerUBuffer,
		 
				 __global const  real* restrict radiusParam, //atomic radius
				 __global        real* restrict VdWDerW,
				 __global        real* restrict GBDerU,
				 __global        real* restrict VdWGBDerForceX,
				 __global        real* restrict VdWGBDerForceY,
				 __global        real* restrict VdWGBDerForceZ,
#ifdef SUPPORTS_64_BIT_ATOMICS
			     __global long*   restrict forceBuffers
#else
			     __global real4* restrict forceBuffers
#endif				 
){
  uint id = get_global_id(0);

  int totalSize = bufferSize*numBuffers;
  real scale = 1/(real) 0x100000000;

 uint atom = id;
#ifdef SUPPORTS_64_BIT_ATOMICS
 while (atom < PADDED_NUM_ATOMS) {  
   VdWDerW[atom] = scale*VdWDerWBuffer_long[atom];
   GBDerU[atom]  = scale*GBDerUBuffer_long[atom];
   VdWGBDerForceX[atom] = scale*forceBuffers[atom];
   VdWGBDerForceY[atom] = scale*forceBuffers[atom+PADDED_NUM_ATOMS];
   VdWGBDerForceZ[atom] = scale*forceBuffers[atom+2*PADDED_NUM_ATOMS];
   atom += get_global_size(0);
 }
#else
 while (atom < PADDED_NUM_ATOMS) {  
   real sum = 0;
   for (int i = atom; i < totalSize; i += bufferSize) sum += VdWDerWBuffer[i];
   VdWDerW[atom] += sum;
   atom += get_global_size(0);
 }
 atom = id;
 while (atom < PADDED_NUM_ATOMS) {  
   real sum = 0;
   for (int i = atom; i < totalSize; i += bufferSize) sum += GBDerUBuffer[i];
   GBDerU[atom] += sum;
   atom += get_global_size(0);
 }
#endif
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 atom = id;
 real fac = 4.*PI/3.0;
 while (atom < PADDED_NUM_ATOMS) {
   real r = radiusParam[atom];
   real vol = fac*r*r*r;
   GBDerU[atom]  /= vol;
   VdWDerW[atom] /= vol;   
   //aggregate GB and vdW into one term
   VdWDerW[atom] += GBDerU[atom];   
   atom += get_global_size(0);
 }
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
