#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)

/**
 * This kernel walks the tree bottom-up to compute atomic self volumes
 * It assumes only one work group
 */
void computeSelfVolumes_ofatom(
   unsigned          const int            padded_tree_size,
   unsigned          const int            offset,
   
   __global          const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
   
   __global          const int*  restrict ovLevel,
   __global          const real* restrict ovVolume,
   __global          const real* restrict ovVSfp,
   __global          const real* restrict ovGamma1i,
   __global          const real4* restrict ovG,
   __global volatile       real* restrict ovSelfVolume,
   
   __global volatile       real4* restrict ovDV1,
   __global volatile       real4* restrict ovDV2,

   __global          const int*  restrict ovLastAtom,
   __global          const int*  restrict ovRootIndex,
   __global          const int*  restrict ovChildrenStartIndex,
   __global          const int*  restrict ovChildrenCount,
   __global volatile       int*  restrict ovProcessedFlag,
   __global volatile       int*  restrict ovOKtoProcessFlag,
   __global volatile       int*  restrict ovChildrenReported){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;      //index of this warp

  const unsigned int totalWarps_inBlock = get_local_size(0)/TILE_SIZE;
  const unsigned int warp_inBlock = get_local_id(0)/TILE_SIZE;      //index of this warp in thread block

  const unsigned int id = get_global_id(0) & (TILE_SIZE-1);  //the index of this thread in the warp
  const unsigned int idb = get_global_id(0) - id;            //the global index of the beginning of the warp

  int nblock = TILE_SIZE; //size of warp
  int niterations = padded_tree_size/nblock;
  __local volatile int nprocessed;

  unsigned int slot = offset + padded_tree_size - id - 1; // start at end of tree

  for(int i=0;i<niterations;i++,slot -= nblock){
    
    do{
      if(id==0) nprocessed = 0;
      barrier(CLK_LOCAL_MEM_FENCE);

      if(ovProcessedFlag[slot] == 0){
	
	if(ovOKtoProcessFlag[slot] > 0){

	  int level = ovLevel[slot];
	  real cf = level % 2 == 0 ? -1.0 : 1.0;
	  real volcoeff  = level > 0 ? cf : 0;
	  real volcoeffp = level > 0 ? volcoeff/(float)level : 0;

	  //"own" volume contribution (volcoeff[level=0] for top root is automatically zero)
	  real self_volume = volcoeffp*ovVolume[slot];
	  //real energy = volcoeffp*ovGamma1i[slot]*ovVolume[slot];

	  //gather self volumes and derivatives from children
	  //dv.w is the gradient of the energy
	  real sfp = ovVSfp[slot];
	  real4 dv1 = (real4)(0,0,0,volcoeffp*sfp*ovGamma1i[slot]);
	  int start = ovChildrenStartIndex[slot];
	  int count = ovChildrenCount[slot];
	  if(start > 0){// start=-1 if no children
	    for(int j=start; j < start+count ; j++){
	      self_volume += ovSelfVolume[j];
	      dv1   += ovDV1[j];	      
	      // /(float)ovProcessedFlag[j];// to catch children not processed: result would be Inf/NaN
	    }
	  }

	  //stores new self_volume
	  ovSelfVolume[slot] = self_volume;

	  //
	  // Recursive rules for derivatives:
	  //
	  real an = global_gaussian_exponent[ovLastAtom[slot]];
	  real a1i = ovG[slot].w;
	  real a1 = a1i - an;
	  real dvvc = dv1.w;
	  ovDV2[slot].xyz = -ovDV1[slot].xyz * dvvc  + (an/a1i)*dv1.xyz; //this gets accumulated later
	  ovDV1[slot].xyz =  ovDV1[slot].xyz * dvvc  + (a1/a1i)*dv1.xyz;
	  ovDV1[slot].w   =  ovDV1[slot].w   * dvvc;

	  //mark parent ok to process counter
	  int parent_index = ovRootIndex[slot];
	  if(parent_index >= 0){
	    atomic_inc(&(ovChildrenReported[parent_index]));
	    if(ovChildrenReported[parent_index] == ovChildrenCount[parent_index]){
	      atomic_inc(&(ovOKtoProcessFlag[parent_index]));
	    }
	  }
	  
	  ovProcessedFlag[slot] = 1; //mark as processed
	  ovOKtoProcessFlag[slot] = 0; //prevent more processing
	}
        atomic_inc(&nprocessed);
      }

      //if(ovChildrenReported[slot] == ovChildrenCount[slot]) ovOKtoProcessFlag[slot] = 1;

      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    } while(nprocessed > 0);
  }
}


__kernel void computeSelfVolumes(
  __global const int* restrict ovAtomTreePointer,
  __global const int* restrict ovAtomTreeSize,
  __global const int* restrict ovAtomTreePaddedSize,
  
  __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
  
  
  __global const int*   restrict ovLevel,
  __global const real*  restrict ovVolume,
  __global const real*  restrict ovVSfp,
  __global const real*  restrict ovGamma1i,
  __global const real4* restrict ovG,
  __global       real*  restrict ovSelfVolume,

  __global       real4* restrict ovDV1,
  __global       real4* restrict ovDV2,
  
  __global const int*  restrict ovLastAtom,
  __global const int*  restrict ovRootIndex,
  __global const int*  restrict ovChildrenStartIndex,
  __global const int*  restrict ovChildrenCount,
  __global       int*  restrict ovProcessedFlag,
  __global       int*  restrict ovOKtoProcessFlag,
  __global       int*  restrict ovChildrenReported){
  const unsigned int totalWarps = get_global_size(0)/TILE_SIZE;
  const unsigned int warp = get_global_id(0)/TILE_SIZE;      //index of this warp

  unsigned int atom = warp; // initial assignment of warp to atoms
  while(atom < PADDED_NUM_ATOMS){
    unsigned int offset = ovAtomTreePointer[atom];
    unsigned int tree_size = ovAtomTreeSize[atom];
    unsigned int padded_tree_size = ovAtomTreePaddedSize[atom];

     //then collect volumes
    computeSelfVolumes_ofatom(padded_tree_size, offset,

			      global_gaussian_exponent,
			      
			      ovLevel,
			      ovVolume,
			      ovVSfp,
			      ovGamma1i,
			      ovG,
			      ovSelfVolume,

			      ovDV1,
			      ovDV2,

			      ovLastAtom,
			      ovRootIndex,
			      ovChildrenStartIndex,
			      ovChildrenCount,
			      ovProcessedFlag,
			      ovOKtoProcessFlag,
			      ovChildrenReported);


    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    atom += totalWarps; //next atom  
  }
}
