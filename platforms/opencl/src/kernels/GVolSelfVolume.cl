#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)

__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void computeSelfVolumes(
  __global const int* restrict ovAtomTreePointer,
  __global const int* restrict ovAtomTreeSize,
  __global       int* restrict NIterations,
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
  __global       int*  restrict ovChildrenReported,
  __global     real4*  restrict ovAtomBuffer
#ifdef SUPPORTS_64_BIT_ATOMICS
  ,
  __global      long*   restrict forceBuffers,
  __global      long*   restrict ovEnergyBuffer_long
#endif
){
  const uint id = get_local_id(0);
  const uint tree = get_group_id(0);      //index of this group
  const uint offset = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
  uint tree_size = ovAtomTreeSize[tree];
  const uint padded_tree_size = ovAtomTreePaddedSize[tree];
  const uint gsize = get_local_size(0);
  const uint nsections = padded_tree_size/gsize;
  uint ov_count = 0;
  const uint buffer_offset = tree*PADDED_NUM_ATOMS; // offset into buffer arrays
  __local volatile uint nprocessed;
  __local volatile uint niterations;

  // The tree for this atom is divided into sections each the size of a workgroup
  // The tree is walked bottom up one section at a time
  for(int isection=nsections-1;isection >= 0; isection--){
    uint slot = offset + isection*gsize + id; //the slot to work on

    //reset accumulators
    ovSelfVolume[slot] = 0;
    ovDV2[slot] = 0;
    int atom = ovLastAtom[slot];
    int level = ovLevel[slot];
    if(id == 0) niterations = 0; 
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    //
    // process section
    //
    do{
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if(id == 0) nprocessed = 0;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      int processed = ovProcessedFlag[slot];
      int ok2process = ovOKtoProcessFlag[slot];
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
      if(processed == 0 && ok2process == 0 && atom >= 0){
	if(ovChildrenReported[slot] == ovChildrenCount[slot]){
	  ok2process = 1;
	  ovOKtoProcessFlag[slot] = 1;
	}
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      if(processed == 0 && ok2process > 0 && atom >= 0) {
	atomic_inc(&(nprocessed));

	real cf = level % 2 == 0 ? -1.0 : 1.0;
	real volcoeff  = level > 0 ? cf : 0;
	real volcoeffp = level > 0 ? volcoeff/(float)level : 0;
	
	//"own" volume contribution (volcoeff[level=0] for top root is automatically zero)
	real self_volume = volcoeffp*ovVolume[slot]*ovGamma1i[slot]; //really the energy, set gamma=1 for self volume
	//real energy = volcoeffp*ovGamma1i[slot]*ovVolume[slot];
	
	//gather self volumes and derivatives from children
	//dv.w is the gradient of the energy
	real4 dv1 = (real4)(0,0,0,volcoeffp*ovVSfp[slot]*ovGamma1i[slot]);
	int start = ovChildrenStartIndex[slot];
	int count = ovChildrenCount[slot];
	if(count > 0 && start >= 0){
	  for(int j=start; j < start+count ; j++){
	    if(ovLastAtom[j] >= 0){
	      self_volume += ovSelfVolume[j];
	      dv1 += ovDV1[j];	     
	    } 
	  }
	}
	
	//stores new self_volume
	ovSelfVolume[slot] = self_volume;
	
	//
	// Recursive rules for derivatives:
	//
	real an = global_gaussian_exponent[atom];
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
	}
	ovProcessedFlag[slot] = 1; //mark as processed
	ovOKtoProcessFlag[slot] = 0; //prevent more processing
      }
      if(id==0) niterations += 1;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }while( nprocessed > 0 && niterations < gsize); // loop until no more work is done
    //
    // End of processing for section
    //
    if(id==0){
      if(niterations > NIterations[tree]) NIterations[tree] = niterations;
    }


    // Updates energy and derivative buffer for this section
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#ifdef SUPPORTS_64_BIT_ATOMICS
    if(atom >= 0){
      real4 dv2 = -ovDV2[slot];
      atom_add(&forceBuffers[atom], (long) (dv2.x*0x100000000));
      atom_add(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (dv2.y*0x100000000));
      atom_add(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (dv2.z*0x100000000));
      if(level == 1){ // energy is stored at the 1-body level
	atom_add(&ovEnergyBuffer_long[atom], (long) (ovSelfVolume[slot]*0x100000000));
      }
    }
#else
    // 
    //without atomics can't do it in parallel due to "atom" collisions
    //
    if(id==0){
      uint tree_offset =  offset + isection*gsize;
      for(uint is = tree_offset ; is < tree_offset + gsize ; is++){ //loop over slots in section
	int at = ovLastAtom[is];
	if(at >= 0){
	  real energy = (ovLevel[is] == 1) ? ovSelfVolume[is] : 0; //energy is stored on 1-body nodes 
	  ovAtomBuffer[buffer_offset + at] += (real4)(ovDV2[is].xyz, energy);
	}
      }
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
