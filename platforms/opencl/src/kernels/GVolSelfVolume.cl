#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)

//computes volume energy and self-volumes
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void computeSelfVolumes(const int ntrees,
  __global const int* restrict ovTreePointer,
  __global const int* restrict ovAtomTreePointer,
  __global const int* restrict ovAtomTreeSize,
  __global       int* restrict NIterations,
  __global const int* restrict ovAtomTreePaddedSize,
  
  __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent

  const int padded_num_atoms,

  __global const int*   restrict ovLevel,
  __global const real*  restrict ovVolume,
  __global const real*  restrict ovVsp,
  __global const real*  restrict ovVSfp,
  __global const real*  restrict ovGamma1i,
  __global const real4* restrict ovG,
  __global       real*  restrict ovSelfVolume,
  __global       double*  restrict ovVolEnergy,

  __global const real4* restrict ovDV1,
  __global       real4* restrict ovDV2,
  __global       real4* restrict ovPF,
  
  __global const int*  restrict ovLastAtom,
  __global const int*  restrict ovRootIndex,
  __global const int*  restrict ovChildrenStartIndex,
  __global const int*  restrict ovChildrenCount,
  __global       int*  restrict ovProcessedFlag,
  __global       int*  restrict ovOKtoProcessFlag,
  __global       int*  restrict ovChildrenReported,
  __global     real4*  restrict ovAtomBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
  __global      long*   restrict forceBuffers,
  __global      long*   restrict selfVolumeBuffer_long,
#endif
   __global      real*   restrict selfVolumeBuffer
){
  const uint id = get_local_id(0);
  const uint gsize = get_local_size(0);
  __local volatile uint nprocessed;
  __local volatile uint niterations;

  uint tree = get_group_id(0);      //index of initial tree
  while(tree < ntrees){
    uint offset = ovTreePointer[tree]; //offset into tree
    uint buffer_offset = tree*padded_num_atoms; // offset into buffer arrays

    uint tree_size = ovAtomTreeSize[tree];
    uint padded_tree_size = ovAtomTreePaddedSize[tree];    
    uint nsections = padded_tree_size/gsize;    
    uint ov_count = 0;

    // The tree for this atom is divided into sections each the size of a workgroup
    // The tree is walked bottom up one section at a time
    for(int isection=nsections-1;isection >= 0; isection--){
      uint slot = offset + isection*gsize + id; //the slot to work on

      //reset accumulators
      ovVolEnergy[slot] = 0;
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
	  real self_volume = volcoeffp*ovVsp[slot]*ovVolume[slot];
	  double energy = ovGamma1i[slot]*self_volume;
	  
	  //gather self volumes and derivatives from children
	  //dv.w is the gradient of the energy
	  real4 dv1 = (real4)(0,0,0,volcoeffp*ovVSfp[slot]*ovGamma1i[slot]);
	  int start = ovChildrenStartIndex[slot];
	  int count = ovChildrenCount[slot];
	  if(count > 0 && start >= 0){
	    for(int j=start; j < start+count ; j++){
	      if(ovLastAtom[j] >= 0){// && ovLastAtom[j] < NUM_ATOMS_TREE){
		energy += ovVolEnergy[j];
		self_volume += ovSelfVolume[j];
		dv1 += ovPF[j];	     
	      } 
	    }
	  }
	  
	  //stores new self_volume
	  ovSelfVolume[slot] = self_volume;

	  //stores energy
	  ovVolEnergy[slot] = energy;
	  
	  //
	  // Recursive rules for derivatives:
	  //
	  real an = global_gaussian_exponent[atom];
	  real a1i = ovG[slot].w;
	  real a1 = a1i - an;
	  real dvvc = dv1.w;
	  ovDV2[slot].xyz = -ovDV1[slot].xyz * dvvc  + (an/a1i)*dv1.xyz; //this gets accumulated later
	  //ovDV2[slot].w = dvvc *  ovDV1[slot].w;//for derivative wrt volumei
	  ovPF[slot].xyz =  ovDV1[slot].xyz * dvvc  + (a1/a1i)*dv1.xyz;
	  ovPF[slot].w   =  ovDV1[slot].w   * dvvc;
	  
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
      if(atom >= 0){// && atom < NUM_ATOMS_TREE){
	real4 dv2 = -ovDV2[slot];
	atom_add(&forceBuffers[atom], (long) (dv2.x*0x100000000));
	atom_add(&forceBuffers[atom+padded_num_atoms], (long) (dv2.y*0x100000000));
	atom_add(&forceBuffers[atom+2*padded_num_atoms], (long) (dv2.z*0x100000000));
	atom_add(&selfVolumeBuffer_long[atom], (long) (ovSelfVolume[slot]*0x100000000));
	// nothing to do here for the volume energy,
	// it is automatically stored in ovVolEnergy at the 1-body level
      }
#else
      // 
      //without atomics can not accumulate in parallel due to "atom" collisions
      //
      if(id==0){
	uint tree_offset =  offset + isection*gsize;
	for(uint is = tree_offset ; is < tree_offset + gsize ; is++){ //loop over slots in section
	  int at = ovLastAtom[is];
	  if(at >= 0){// && at < NUM_ATOMS_TREE){
	    // nothing to do here for the volume energy,
	    // it is automatically stored in ovVolEnergy at the 1-body level
	    ovAtomBuffer[buffer_offset + at] += (real4)(ovDV2[is].xyz, 0); //.w element was used to store the energy
	    selfVolumeBuffer[buffer_offset + at] += ovSelfVolume[is];
	  }
	}
      }
#endif
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    // moves to next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}


//same as self-volume kernel above but does not update self volumes
__kernel __attribute__((reqd_work_group_size(OV_WORK_GROUP_SIZE,1,1)))
void computeVolumeEnergy(const int ntrees,
  __global const int* restrict ovTreePointer,
  __global const int* restrict ovAtomTreePointer,
  __global const int* restrict ovAtomTreeSize,
  __global       int* restrict NIterations,
  __global const int* restrict ovAtomTreePaddedSize,
  
  __global const real* restrict global_gaussian_exponent, //atomic Gaussian exponent
  
  
  __global const int*   restrict ovLevel,
  __global const real*  restrict ovVolume,
  __global const real*  restrict ovVsp,
  __global const real*  restrict ovVSfp,
  __global const real*  restrict ovGamma1i,
  __global const real4* restrict ovG,
  __global       double*  restrict ovVolEnergy,

  __global const real4* restrict ovDV1,
  __global       real4* restrict ovDV2,
  __global       real4* restrict ovPF,
			 
  __global const int*  restrict ovLastAtom,
  __global const int*  restrict ovRootIndex,
  __global const int*  restrict ovChildrenStartIndex,
  __global const int*  restrict ovChildrenCount,
  __global       int*  restrict ovProcessedFlag,
  __global       int*  restrict ovOKtoProcessFlag,
  __global       int*  restrict ovChildrenReported,
  __global     real4*  restrict ovAtomBuffer
#ifdef SUPPORTS_64_BIT_ATOMICS
   , __global      long*   restrict forceBuffers
#endif
){
  const uint id = get_local_id(0);
  const uint gsize = get_local_size(0);
  __local volatile uint nprocessed;
  __local volatile uint niterations;

  uint tree = get_group_id(0);      //index of initial tree
  while(tree < ntrees){
    uint offset = ovTreePointer[tree]; //offset into tree
    uint buffer_offset = tree*PADDED_NUM_ATOMS; // offset into buffer arrays

    uint tree_size = ovAtomTreeSize[tree];
    uint padded_tree_size = ovAtomTreePaddedSize[tree];    
    uint nsections = padded_tree_size/gsize;    
    uint ov_count = 0;

    // The tree for this atom is divided into sections each the size of a workgroup
    // The tree is walked bottom up one section at a time
    for(int isection=nsections-1;isection >= 0; isection--){
      uint slot = offset + isection*gsize + id; //the slot to work on

      //reset accumulators
      ovVolEnergy[slot] = 0;
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
	  double energy = volcoeffp*ovGamma1i[slot]*ovVsp[slot]*ovVolume[slot];
	  
	  //gather self volumes and derivatives from children
	  //dv.w is the gradient of the energy
	  real4 dv1 = (real4)(0,0,0,volcoeffp*ovVSfp[slot]*ovGamma1i[slot]);
	  int start = ovChildrenStartIndex[slot];
	  int count = ovChildrenCount[slot];
	  if(count > 0 && start >= 0){
	    for(int j=start; j < start+count ; j++){
	      if(ovLastAtom[j] >= 0 && ovLastAtom[j] < NUM_ATOMS_TREE){
		energy += ovVolEnergy[j];
		dv1 += ovPF[j];	     
	      } 
	    }
	  }
	  
	  //stores energy
	  ovVolEnergy[slot] = energy;
	  
	  //
	  // Recursive rules for derivatives:
	  //
	  real an = global_gaussian_exponent[atom];
	  real a1i = ovG[slot].w;
	  real a1 = a1i - an;
	  real dvvc = dv1.w;
	  ovDV2[slot].xyz = -ovDV1[slot].xyz * dvvc  + (an/a1i)*dv1.xyz; //this gets accumulated later
	  ovPF[slot].xyz =  ovDV1[slot].xyz * dvvc  + (a1/a1i)*dv1.xyz;
	  ovPF[slot].w   =  ovDV1[slot].w   * dvvc;
	  
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
      if(atom >= 0 && atom < NUM_ATOMS_TREE){
	real4 dv2 = -ovDV2[slot];
	atom_add(&forceBuffers[atom], (long) (dv2.x*0x100000000));
	atom_add(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (dv2.y*0x100000000));
	atom_add(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (dv2.z*0x100000000));
	// nothing to do here for the volume energy,
	// it is automatically stored in ovVolEnergy at the 1-body level
      }
#else
      // 
      //without atomics can not accumulate in parallel due to "atom" collisions
      //
      if(id==0){
	uint tree_offset =  offset + isection*gsize;
	for(uint is = tree_offset ; is < tree_offset + gsize ; is++){ //loop over slots in section
	  int at = ovLastAtom[is];
	  if(at >= 0 && atom < NUM_ATOMS_TREE){
	    // nothing to do here for the volume energy,
	    // it is automatically stored in ovVolEnergy at the 1-body level
	    ovAtomBuffer[buffer_offset + at] += (real4)(ovDV2[is].xyz, 0); //.w element was used to store the energy
	  }
	}
      }
#endif
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    // moves to next tree
    tree += get_num_groups(0);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
}
