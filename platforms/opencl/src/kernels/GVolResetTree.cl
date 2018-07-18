#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)

/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */
void resetTreeCounters(
  unsigned          const int            padded_tree_size,
  unsigned          const int            tree_size,
  unsigned          const int            offset,
  __global                int*   restrict ovProcessedFlag,
  __global                int*   restrict ovOKtoProcessFlag,
  __global          const int*   restrict ovChildrenStartIndex,
  __global          const int*   restrict ovChildrenCount,
  __global                int*   restrict ovChildrenReported
){
  const unsigned int id = get_local_id(0);  //the index of this thread in the workgroup
  const unsigned int nblock = get_local_size(0); //size of work group
  unsigned int begin = offset + id;
  unsigned int size = offset + tree_size;
  unsigned int end  = offset + padded_tree_size;

  for(int slot=begin; slot<end ; slot+=nblock){
    ovProcessedFlag[slot] = (slot >= size) ? 1 : 0; //mark slots with overlaps as not processed
  }
  for(int slot=begin; slot<end ; slot+=nblock){
    ovOKtoProcessFlag[slot] = (slot >= size) ? 0 : 
      ( ovChildrenCount[slot] == 0 ? 1 : 0); //marks leaf nodes (no children) as ok to process
  }
  for(int slot=begin; slot<end ; slot+=nblock){
    ovChildrenReported[slot] = 0;
  }
}


//assume num. groups = num. tree sections
__kernel void resetSelfVolumes(const int ntrees,
			       __global const int*   restrict ovTreePointer,
			       __global const int*   restrict ovAtomTreePointer,
			       __global const int*   restrict ovAtomTreeSize,
			       __global const int*   restrict ovAtomTreePaddedSize,
			       __global const int*   restrict ovChildrenStartIndex,
			       __global const int*   restrict ovChildrenCount,
			       __global       int*   restrict ovProcessedFlag,
			       __global       int*   restrict ovOKtoProcessFlag,
			       __global       int*   restrict ovChildrenReported,
			       __global       int*   restrict PanicButton
){
    uint tree = get_group_id(0);      //initial tree
    if(PanicButton[0] > 0) return;
    while (tree < ntrees){

      uint offset = ovTreePointer[tree];
      uint tree_size = ovAtomTreeSize[tree];
      uint padded_tree_size = ovAtomTreePaddedSize[tree];
      resetTreeCounters(padded_tree_size, tree_size, offset, 
			ovProcessedFlag,
			ovOKtoProcessFlag,
			ovChildrenStartIndex,
			ovChildrenCount,
			ovChildrenReported);
      tree += get_num_groups(0);
    }
}


/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */
void resetTreeSection(
		      unsigned const int padded_tree_size,
		      unsigned const int offset,
		      __global       int*   restrict ovLevel,
		      __global       real*  restrict ovVolume,
		      __global       real*  restrict ovVsp,
		      __global       real*  restrict ovVSfp,
		      __global       real*  restrict ovSelfVolume,
		      __global       double* restrict ovVolEnergy,
		      __global       int*   restrict ovLastAtom,
		      __global       int*   restrict ovRootIndex,
		      __global       int*   restrict ovChildrenStartIndex,
		      __global       int*   restrict ovChildrenCount,
		      __global       real4* restrict ovDV1,
		      __global       real4* restrict ovDV2,
		      __global       int*   restrict ovProcessedFlag,
		      __global       int*   restrict ovOKtoProcessFlag,
		      __global       int*   restrict ovChildrenReported){
  const unsigned int nblock = get_local_size(0); //size of thread block
  const unsigned int id = get_local_id(0);  //the index of this thread in the warp

  unsigned int begin = offset + id;
  unsigned int end  = offset + padded_tree_size;

  for(int slot=begin; slot<end ; slot+=nblock) ovLevel[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovVsp[slot] = 1;
  for(int slot=begin; slot<end ; slot+=nblock) ovVSfp[slot] = 1;
  for(int slot=begin; slot<end ; slot+=nblock) ovSelfVolume[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovVolEnergy[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovLastAtom[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovRootIndex[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenStartIndex[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenCount[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovDV1[slot] = (real4)0;
  for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = (real4)0;
  for(int slot=begin; slot<end ; slot+=nblock) ovProcessedFlag[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovOKtoProcessFlag[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenReported[slot] = 0;
}


__kernel void resetBuffer(unsigned const int             bufferSize,
			  unsigned const int             numBuffers,
			  __global       real4* restrict ovAtomBuffer,
			  __global        real* restrict selfVolumeBuffer
#ifdef SUPPORTS_64_BIT_ATOMICS
			  ,
			  __global long*   restrict selfVolumeBuffer_long
#endif
){
  unsigned int id = get_global_id(0);
#ifdef SUPPORTS_64_BIT_ATOMICS
  while (id < bufferSize){
    selfVolumeBuffer_long[id] = 0;
    id += get_global_size(0);
  }
#else
  while(id < bufferSize*numBuffers){
    ovAtomBuffer[id] = (real4)0;
    selfVolumeBuffer[id] = 0;
    id += get_global_size(0);
  }
#endif
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


__kernel void resetTree(const int ntrees,
			__global const int*   restrict ovTreePointer,
			__global const int*   restrict ovAtomTreePointer,
			__global       int*   restrict ovAtomTreeSize,
			__global const int*   restrict ovAtomTreePaddedSize,
			__global       int*   restrict ovLevel,
			__global       real*  restrict ovVolume,
			__global       real*  restrict ovVsp,
			__global       real*  restrict ovVSfp,
			__global       real*  restrict ovSelfVolume,
			__global       double*  restrict ovVolEnergy,
			__global       int*   restrict ovLastAtom,
			__global       int*   restrict ovRootIndex,
			__global       int*   restrict ovChildrenStartIndex,
			__global       int*   restrict ovChildrenCount,
			__global       real4* restrict ovDV1,
			__global       real4* restrict ovDV2,

			__global       int*  restrict ovProcessedFlag,
			__global       int*  restrict ovOKtoProcessFlag,
			__global       int*  restrict ovChildrenReported,
			__global       int*  restrict ovAtomTreeLock
			){


  unsigned int section = get_group_id(0); // initial assignment of warp to tree section
  while(section < ntrees){
    unsigned int offset = ovTreePointer[section];
    unsigned int padded_tree_size = ovAtomTreePaddedSize[section];

    //each block resets one section of the tree
    resetTreeSection(padded_tree_size, offset, 
		     ovLevel,
		     ovVolume,
		     ovVsp,
		     ovVSfp,
		     ovSelfVolume,
		     ovVolEnergy,
		     ovLastAtom,
		     ovRootIndex,
		     ovChildrenStartIndex,
		     ovChildrenCount,
		     ovDV1,
		     ovDV2,
		     ovProcessedFlag,
		     ovOKtoProcessFlag,
		     ovChildrenReported
		     );
    ovAtomTreeLock[section] = 0;
    section += get_num_groups(0); //next section  
  }
}
