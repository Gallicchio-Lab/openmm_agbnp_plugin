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
  __global                int*   restrict ovChildrenReported,
  __global                real*  restrict ovSelfVolume,
  __global                real4* restrict ovDV2 ){
  const unsigned int id = get_local_id(0);  //the index of this thread in the workgroup
  const unsigned int nblock = get_local_size(0); //size of work group
  unsigned int begin = offset + id;
  unsigned int size = offset + tree_size;
  unsigned int end  = offset + padded_tree_size;

  //Looks like these are not needed since computeSelfVolumes_oftree() resets them?
  //for(int slot=begin; slot<end ; slot+=nblock) ovSelfVolume[slot] = 0;
  //for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = (real4)0;

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
__kernel void resetSelfVolumes(__global const int*  restrict ovAtomTreePointer,
			       __global const int*  restrict ovAtomTreeSize,
			       __global const int*  restrict ovAtomTreePaddedSize,
			       __global const int*  restrict ovChildrenStartIndex,
			       __global const int*   restrict ovChildrenCount,
			       __global       int*  restrict ovProcessedFlag,
			       __global       int*  restrict ovOKtoProcessFlag,
			       __global       int*  restrict ovChildrenReported,
			       __global       real*  restrict ovSelfVolume,
			       __global       real4* restrict ovDV2
){
    uint tree = get_group_id(0);      //index of this group
    uint offset = ovAtomTreePointer[tree*ATOMS_PER_SECTION];
    uint tree_size = ovAtomTreeSize[tree];
    uint padded_tree_size = ovAtomTreePaddedSize[tree];
    resetTreeCounters(padded_tree_size, tree_size, offset, 
		      ovProcessedFlag,
		      ovOKtoProcessFlag,
		      ovChildrenStartIndex,
		      ovChildrenCount,
		      ovChildrenReported,
		      ovSelfVolume,
		      ovDV2);
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
		      __global       real*  restrict ovVSfp,
		      __global       real*  restrict ovSelfVolume,
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
  for(int slot=begin; slot<end ; slot+=nblock) ovVSfp[slot] = 1;
  for(int slot=begin; slot<end ; slot+=nblock) ovSelfVolume[slot] = 0;
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
			  __global       int*   restrict ovAtomLock
#ifdef SUPPORTS_64_BIT_ATOMICS
			  ,
			  __global long*   restrict energyBuffer_long
#endif
){
  unsigned int id = get_global_id(0);
  if(id < PADDED_NUM_ATOMS) ovAtomLock[id] = 0;
#ifdef SUPPORTS_64_BIT_ATOMICS
  if(id < PADDED_NUM_ATOMS) energyBuffer_long[id] = 0;
#endif
  while(id < bufferSize*numBuffers){
    ovAtomBuffer[id] = (real4)0;
    id += get_global_size(0);
  }
}

__kernel void resetOvCount(unsigned const int             bufferSize,
			   unsigned const int             numBuffers,
			   __global       int*   restrict ovCount){
  unsigned int id = get_global_id(0);
  while(id < bufferSize*numBuffers){
    ovCount[id] = 0;
    id += get_global_size(0);
  }
}


__kernel void resetTree(__global const int*   restrict ovAtomTreePointer,
			__global       int*   restrict ovAtomTreeSize,
			__global const int*   restrict ovAtomTreePaddedSize,
			__global       int*   restrict ovLevel,
			__global       real*  restrict ovVolume,
			__global       real*  restrict ovVSfp,
			__global       real*  restrict ovSelfVolume,
			__global       int*   restrict ovLastAtom,
			__global       int*   restrict ovRootIndex,
			__global       int*   restrict ovChildrenStartIndex,
			__global       int*   restrict ovChildrenCount,
			__global       real4* restrict ovDV1,
			__global       real4* restrict ovDV2,

			__global       int*  restrict ovProcessedFlag,
			__global       int*  restrict ovOKtoProcessFlag,
			__global       int*  restrict ovChildrenReported,
			__global       int*  restrict ovAtomTreeLock,

			__global const int*   restrict ishydrogenParam //1=hydrogen atom
			){


  const unsigned int totalWarps = get_global_size(0)/get_local_size(0);
  const unsigned int warp = get_global_id(0)/get_local_size(0);      //index of this warp

  unsigned int section = warp; // initial assignment of warp to tree section

  while(section < NUM_BLOCKS){
    int atom = section*ATOMS_PER_SECTION; //pointer to start of section
    unsigned int offset = ovAtomTreePointer[atom];
    unsigned int padded_tree_size = ovAtomTreePaddedSize[section];

    //each block resets one section of the tree
    resetTreeSection(padded_tree_size, offset, 
		     ovLevel,
		     ovVolume,
		     ovVSfp,
		     ovSelfVolume,
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
							 //ovAtomTreeSize[atom] = (atom < NUM_ATOMS && !ishydrogenParam[atom]) ? 1 : 0;
							 //ovLevel[offset] = 1;
							 //ovLastAtom[offset] = atom;
    ovAtomTreeLock[section] = 0;
    section += totalWarps; //next section  
  }
}
