#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)


void reduceSelfVolumes(int tree_size,
		       int buffersize,
		       int offset_atom_buffer,
		       int offset_tree,
		       __global       real4* restrict ovAtomBuffer,
		       __global const real* restrict ovSelfVolume,
		       __global const real4* restrict ovDV2,
		       __global const int*  restrict ovLastAtom){
  unsigned int id = get_local_id(0);
  // assume tree size is a multiple of FORCE_WORK_GROUP_SIZE
  unsigned int num_tree_sections = tree_size/FORCE_WORK_GROUP_SIZE;
  unsigned int na = PADDED_NUM_ATOMS/FORCE_WORK_GROUP_SIZE;
  unsigned int num_atom_sections = PADDED_NUM_ATOMS % FORCE_WORK_GROUP_SIZE == 0 ? na  : na + 1;
  __local real4 localData[FORCE_WORK_GROUP_SIZE]; // xyz is derivative, w is self volume 
  __local int   localLastAtom[FORCE_WORK_GROUP_SIZE];

  unsigned int atom = id; //initial atom assignment
  //  for(unsigned int atomsect=0 ; atomsect < num_atom_sections; atomsect++){
  while(atom < PADDED_NUM_ATOMS){
    real4 sum = 0;

    for(unsigned int treesect = 0; treesect < num_tree_sections ; treesect++){

      unsigned int offset = offset_tree + treesect*FORCE_WORK_GROUP_SIZE;
      //load tree section into local space
      localData[id].xyz = ovDV2[offset + id].xyz;
      localData[id].w = ovSelfVolume[offset + id]; 
      localLastAtom[id] = ovLastAtom[offset + id];
      barrier(CLK_LOCAL_MEM_FENCE);

      //scan the tree with staggered access
      unsigned int index = id;
      for (unsigned int i = 0 ; i < FORCE_WORK_GROUP_SIZE; i++, index++){
	unsigned int slot = index % FORCE_WORK_GROUP_SIZE;
	//add if assigned atom matches the last atom of the overlap
	sum += localData[slot] * ((localLastAtom[slot] == atom) ? 1 : 0 );
      }

    }

    ovAtomBuffer[offset_atom_buffer + atom] = sum;
    atom += get_local_size(0);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}



/**
 * Reduce the atom self volumes. It scans the tree looking for slots corresponding
 * to the assigned atoms. It is parallelized by having multiple
 * thread groups scan a different section of the tree and update one section
 * of the AtomBuffer, which then get subsequently reduced.
 */
__kernel void reduceSelfVolumes_tree(int buffersize,
				     int nbuffers,
				     int total_tree_size,
				     __global       real4* restrict ovAtomBuffer,
				     __global const real* restrict ovSelfVolume,
				     __global const real4* restrict ovDV2,
				     __global const int*  restrict ovLastAtom){
  //assumes buffersize is = padded number of atoms
  unsigned int gid = get_global_id(0);
  unsigned int ngroups = get_global_size(0)/get_local_size(0);
  unsigned int group_id = gid/get_local_size(0);
 //here we assume that tree_size is a multiple of ngroups
  unsigned int tree_section_size = total_tree_size/ngroups;

  unsigned int offset_atom_buffer = group_id*buffersize;     //beginning of buffer in ovAtomBuffer
  unsigned int offset_tree = group_id*tree_section_size; //beg. of section in tree



  reduceSelfVolumes(tree_section_size, buffersize,
  		    offset_atom_buffer, offset_tree,
  		    ovAtomBuffer,
  		    ovSelfVolume,
  		    ovDV2,
  		    ovLastAtom);

}


/**
 * Reduce the atom self volumes
 */
__kernel void reduceSelfVolumes_buffer(int bufferSize, int numBuffers, 
				       __global const int*   restrict ovAtomTreePointer,
				       __global const real*  restrict ovGamma1i,
				       __global       real4* restrict ovAtomBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global long*   restrict forceBuffers,
#else
				       __global float4* restrict forceBuffers,
#endif
				       __global float*  restrict energyBuffer){
  unsigned int atom = get_global_id(0);
  int totalSize = bufferSize*numBuffers;

  while (atom < PADDED_NUM_ATOMS) {
    real4 sum = 0;
    for (int i = atom; i < totalSize; i += bufferSize) sum += ovAtomBuffer[i];
    //add to energyBuffer, first section
    // energy is sum_i gamma_i self_volume_i
    int slot = ovAtomTreePointer[atom];
    ovAtomBuffer[atom] = sum.w;
    energyBuffer[atom] += ovGamma1i[slot]*sum.w;

    //add to gradients
#ifdef SUPPORTS_64_BIT_ATOMICS
    atom_add(&forceBuffers[atom], (long) (-sum.x*0x100000000));
    atom_add(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (-sum.y*0x100000000));
    atom_add(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (-sum.z*0x100000000));
#else
    forceBuffers[atom].xyz -= sum.xyz;
#endif
    atom += get_global_size(0);
  }
}
