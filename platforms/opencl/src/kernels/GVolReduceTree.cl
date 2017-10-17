#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)


/**
 * Reduce the atom self volumes
 * energyBuffer could be linked to the selfVolumeBuffer, depending on the use
 */
__kernel void reduceSelfVolumes_buffer(int bufferSize, int numBuffers, 
				       __global const int*   restrict ovAtomTreePointer,
				       __global const real4* restrict ovAtomBuffer,
				       __global const  real* restrict ovVolEnergy,

				       //self volumes
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global       long*  restrict selfVolumeBuffer_long,
#endif
				       __global       real*  restrict selfVolumeBuffer,
				       __global       real*  restrict selfVolume,

				       //energy and forces
				       int update_energy, 
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global long*   restrict forceBuffers,
#else
				       __global real4* restrict forceBuffers,
#endif
				       __global mixed*  restrict energyBuffer
){
  uint id = get_global_id(0);
  int totalSize = bufferSize*numBuffers;
#ifdef SUPPORTS_64_BIT_ATOMICS
  real scale = 1/(real) 0x100000000;
#endif
  
  //accumulate self volumes
  uint atom = id;
  while (atom < NUM_ATOMS) {  
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

 if(update_energy > 0){
   uint atom = id;
   while (atom < NUM_ATOMS_TREE) {
     // volume energy is stored at the 1-body level
     uint slot = ovAtomTreePointer[atom];
     energyBuffer[id] += ovVolEnergy[slot];

#ifdef SUPPORTS_64_BIT_ATOMICS
     // do nothing with forces, they are stored in computeSelfVolumes kernel
#else
     real4 sum = 0;
     for (int i = atom; i < totalSize; i += bufferSize) sum += ovAtomBuffer[i];
     forceBuffers[atom].xyz -= sum.xyz;
#endif
     atom += get_global_size(0);
   }
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
 }

}
