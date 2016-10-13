#ifdef SUPPORTS_64_BIT_ATOMICS
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

#define PI (3.14159265359f)


/**
 * Reduce the atom self volumes
 */
__kernel void reduceSelfVolumes_buffer(int bufferSize, int numBuffers, 
				       __global const int*   restrict ovAtomTreePointer,
				       __global const real*  restrict AtomicGamma,
				       __global       real4* restrict ovAtomBuffer,
#ifdef SUPPORTS_64_BIT_ATOMICS
				       __global long*   restrict forceBuffers,
				       __global long*   restrict energyBuffer_long,
#else
				       __global real4* restrict forceBuffers,
#endif
				       __global real*  restrict energyBuffer
){
  uint id = get_global_id(0);

  int totalSize = bufferSize*numBuffers;
  real scale = 1/(real) 0x100000000;

 uint atom = id;
 while (atom < NUM_ATOMS) {  
#ifdef SUPPORTS_64_BIT_ATOMICS
   // do nothing with forces, they are stored in computeSelfVolumes kernel
   //
   //atom_add(&forceBuffers[atom], (long) (-sum.x*0x100000000));
   //atom_add(&forceBuffers[atom+PADDED_NUM_ATOMS], (long) (-sum.y*0x100000000));
   //atom_add(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (long) (-sum.z*0x100000000));
   //
   // copy from long energy buffer to regular one
   energyBuffer[id] += scale*energyBuffer_long[atom];
#else
   real4 sum = 0;
   for (int i = atom; i < totalSize; i += bufferSize) sum += ovAtomBuffer[i];
   energyBuffer[id] += sum.w;
   forceBuffers[atom].xyz -= sum.xyz;
#endif
   atom += get_global_size(0);
 }
 barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);    
}
