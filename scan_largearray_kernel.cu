#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif 
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
__device__ void prescanArray(float *outArray, float *inArray, int numElements)
{

  extern __shared__ float temp[];

  int thid = threadIdx.x;

  int offset = 1;

  /*
    block A
  */
  // temp[2*thid] = g_idata[2*thid]; // lodd input into shared array
  // temp[2*thid+1] = g_idata[2*thid+1];

  int ai = thid; 
  int bi = thid + (numElements/2);

  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(ai);

  temp[ai + bankOffsetA] = inArray[ai];
  temp[bi + bankOffsetB] = inArray[bi];

  for (int d = numElements>>1; d > 0; d >>= 1){
    __syncthreads();
    if (thid < d) {

      /*
        block B
      */
      // int ai = offset*(2*thid+1) - 1;
      // int bi = offset*(2*thid+2) -1;

      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  /*
    block C
  */
  if (thid == 0) {
    temp[numElements-1 + CONFLICT_FREE_OFFSET(numElements - 1)] = 0;
  }

  for (int d = 1; d<numElements; d*=2){
    offset >>= 1;
    __syncthreads();

    if (thid < d) {

      /* 
        block d
      */
      // int ai = offset*(2*thid+1) - 1;
      // int bi = offset*(2*thid+2) -1;

      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();

  /* 
    block e
  */

  // g_odata[2*thid] = temp[2*thid];
  // g_odata[2*thid+1] = temp[2*thid+1];
  outArray[ai] = temp[ai + bankOffsetA];
  outArray[bi] = temp[bi + bankOffsetB];


}
// **===-----------------------------------------------------------===**

__global__ void hostPrescanArray(float *outArray, float *inArray, int numElements){
  prescanArray(outArray, inArray, numElements);
}
#endif // _PRESCAN_CU_
