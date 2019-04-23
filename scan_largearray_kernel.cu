#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif 
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512

// Lab4: Host Helper Functions (allocate your own data structure...)
bool isPowerTwo(int n) {
  if(n==0) 
    return false; 
 
  return (ceil(log2((double) n)) == floor(log2((double) n))); 
}

int padCalculator(int n) {
  return (512 - (n % 512));
}

// Lab4: Device Functions

__global__ void uniformAdd(float *outArray, float *incArray) {
  // for (int ii = 0; ii<numElements; ii++) {
  //   printf("ii: %d\n", outArray[ii]);
  //  // outArray[ii] = outArray[ii] + incArray[(ii%BLOCK_SIZE)];
  // }

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // printf("outArray[0]: %f\n", outArray[0]);
  // printf("incArray[0]: %d\n", incArray[0]);
  // printf("incArray[1]: %d\n", incArray[1]);

  outArray[idx] = outArray[idx] + incArray[(idx/BLOCK_SIZE)];

  // printf("incArray[0]: %d\n", incArray[idx/BLOCK_SIZE]);
}

// Lab4: Kernel Functions

__global__ void prescanArray(float *outArray, float *inArray, float *sumsArray, int numElements, int padding, int numBlocks, bool sums)
{

  __shared__ float temp[512+CONFLICT_FREE_OFFSET(512)];

  int thid = threadIdx.x + blockDim.x*blockIdx.x;
  int threadId = threadIdx.x;
  int blockId = blockIdx.x;

  int offset = 1;

  if (threadId == 0) {
    for (int i = 0; i < 512+CONFLICT_FREE_OFFSET(512); i++) {
      temp[i] = 0;
    }
  }
  __syncthreads();

  /*
    block A
  */
  // temp[2*thid] = g_idata[2*thid]; // lodd input into shared array
  // temp[2*thid+1] = g_idata[2*thid+1];

  int ai = thid; 
  int bi = thid + (numElements/2);

  int ai2 = threadId;
  int bi2 = threadId + (numElements/2);

  int bankOffsetA = CONFLICT_FREE_OFFSET(ai2);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi2);


  temp[ai2 + bankOffsetA] = inArray[ai];
  temp[bi2 + bankOffsetB] = inArray[bi];

  for (int d = numElements>>1; d > 0; d >>= 1){
    __syncthreads();
    if (threadId < d) {

      /*
        block B
      */
      // int ai = offset*(2*thid+1) - 1;
      // int bi = offset*(2*thid+2) -1;

      int ai = (offset*(2*threadId+1)-1);
      int bi = (offset*(2*threadId+2)-1);
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  /*
    block C
  */
  if (threadId == 0) {
    // 2. writing to sums array
    // int last_element = (BLOCK_SIZE - 1) + (blockId * BLOCK_SIZE);
    int last_element = 511;
    if (!sums){
      sumsArray[blockId] = temp[last_element + CONFLICT_FREE_OFFSET(last_element)];
      printf("sumsArray[%d]: %f\n", blockId, sumsArray[blockId]);
    }
    temp[last_element + CONFLICT_FREE_OFFSET(last_element)] = 0;
  }
  __syncthreads();
  for (int d = 1; d<numElements; d*=2){
    offset >>= 1;
    __syncthreads();

    if (threadId < d) {

      /* 
        block d
      */
      // int ai = offset*(2*thid+1) - 1;
      // int bi = offset*(2*thid+2) -1;

      int ai = (offset*(2*threadId+1)-1);
      int bi = (offset*(2*threadId+2)-1);
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();
  

  // /* 
  //   block e
  // */

  // g_odata[2*thid] = temp[2*thid];
  // g_odata[2*thid+1] = temp[2*thid+1];

  // int ai21 = (offset*(2*thid+1)-1);
  // ai21 += CONFLICT_FREE_OFFSET(ai21);
  // int bi21 = (offset*(2*thid+2)-1);
  // bi21 += CONFLICT_FREE_OFFSET(bi21);

  outArray[ai] = temp[ai2 + bankOffsetA];
  outArray[bi] = temp[bi2 + bankOffsetB];



}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

// **===-----------------------------------------------------------===**

void hostPrescanArray(float *outArray, float *inArray, float *sumsArray, float *incArray, float *dumArray, int numElements){
 // prescanArray(outArray, inArray, numElements);
  /*
    1. divide array into blocks to be scanned by a single thread block
    2. write total sum into another array of block sums
    3. scan block sums, generates array of block increments
    4. go through outArray and increment the values as necessary
  */

/*
  1. divide array into blocks
*/

  bool sums = false;
  int numBlocks = 0;
  int numThreads = BLOCK_SIZE/2;
  printf("numElements: %d\n", numElements);
  int padding = padCalculator(numElements);
  if (padding == 512)
    padding = 0;
  printf("padding: %d\n", padding);

  if (numElements < BLOCK_SIZE) {
    numBlocks = 1;
  }
  else if (numElements % BLOCK_SIZE == 0) {
    // done
    numBlocks = numElements/BLOCK_SIZE;
  }
  else {
    numBlocks = (numElements/BLOCK_SIZE) + 1;
  }

  /*
    2. write total sum into sumArray (happens in prescan call)
  */
  //float sumsArray[numBlocks];
  dim3 threadPerBlock(numThreads);
  dim3 blocks(numBlocks);
  prescanArray<<<blocks,threadPerBlock>>>(outArray, inArray, sumsArray, BLOCK_SIZE, padding, numBlocks, sums);
  cudaThreadSynchronize();

  // __global__ void prescanArray(float *outArray, float *inArray, float *sumsArray, int numElements, int padding, int numBlocks, bool sums)
  /* 
    3. scan block sums. generate incArray
  */
  
  int numSums = numBlocks;
  int numSumBlocks = 0;
  numThreads = BLOCK_SIZE/2;
  sums = true;
  int sumsPadding = padCalculator(numSums);

  if (numSums < BLOCK_SIZE) {
    numSumBlocks = 1;
  }
  else if (numSums % BLOCK_SIZE == 0){
    numSumBlocks = numSums / BLOCK_SIZE;
  }
  else {
    numSumBlocks = (numSums / BLOCK_SIZE) + 1;
  }
  
  //float incArray[numBlocks];
  //float dumArray[0];
  dim3 threadPerBlockSum(numThreads);
  dim3 blocksSum(numSumBlocks);
  prescanArray<<<blocksSum,threadPerBlockSum>>>(incArray, sumsArray, dumArray, numSums, sumsPadding, numSumBlocks, sums);

  cudaThreadSynchronize(); 
  /*
    4. goes through out array and increments the values
  */
  uniformAdd<<<numElements/512, 512>>>(outArray, incArray);

}


#endif // _PRESCAN_CU_
