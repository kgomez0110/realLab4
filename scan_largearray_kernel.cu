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



// Lab4: Kernel Functions

__global__ void prescanArray(float *outArray, float *inArray, float *sumsArray, int numElements, int padding, int numBlocks, bool sums)
{

  extern __shared__ float temp[];

  int thid = threadIdx.x + blockDim.x*blockIdx.x;
  int blockId = blockIdx.x;

  int offset = 1;

  if (blockId == (numBlocks - 1)) {
    if (padding) {
      int index = 512 - padding;
      for (int i = 0; i < padding; i++) {
        temp[index] = 0;
        index++;
      }
    }
  }

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

      int ai = (offset*(2*thid+1)-1);
      int bi = (offset*(2*thid+2)-1);
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  /*
    block C
  */
  if (thid == 0 || thid%512 == 0) {
    // 2. writing to sums array
    int last_element = (BLOCK_SIZE - 1 + BLOCK_SIZE*blockId);
    if (!sums){
      sumsArray[blockId] = temp[last_element + CONFLICT_FREE_OFFSET(last_element)];
    }
    temp[last_element + CONFLICT_FREE_OFFSET(last_element)] = 0;
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

      int ai = (offset*(2*thid+1)-1);
      int bi = (offset*(2*thid+2)-1);
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

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

// **===-----------------------------------------------------------===**

__host__ void hostPrescanArray(float *outArray, float *inArray, int numElements){
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

printf("startPrescanArray");

  bool sums = false;
  int numBlocks = 0;
  int numThreads = BLOCK_SIZE/2;
  int padding = padCalculator(numElements);

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
  printf("allocating sumsArray");
  float sumsArray[numBlocks];
  dim3 threadPerBlock(numThreads);
  dim3 blocks(numBlocks);
  printf("first prescan call");
  prescanArray<<<blocks,threadPerBlock>>>(outArray, inArray, sumsArray, BLOCK_SIZE, padding, numBlocks, sums);

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

  float incArray[numBlocks];
  float dumm[0];
  dim3 threadPerBlockSum(numThreads);
  dim3 blocksSum(numSumBlocks);
  prescanArray<<<blocksSum,threadPerBlockSum>>>(incArray, sumsArray, dumm, numSums, sumsPadding, numSumBlocks, sums);


  /*
    4. goes through out array and increments the values
  */
  for (int ii = 0; ii<numElements; ii++) {
    outArray[ii] = outArray[ii] + incArray[(ii%BLOCK_SIZE)];
  }

}


#endif // _PRESCAN_CU_
