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

__global__ void uniformAdd(float *outArray, float *incArray, float *temp_out, int numElements) {


  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < numElements){
    outArray[idx] = temp_out[idx] + incArray[(idx/BLOCK_SIZE)];
      
  }
}

// Lab4: Kernel Functions

__global__ void prescanArray(float *outieArray, float *inArray, float *sumsArray, int numElements, int padding, int numBlocks, bool sums, int totalElements)
{

  __shared__ float temp[512+CONFLICT_FREE_OFFSET(512)];


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

  int ai = threadId + (blockId * BLOCK_SIZE); 
  int bi = threadId + (blockId * BLOCK_SIZE) + (BLOCK_SIZE/2);

  int ai2 = threadId;
  int bi2 = threadId + (numElements/2);

  int bankOffsetA = CONFLICT_FREE_OFFSET(ai2);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi2);

  temp[ai2 + bankOffsetA] = (ai < totalElements) ? inArray[ai] : 0;
  temp[bi2 + bankOffsetB] = (bi < totalElements) ? inArray[bi] : 0;


  for (int d = numElements>>1; d > 0; d >>= 1){
    __syncthreads();
    if (threadId < d) {

      /*
        block B
      */

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

    int last_element = 511;
    if (!sums){
      sumsArray[blockId] = temp[last_element + CONFLICT_FREE_OFFSET(last_element)];
    }
    temp[last_element + CONFLICT_FREE_OFFSET(last_element)] = 0;
  }
  for (int d = 1; d<numElements; d*=2){
    offset >>= 1;
    __syncthreads();

    if (threadId < d) {

      /* 
        block d
      */

      int ai = (offset*(2*threadId+1)-1);
      int bi = (offset*(2*threadId+2)-1);
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  
  


  // /* 
  //   block e
  // */

  
  if (ai < totalElements)
    outieArray[ai] = temp[ai2 + bankOffsetA];
  
  if (bi < totalElements)
    outieArray[bi] = temp[bi2 + bankOffsetB];

 
}



// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

// **===-----------------------------------------------------------===**

void hostPrescanArray(float *out, float *in, int numElements){
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
  int padding = padCalculator(numElements);
  if (padding == 512)
    padding = 0;

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

  int numSums = numBlocks;
  int numSumBlocks = 0;

  int sumsPadding = padCalculator(numSums);
  if (sumsPadding == 512)
    sumsPadding = 0;
  
  if (numSums < BLOCK_SIZE) {
    numSumBlocks = 1;
  }
  else if (numSums % BLOCK_SIZE == 0){
    numSumBlocks = numSums / BLOCK_SIZE;
  }
  else {
    numSumBlocks = (numSums / BLOCK_SIZE) + 1;
  }

  /*
    2. write total sum into sumArray (happens in prescan call)
  */

  dim3 threadPerBlock(numThreads);
  dim3 blocks(numBlocks);
  float* temp_out = NULL;
  float* sumsArray = NULL;
  float* incArray = NULL;

  
  unsigned int temp_size = ((numElements+padding) * sizeof( float));
  unsigned int sums_size = ((numSums+sumsPadding) * sizeof( float));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &temp_out, temp_size ));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &sumsArray, sums_size));
  CUDA_SAFE_CALL( cudaMalloc( (void**) &incArray, sums_size));

  float* zeroes = (float*) malloc( temp_size);
  float* sums_zeroes = (float*) malloc ( sums_size);
  for (int ii = 0; ii<numElements+padding; ii++){
    zeroes[ii] = 0;
  }

  for (int ii = 0; ii < numSums+sumsPadding; ii++){
    sums_zeroes[ii] = 0;
  }


  CUDA_SAFE_CALL( cudaMemcpy( temp_out, zeroes, temp_size, cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy( sumsArray, zeroes, sums_size, cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy( incArray, zeroes, sums_size, cudaMemcpyHostToDevice) );
  
  
  prescanArray<<<blocks,threadPerBlock>>>(temp_out, in, sumsArray, BLOCK_SIZE, padding, numBlocks, sums, numElements);

  /* 
    3. scan block sums. generate incArray
  */
  
  sums = true;
  dim3 threadPerBlockSum(numThreads);
  dim3 blocksSum(numSumBlocks);
  float* dumArray = NULL;
  prescanArray<<<blocksSum,threadPerBlockSum>>>(incArray, sumsArray, dumArray, BLOCK_SIZE, sumsPadding, numSumBlocks, sums, numSums);

  //cudaDeviceSynchronize(); 
  /*
    4. goes through out array and increments the values
  */
  uniformAdd<<<(numElements/512)+1, 512>>>(out, incArray, temp_out, numElements);
  cudaFree(temp_out);
  cudaFree(sumsArray);
	cudaFree(incArray);

}


#endif // _PRESCAN_CU_
