#include <stdio.h>
#include <time.h>
#include <cuda.h>

__global__ void arrayAddition(int *device_a, int *device_b, int *device_result, int sizeOfArray)
{

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadId < sizeOfArray)
    device_result[threadId] = device_a[threadId] + device_b[threadId];
}

__global__ void arrayMultiplication(int *device_a, int *device_b, int *device_result, int sizeOfArray)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadId < sizeOfArray)
    device_result[threadId] = device_a[threadId] * device_b[threadId];
}

// combines the outputs of the add and multiply kernels: product - sum
__global__ void arraySubtraction(int *device_product, int *device_sum, int *device_result, int sizeOfArray)
{
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  if (threadId < sizeOfArray)
    device_result[threadId] = device_product[threadId] - device_sum[threadId];
}

/*
 * Three streams with dependencies:
 *   streamAdd:     copy a and b to the device, then sum = a + b
 *   streamMul:     waits for the copies, then product = a * b
 *   streamCombine: waits for both kernels to complete, then result = product - sum
 */
void stream_dependencies(int totalThreads, int blockSize, bool printResults = true)
{

  int numBlocks = totalThreads / blockSize;

  // validate command line arguments
  if (totalThreads % blockSize != 0)
  {
    ++numBlocks;
    totalThreads = numBlocks * blockSize;

    printf("Warning: Total thread count is not evenly divisible by the block size\n");
    printf("The total number of threads will be rounded up to %d\n", totalThreads);
  }

  int *host_a, *host_b, *host_result;
  int *device_a, *device_b, *device_sum, *device_product, *device_result;

  cudaStream_t streamAdd, streamMul, streamCombine;
  // create the streams. we make them asynchronous.. the only "synch" operations are
  // handled by making them wait for certain events.
  cudaStreamCreate(&streamAdd);
  cudaStreamCreate(&streamMul);
  cudaStreamCreate(&streamCombine);

  // start/stop time the whole run, the others mark when each stage finishes
  cudaEvent_t start, copyDone, addDone, mulDone, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&copyDone);
  cudaEventCreate(&addDone);
  cudaEventCreate(&mulDone);
  cudaEventCreate(&stop);

  // device memory (global)
  cudaMalloc((void **)&device_a, totalThreads * sizeof(int));
  cudaMalloc((void **)&device_b, totalThreads * sizeof(int));
  cudaMalloc((void **)&device_sum, totalThreads * sizeof(int));
  cudaMalloc((void **)&device_product, totalThreads * sizeof(int));
  cudaMalloc((void **)&device_result, totalThreads * sizeof(int));

  // allocates page locked memory on the host (pinned memory)
  // need it to be page locked in order to copy it in a stream
  cudaHostAlloc((void **)&host_a, totalThreads * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_b, totalThreads * sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_result, totalThreads * sizeof(int), cudaHostAllocDefault);

  for (int index = 0; index < totalThreads; index++)
  {
    host_a[index] = rand() % 10;
    host_b[index] = rand() % 10;
  }

  cudaEventRecord(start, streamAdd); // just for timing purposes

  // copy inputs to the device in streamAdd stream.
  cudaMemcpyAsync(device_a, host_a, totalThreads * sizeof(int), cudaMemcpyHostToDevice, streamAdd);
  cudaMemcpyAsync(device_b, host_b, totalThreads * sizeof(int), cudaMemcpyHostToDevice, streamAdd);
  cudaEventRecord(copyDone, streamAdd);

  // run the array addition in streamAdd (already ordered after the copies).
  arrayAddition<<<numBlocks, blockSize, 0, streamAdd>>>(device_a, device_b, device_sum, totalThreads);
  cudaEventRecord(addDone, streamAdd);

  // run the multiplication but we must WAIT until the copyDone event fires.
  cudaStreamWaitEvent(streamMul, copyDone, 0);
  arrayMultiplication<<<numBlocks, blockSize, 0, streamMul>>>(device_a, device_b, device_product, totalThreads);
  cudaEventRecord(mulDone, streamMul);

  // streamcombine we will make it wait for BOTH other streams to be done.
  cudaStreamWaitEvent(streamCombine, addDone, 0);
  cudaStreamWaitEvent(streamCombine, mulDone, 0);
  arraySubtraction<<<numBlocks, blockSize, 0, streamCombine>>>(device_product, device_sum, device_result, totalThreads);
  cudaMemcpyAsync(host_result, device_result, totalThreads * sizeof(int), cudaMemcpyDeviceToHost, streamCombine);
  cudaEventRecord(stop, streamCombine);

  // host command to wait for all the kernels to finish
  cudaEventSynchronize(stop);

  // all the timings we need to track
  float copyTime, addTime, mulTime, totalTime;
  cudaEventElapsedTime(&copyTime, start, copyDone);
  cudaEventElapsedTime(&addTime, start, addDone);
  cudaEventElapsedTime(&mulTime, start, mulDone);
  cudaEventElapsedTime(&totalTime, start, stop);

  if (printResults)
  {
    printf("\n Streaming with dependencies: %d elements, block size %d\n", totalThreads, blockSize);
    printf("  Copies done:   %.3f ms\n", copyTime);
    printf("  Add done:      %.3f ms\n", addTime);
    printf("  Multiply done: %.3f ms\n", mulTime);
    printf("  Combine + copy back done: %.3f ms\n", totalTime);
  }

  // free the memory from this call.
  // without this we'd have GPU memory and pinned memory leak for subsequent calls
  // to this function
  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_result);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_sum);
  cudaFree(device_product);
  cudaFree(device_result);

  // destroy streams. so that subsequent calls also have this cleaned up
  cudaEventDestroy(start);
  cudaEventDestroy(copyDone);
  cudaEventDestroy(addDone);
  cudaEventDestroy(mulDone);
  cudaEventDestroy(stop);
  cudaStreamDestroy(streamAdd);
  cudaStreamDestroy(streamMul);
  cudaStreamDestroy(streamCombine);
}

int main(int argc, char **argv)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Device count: %d\n", deviceCount);

  // use the first device
  cudaSetDevice(0);
  // Printing out some cuda properties to see what device we're on
  for (int device = 0; device < deviceCount; device++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device %d name: %s\n", device, prop.name);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  }

  // WARM UP RUN without printing timings so we dont have latency from gpu lazy load
  stream_dependencies(4096, 256, false);

  // run three dependent streams synchronized with events
  stream_dependencies(4096, 256);
  stream_dependencies(1 << 20, 256);
  stream_dependencies(1 << 24, 256);

  stream_dependencies(4096, 512);
  stream_dependencies(1 << 20, 512);
  stream_dependencies(1 << 24, 512);
}