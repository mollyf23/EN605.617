//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CONSTANT MEMORY
// the scalar a and the array size are the same for every thread, so they're broadcast from constant memory
__constant__ float const_a;
__constant__ int const_n;

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// wrapper for synching the device and returning the elapsed time in milliseconds

__host__ float synch_and_end(cudaEvent_t start_time)
{
	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	cudaEventDestroy(start_time);
	cudaEventDestroy(end_time);
	return delta;
}

//saxpy from module 4 example, combined with dynamicReverse from module 5 shared_memory2.cu
// works on either device or mapped host pointers
// each block's x values are reversed through shared memory: y[i] = a*x[reversed i] + y[i]
__global__
void saxpy_dynamicreverse(float *x, float *y)
{
  // SHARED MEMORY:
  // run dynamic reverse on each block's x values, then compute y[i] = a*x[reversed i] + y[i]
  extern __shared__ float s_x[];

  int t = threadIdx.x;
  int i = blockIdx.x*blockDim.x + t;
  int tr = blockDim.x-t-1;

  if (i < const_n) s_x[t] = x[i];
  __syncthreads();

  if (i < const_n) {
    // REGISTER MEMORY: each thread's values live in local variables
    // use the s_x array to get the reversed x value for this thread's index
    float xr = s_x[tr];
    float yi = y[i];
    y[i] = const_a*xr + yi;
  }
}

// run the kernel and return the elapsed time in milliseconds
__host__ float run_saxpy_dynamicreverse(float *x, float *y, int n, int num_threads)
{
	int num_blocks = (n + num_threads - 1) / num_threads;

	cudaEvent_t start_time = get_time();

	saxpy_dynamicreverse<<<num_blocks, num_threads, num_threads*sizeof(float)>>>(x, y);

	return synch_and_end(start_time);
}

void init_data(float *x, float *y, int n)
{
	for (int i = 0; i < n; i++) {
		x[i] = (float)i;
		y[i] = 2.0f;
	}
}

// Kernel reads/writes pinned host memory directly (zero-copy), no cudaMemcpy
void run_host_memory(int n, int num_threads, float a)
{
	float *h_x, *h_y, *d_x, *d_y;

    // use mapped pinned host memory for x and y, so the kernel can read/write them directly
	cudaHostAlloc(&h_x, n*sizeof(float), cudaHostAllocMapped);
	cudaHostAlloc(&h_y, n*sizeof(float), cudaHostAllocMapped);

    // set the device pointers to the mapped host memory
	cudaHostGetDevicePointer(&d_x, h_x, 0);
	cudaHostGetDevicePointer(&d_y, h_y, 0);

	init_data(h_x, h_y, n);

	float duration = run_saxpy_dynamicreverse(d_x, d_y, n, num_threads);

	printf("Host memory: %8.3fms\n", duration);

	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
}

// Same pattern as module4/host_memory.cu: copy to global memory, run, copy back
void run_global_memory(int n, int num_threads, float a)
{
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));

	cudaMalloc(&d_x, n*sizeof(float));
	cudaMalloc(&d_y, n*sizeof(float));

	init_data(x, y, n);

	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

	float duration = run_saxpy_dynamicreverse(d_x, d_y, n, num_threads);

	cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

	printf("Global memory: %8.3fms\n", duration);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}

int main(int argc, char** argv)
{
	// allow kernels to access mapped host memory; must be set before any other CUDA call
	cudaSetDeviceFlags(cudaDeviceMapHost);

	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	// (n is always a multiple of blockSize, so every block is full and the reverse stays in bounds)
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	const int n = totalThreads;
	const float a = 2.0f;
	printf("SAXPY - Dynamic Reverse on %d elements, block size %d\n", n, blockSize);

	// copy the scalars into constant memory for the kernel
	cudaMemcpyToSymbol(const_a, &a, sizeof(float));
	cudaMemcpyToSymbol(const_n, &n, sizeof(int));

	run_host_memory(n, blockSize, a);
	run_global_memory(n, blockSize, a);

	cudaDeviceReset();
	return 0;
}
