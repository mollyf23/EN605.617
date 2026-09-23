//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_PATHS   3
#define ITERATIONS  512 // number of iterations in each path
#define WARP_SIZE   32

// define 3 different paths of work 

//path 0 - uses fmaf() to perform a multiply-add operation
__device__ float path0(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fmaf(acc, 0.30f, x);
	}
	return acc;
}

// path 1 - uses fabsf() to perform an absolute value operation
__device__ float path1(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fabsf(acc) * 0.40f + x * x;
	}
	return acc;
}

// path 2 - uses fminf() to perform a minimum operation
__device__ float path2(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fminf(acc, x) + acc * 0.50f;
	}
	return acc;
}

//perform same op on everything
__global__ void uniform_kernel(const float* in, float* out, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	out[i] = path0(in[i]);
}

//perform different ops on different elements
__global__ void interleaved_kernel(const float* in, float* out, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	// interleave the paths so that every thread takes a different path
	const int path = i % NUM_PATHS;          
	const float x = in[i];

	if (path == 0) {
		out[i] = path0(x);
	} else if (path == 1) {
		out[i] = path1(x);
	} else {
		out[i] = path2(x);
	}
}

// same branch and same total work as interleaved_kernel, but the path only
// changes at a warp boundary, so all 32 threads in a warp agree on it
__global__ void warp_aligned_kernel(const float* in, float* out, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	const int path = (i / WARP_SIZE) % NUM_PATHS;
	const float x = in[i];

	if (path == 0) {
		out[i] = path0(x);
	} else if (path == 1) {
		out[i] = path1(x);
	} else {
		out[i] = path2(x);
	}
}

// cpu counterparts of the device functions for validation

float cpu_path0(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fmaf(acc, 0.30f, x);
	}
	return acc;
}

float cpu_path1(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fabsf(acc) * 0.40f + x * x;
	}
	return acc;
}

float cpu_path2(float x)
{
	float acc = x;
	for (int k = 0; k < ITERATIONS; ++k) {
		acc = fminf(acc, x) + acc * 0.50f;
	}
	return acc;
}

void cpu_uniform(const float* in, float* out, int n)
{
	for (int i = 0; i < n; ++i) {
		out[i] = cpu_path0(in[i]);
	}
}

void cpu_interleaved(const float* in, float* out, int n)
{
	for (int i = 0; i < n; ++i) {
		const int path = i % NUM_PATHS;
		const float x = in[i];

		if (path == 0) {
			out[i] = cpu_path0(x);
		} else if (path == 1) {
			out[i] = cpu_path1(x);
		} else {
			out[i] = cpu_path2(x);
		}
	}
}

void cpu_warp_aligned(const float* in, float* out, int n)
{
	for (int i = 0; i < n; ++i) {
		const int path = (i / WARP_SIZE) % NUM_PATHS;
		const float x = in[i];

		if (path == 0) {
			out[i] = cpu_path0(x);
		} else if (path == 1) {
			out[i] = cpu_path1(x);
		} else {
			out[i] = cpu_path2(x);
		}
	}
}

static double get_time(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1000.0 + t.tv_nsec / 1000000.0;
}

int main(int argc, char** argv)
{
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
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	const int num_elements = totalThreads;
	const size_t size_in_bytes = num_elements * sizeof(float);

	float *cpu_in          = (float *)malloc(size_in_bytes);
	float *cpu_out_uniform = (float *)malloc(size_in_bytes);
	float *cpu_out_interle = (float *)malloc(size_in_bytes);
	float *gpu_result      = (float *)malloc(size_in_bytes);

	for (int i = 0; i < num_elements; i++) {
		cpu_in[i] = (float)(i % 1000) / 1000.0f;
	}

	float *gpu_in;
	float *gpu_out;

	cudaMalloc((void **)&gpu_in, size_in_bytes);
	cudaMalloc((void **)&gpu_out, size_in_bytes);

	cudaMemcpy(gpu_in, cpu_in, size_in_bytes, cudaMemcpyHostToDevice);

	printf("Elements: %d, Blocks: %d, Threads per block: %d\n",
			num_elements, numBlocks, blockSize);
	printf("Paths: %d, Iterations per path: %d\n\n", NUM_PATHS, ITERATIONS);

	double c0;
	double cpu_uniform_ms, cpu_interleaved_ms, cpu_warp_aligned_ms;

	//uniform gpu and cpu 

	uniform_kernel<<<numBlocks, blockSize>>>(gpu_in, gpu_out, num_elements);
	cudaMemcpy(gpu_result, gpu_out, size_in_bytes, cudaMemcpyDeviceToHost);

	c0 = get_time();
	cpu_uniform(cpu_in, cpu_out_uniform, num_elements);
	cpu_uniform_ms = get_time() - c0;

	//interleaved gpu and cpu

	interleaved_kernel<<<numBlocks, blockSize>>>(gpu_in, gpu_out, num_elements);
	cudaMemcpy(gpu_result, gpu_out, size_in_bytes, cudaMemcpyDeviceToHost);

	c0 = get_time();
	cpu_interleaved(cpu_in, cpu_out_interle, num_elements);
	cpu_interleaved_ms = get_time() - c0;

	//warp aligned gpu and cpu

	warp_aligned_kernel<<<numBlocks, blockSize>>>(gpu_in, gpu_out, num_elements);
	cudaMemcpy(gpu_result, gpu_out, size_in_bytes, cudaMemcpyDeviceToHost);

	c0 = get_time();
	cpu_warp_aligned(cpu_in, cpu_out_interle, num_elements);
	cpu_warp_aligned_ms = get_time() - c0;

	printf("CPU timing:\n");
	printf("no branch:    %9.3f\n", cpu_uniform_ms);
	printf("branching:    %9.3f\n", cpu_interleaved_ms);
	printf("warp aligned: %9.3f\n", cpu_warp_aligned_ms);

	printf("gpu timing determined by using ncu profiler.\n");

	cudaFree(gpu_in);
	cudaFree(gpu_out);

	free(cpu_in);
	free(cpu_out_uniform);
	free(cpu_out_interle);
	free(gpu_result);

	return EXIT_SUCCESS;
}
