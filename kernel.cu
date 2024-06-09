#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "header.cuh"

#include <stdio.h> 
#include <math.h>

#define N 2048
#define TPB 1024 // change to 1025+ to throw sync error (no more than 1024 threads on my system)

const int GRIDSIZE = (N + TPB - 1) / TPB;

float scale(int in, int size) { return ((float)in) / (size - 1); }

__device__ float distance(float x1, float x2) { return sqrt(pow(x2 - x1, 2)); }

__global__ void distanceKernel(float* d_out, float* d_in, float ref) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const float x = d_in[i];
	d_out[i] = distance(x, ref);
	printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]); 
}

int main() {

	deviceProps();

	const float ref = 0.5f; 

	float* in = 0; 
	float* out = 0; 

	cudaError_t inMalErr = cudaMallocManaged(&in, N * sizeof(float));
	if (inMalErr != cudaSuccess) { printf("Input Array Malloc Error: code %d - %s.\n", cudaError(inMalErr), cudaGetErrorString(inMalErr)); return -1; }
	cudaError_t outMalErr = cudaMallocManaged(&out, N * sizeof(float));
	if (outMalErr != cudaSuccess) {
		printf("Output Array Malloc Error: code %d - %s.\n", cudaError(outMalErr), cudaGetErrorString(outMalErr)); return -1; }

	for (int i = 0; i < N; ++i) in[i] = scale(i, N);

	distanceKernel << <GRIDSIZE, TPB >> > (out, in, ref);
	cudaError_t syncErr = cudaGetLastError();
	cudaError_t asyncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess) { printf("Sync Kernel Error: code %d - %s.\n", cudaError(syncErr), cudaGetErrorString(syncErr)); return -1; }
	if (asyncErr != cudaSuccess) {
		printf("Async Kernel Error: code %d - %s.\n", cudaError(asyncErr), cudaGetErrorString(asyncErr)); return -1; }

	cudaFree(in);
	cudaFree(out);

	return 0;
}