#include "global.h"
#include "cuda_functions.h"
#include <math.h>
// This file belongs to HEXCUDA V2.00

__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x *gridDim.x + col;
	if (time > start_time && time < stop_time) {
		if ((col == src_pos_x) && (row == src_pos_y)) {
			if (type == 0) {        // Cosine
				dEz[offset] = dEz[offset] + cos(2 * PI*factor * time);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
			else if (type == 1) {   // Sine
				dEz[offset] = dEz[offset] + sin(2 * PI*factor * time);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
			else if (type == 2) {   // Ricker Wavelet
				double fraction = PI*(Sc * time - loc) / (ppw - 1.0);
				dEz[offset] = dEz[offset] + fraction * fraction;
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}
		}
	}
}

__global__ void Update_Ez_kernel(double *dEz, float *dEz_float, double *dH1, double *dH2, double *dH3, double *dCez, double hex_t, int M, int N, int time)
{
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x * gridDim.x + col;					// (i, j)
	int ezoffset = (row + 1) * blockDim.x * gridDim.x + (col + 1);		// (i+1, j+1)
	int offset_t1 = row * blockDim.x * gridDim.x + (col + 1);			// (i+1, j)
	int offset_t2 = (row + 1) * blockDim.x * gridDim.x + col;			// (i, j+1)
	int size_Ez = M * N;

	////////////////////////////////////////////////////////////////////////////////////
	// Calculate Ez
	if (row == N - 1) ezoffset = col + 1;
	if (col == M - 1) ezoffset = (row + 1) * blockDim.x * gridDim.x;
	if (row == M - 1 && col == N - 1) ezoffset = 0;
	if (ezoffset < size_Ez) {
		if (row == N - 1){
			dEz[ezoffset] = 0.0;
			dEz_float[offset] = 0.0;
		}
		else{
			dEz[ezoffset] = dEz[ezoffset] + dCez[ezoffset] * (hex_t * (dH1[offset] + dH2[offset] + dH3[offset] -
				dH1[offset_t1] - dH2[ezoffset] - dH3[offset_t2]));
			dEz_float[offset] = __double2float_rd(dEz[offset]);
		}
	}
	/*if (row == src_pos_y && col == src_pos_x)
	{
		dEz[offset] = cos(2 * PI*time / 25);
		dEz_float[offset] = __double2float_rd(dEz[offset]);
	}*/
	__syncthreads();		
}

__global__ void Update_3H_kernel(double *dH1, double *dCh1, double *dH2, double *dCh2, double *dH3, double *dCh3, double *dEz, int M, int N)
{
	/*--------------------------------------------------------------------------------*/
	/* H1 Block */
	// Map from threadIdx/blockIdx to cell position
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * blockDim.x * gridDim.x + col;
	int ezoffset1 = row * blockDim.x * gridDim.x + (col + 1);
	int ezoffset2 = (row + 1) * blockDim.x * gridDim.x + (col + 1);
	int ezoffset3 = (row + 1) * blockDim.x * gridDim.x + col;
	int size_H = M * N;
	/*--------------------------------------------------------------------------------*/
	// Calculate H1, H2, H3
	if (offset < size_H) {
		if (row == N - 1) {
			dH1[offset] = 0.0;
			dH2[offset] = 0.0;
			dH3[offset] = 0.0;
		}
		else if (col == M - 1) {
			dH1[offset] = 0.0;
			dH2[offset] = 0.0;
			dH3[offset] = 0.0;
		}
		else {
			dH1[offset] = dH1[offset] + dCh1[offset] * (dEz[ezoffset1] - dEz[ezoffset2]);
			dH2[offset] = dH2[offset] + dCh2[offset] * (dEz[offset] - dEz[ezoffset2]);
			dH3[offset] = dH3[offset] + dCh3[offset] * (dEz[ezoffset3] - dEz[ezoffset2]);
		}
	}
	__syncthreads();
	/*--------------------------------------------------------------------------------*/

}


void update_all_fields_hex_CUDA()
{
	// Calculate CUDA grid dimensions.  Block dimension fixed at 32x32 threads
	int Bx = (g->im + (TILE_SIZE - 1)) / TILE_SIZE;
	int By = (g->jm + (TILE_SIZE - 1)) / TILE_SIZE;
	dim3 BLK(Bx, By, 1);
	dim3 THD(TILE_SIZE, TILE_SIZE, 1);
	double factor = Sc / N_lambda;

	// Launch a kernel on the GPU with one thread for each element.
	Update_3H_kernel << <BLK, THD >> >(dev_h1, dev_ch1, dev_h2, dev_ch2, dev_h3, dev_ch3, dev_ez, g->im, g->jm);	// may need to reduce the last two arguments to g->im-1 and g->jm-1

	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 

	// Launch a kernel on the GPU with one thread for each element.
	Update_Ez_kernel << <BLK, THD >> >(dev_ez, dev_ez_float, dev_h1, dev_h2, dev_h3, dev_cez, g->hex_t, g->im, g->jm, g->time);	// may need to reduce the last two arguments to g->im-1 and g->jm-1

	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 

	Source_Update_Kernel << <BLK, THD >> >(dev_ez, dev_ez_float, src_pos_x, src_pos_y, 2, g->time, factor, 150, N_lambda, Sc, 0, 1500);
	g->time += 1;										// Must advance time manually here
	
	/*float *ez_check;
	ez_check = new float[M*N];
	cudaMemcpy(ez_check, dev_ez_float, M*N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int k = 0; k < M*N; k++) {
		if (ez_check[k] != 0)
			printf("vertex[%i] Ez = %f\n", k, ez_check[k]);
	}*/
}

void resetBeforeExit() {

	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		//return 1;
	}
	
}

void pickGPU(int gpuid) {

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(gpuid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

void checkErrorAfterKernelLaunch() {
	
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}

void deviceSyncAfterKernelLaunch() {

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel.\n", cudaStatus);
	}
}

void initializeHexGlobalDevicePointers() {
	
	// Initialize the extern variables below prior to first use
	dev_h1 = 0;  dev_h2  = 0; dev_h3 = 0;		// The double-precision Hx field on Device memory
	dev_ez = 0;	 dev_cez = 0;					// Same for Ez
	dev_ch1 = 0; dev_ch2 = 0; dev_ch3 = 0;
	dev_ez_float = 0;							// The single-precision fields on Device memory	
}

int copyHexTMzArraysToDevice()
{
	int size = g->nCells;
	int size_f = sizeof(float);
	int size_d = sizeof(double);
	cudaError_t et;

	et = cudaMalloc((void**)&dev_h1,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ch1,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_h2,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ch2, 		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_h3,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ch3,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_cez,		size*size_d);		if (et == cudaErrorMemoryAllocation) return 1;
	et = cudaMalloc((void**)&dev_ez_float,	size*size_f);		if (et == cudaErrorMemoryAllocation) return 1;

	// Note that the float copies of the field components do not need to be copied because
	// they are generated by the update kernel.
	cudaMemcpy(dev_h1,	g->h1,	size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ch1, g->ch1, size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h2,	g->h2,	size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ch2, g->ch2, size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h3,	g->h3,	size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ch3, g->ch3, size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ez,	g->ez,	size*size_d,	cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cez, g->cez, size*size_d,	cudaMemcpyHostToDevice);
	
	et = cudaMalloc((void**)&dvminimum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	
	et = cudaMalloc((void**)&dvmaximum_field_value, sizeof(float)*TILE_SIZE);	if (et == cudaErrorMemoryAllocation) return 1;	
	
	return 0;
}

bool copyHexFieldSnapshotsFromDevice()
{
	int size = g->nCells;
	int size_d = sizeof(double);
	int size_f = sizeof(float);		// only for debugging use 

	// Copy an electric field frame.
	cudaMemcpy(g->ez, dev_ez, size * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->h1, dev_h1, size * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->h2, dev_h2, size * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->h3, dev_h3, size * size_d, cudaMemcpyDeviceToHost);
	cudaMemcpy(g->ez_float, dev_ez_float, size *size_f, cudaMemcpyDeviceToHost);

	for (int i=0; i < (g->nCells); i++){
		if (g->ez_float[i] > 0.1 || g->ez_float[i] < -0.1) {
			//printf("g->ez_float[%i] = %g\n", i, g->ez_float[i]);
		}
	}

	return true;
}

bool deallocateHexCudaArrays()
{
	cudaFree(dev_h1);
	cudaFree(dev_ch1);
	cudaFree(dev_h2);
	cudaFree(dev_ch2);
	cudaFree(dev_h3);
	cudaFree(dev_ch3);
	cudaFree(dev_ez);
	cudaFree(dev_cez);
	cudaFree(dev_ez_float);
	cudaFree(dvminimum_field_value);
	cudaFree(dvmaximum_field_value);

	return true;
}