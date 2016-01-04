#include "global.h"
#include "cuda_functions.h"
#include <math.h>
// This file belongs to HEXCUDA V2.00

__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time, int m)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int col = offset % m;
	int row = offset / m;
	
	if (time > start_time && time < stop_time) {
		if ((col == x) && (row == y)) {
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
				dEz[offset] = (dEz[offset] + fraction * fraction);
				dImEz[offset] = __double2float_rd(dEz[offset]);
			}			
		}
	}
}

//__global__ void Update_Ez_kernel(double *dEz, float *dEz_float, double *dH1, double *dH2, double *dH3, double *dCez, double hex_t, int M, int N)
//{
//	// Map from threadIdx/blockIdx to cell position
//	int row = blockIdx.y * blockDim.y + threadIdx.y;		// row is also y or j
//	int col = blockIdx.x * blockDim.x + threadIdx.x;		// col is also i or x
//	int offset = row * blockDim.x * gridDim.x + col;		// (i, j)
//	 
//	int flag_odd = row % 2;
//	int flag_even;
//	if (flag_odd == 1) flag_even = 0; else flag_even = 1;
//	int offset_mdH1 = (row - 1) * blockDim.x * gridDim.x + (col + flag_odd);
//	int offset_pdH3 = (row - 1) * blockDim.x * gridDim.x + (col - flag_even);
//
//	int size = M * N;										// used in the in-bounds check
//	double mdH1, mdH2, mdH3, pdH1, pdH2, pdH3;				// place holders for the dH terms
//
//	if (offset < size) {									// is threadIndex in bounds of Ez array?
//		mdH1 = dH1[offset_mdH1];
//		pdH3 = dH3[offset_pdH3];
//		
//		if (row == 0) {										// the following if else statements are
//			mdH1 = 0.f;										// boundary checks to ensure no field cells 
//			pdH3 = 0.f;										// with indexes out of bounds are accessed
//		}
//		if (col == N - 1 && flag_odd == 1)
//			mdH1 = 0.f;
//		if (col == 0 && flag_even == 1)
//			pdH3 = 0.f;
//		
//		if (col == 0)
//			pdH2 = 0.f;
//		else
//			pdH2 = dH2[offset - 1];
//
//		pdH1 = dH1[offset];
//		mdH2 = dH2[offset];
//		mdH3 = dH3[offset];
//		
//		dEz[offset] = dEz[offset] + dCez[offset] * (hex_t *(pdH1 + pdH2 + pdH3 - mdH1 - mdH2 - mdH3));
//		dEz_float[offset] = __double2float_rd(dEz[offset]);
//	}
//	syncthreads();	
//}

//__global__ void Update_3H_kernel(double *dH1, double *dCh1, double *dH2, double *dCh2, double *dH3, double *dCh3, double *dEz, int M, int N)
//{
//	int row = blockIdx.y * blockDim.y + threadIdx.y;					// calculates y or j
//	int col = blockIdx.x * blockDim.x + threadIdx.x;					// calculates x or i
//	int offset = row * blockDim.x * gridDim.x + col;					// calculates the offset for position (i,j)
//	
//	int flag_odd = row % 2;
//	int flag_even;
//	if (flag_odd == 1) flag_even = 0; else flag_even = 1;
//
//	int offset_H1_Ez1 = (row + 1) * blockDim.x * gridDim.x + col - flag_even;	// offset for the positive Ez term in H1 update equation
//	int offset_H2_Ez2 = offset + 1;												// offset for the positive Ez term in H2 update equation
//	int offset_H3_Ez2 = (row + 1) * blockDim.x * gridDim.x + col + flag_odd;	// offset for the positive Ez term in H3 update equation
//	
//	int size = M * N;					// used for the in-bounds check
//	double Ez1_H1, Ez2_H2, Ez2_H3;		// placeholders
//
//	if (offset < size) {				// in-bounds check
//		
//		if (row == N - 1){				// The following if else blocks implement boundary
//			Ez1_H1 = 0.f;					// controls to prevent out-of-bounds access
//			Ez2_H3 = 0.f;
//		}
//		else if (col == 0 && flag_even == 1)
//			Ez1_H1 = 0.f;
//		else {
//			Ez1_H1 = dEz[offset_H1_Ez1];
//			Ez2_H3 = dEz[offset_H3_Ez2];
//		}
//		if (col == M - 1){
//			Ez2_H2 = 0.f;
//			if (flag_odd == 1)
//				Ez2_H3 = 0.f;
//		}
//		else
//			Ez2_H2 = dEz[offset_H2_Ez2];
//
//		dH1[offset] = dH1[offset] + dCh1[offset] * (Ez1_H1 - dEz[offset]);
//		dH2[offset] = dH2[offset] + dCh2[offset] * (dEz[offset] - Ez2_H2);
//		dH3[offset] = dH3[offset] + dCh3[offset] * (dEz[offset] - Ez2_H3);
//		
//	}
//	__syncthreads();
//}
__global__ void Update_Ez_kernel(double *dEz, float *dEz_float, double *dH1, double *dH2, double *dH3, double *dCez, double hex_t, int M, int N)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int col = offset % M;
	int row = offset / M;
	int flag_odd = row % 2;
	int flag_even;
	if (flag_odd == 1) flag_even = 0; else flag_even = 1;
	int offset_mdH1 = offset - M + flag_odd;				// (i + flag_odd, j - 1)
	int offset_pdH3 = offset - M - flag_even;
	int size = M * N;										// used in the in-bounds check
	double mdH1, mdH2, mdH3, pdH1, pdH2, pdH3;				// place holders for the dH terms
	
	if (offset < size) {									// is threadIndex in bounds of Ez array?
		
		if (row == 0) {										// the following if else statements are
			mdH1 = 0.f;										// boundary checks to ensure no field cells 
			pdH3 = 0.f;										// with indexes out of bounds are accessed
		}
		else if (col == M - 1 && flag_odd == 1) {
			mdH1 = 0.f;
			pdH3 = dH3[offset_pdH3];
		}
		else if (col == 0 && flag_even == 1){
			mdH1 = dH1[offset_mdH1];
			pdH3 = 0.f;
		}
		else {
			mdH1 = dH1[offset_mdH1];
			pdH3 = dH3[offset_pdH3];
		}

		if (col == 0)
			pdH2 = 0.f;
		else
			pdH2 = dH2[offset - 1];

		pdH1 = dH1[offset];
		mdH2 = dH2[offset];
		mdH3 = dH3[offset];

		dEz[offset] = dEz[offset] + dCez[offset] * (hex_t *(pdH1 + pdH2 + pdH3 - mdH1 - mdH2 - mdH3));
		dEz_float[offset] = __double2float_rd(dEz[offset]);
	}
	syncthreads();
}

__global__ void Update_3H_kernel(double *dH1, double *dCh1, double *dH2, double *dCh2, double *dH3, double *dCh3, double *dEz, int M, int N)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int col = offset % M;
	int row = offset / M;
	int flag_odd = row % 2;
	int flag_even;
	if (flag_odd == 1) flag_even = 0; else flag_even = 1;
	int size = M * N;										// used in the in-bounds check
	int offset_H1_Ez1 = offset + M - flag_even;
	int offset_H2_Ez2 = offset + 1;
	int offset_H3_Ez2 = offset + M + flag_odd;
	double Ez1_H1, Ez2_H2, Ez2_H3;		// placeholders	

	if (offset < size) {				// in-bounds check

		if (row == N - 1){				// The following if else blocks implement boundary
			Ez1_H1 = 0.f;					// controls to prevent out-of-bounds access
			Ez2_H3 = 0.f;
		}
		else if (col == 0 && flag_even == 1)
			Ez1_H1 = 0.f;
		else {
			Ez1_H1 = dEz[offset_H1_Ez1];
			Ez2_H3 = dEz[offset_H3_Ez2];
		}
		if (col == M - 1){
			Ez2_H2 = 0.f;
			if (flag_odd == 1)
				Ez2_H3 = 0.f;
		}
		else
			Ez2_H2 = dEz[offset_H2_Ez2];

		dH1[offset] = dH1[offset] + dCh1[offset] * (Ez1_H1 - dEz[offset]);
		dH2[offset] = dH2[offset] + dCh2[offset] * (dEz[offset] - Ez2_H2);
		dH3[offset] = dH3[offset] + dCh3[offset] * (dEz[offset] - Ez2_H3);

	}
	__syncthreads();
}
void update_all_fields_hex_CUDA()
{
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;
	int Bx = (TILE_SQUARED - 1 + g->nCells) / TILE_SQUARED;
	dim3 BLK(Bx, 1, 1);
	dim3 THD(TILE_SQUARED, 1, 1);
	double factor = g->cdtds / g->N_lambda;
	
	// Debugging variables //
	//float *ez_check;
	/*double *ez_check_double;*/
	//ez_check = new float[M*N];
	//ez_check_double = new double[M*N];
	//double *h1_check, *h2_check, *h3_check;
	//h1_check = new double[M*N];
	//h2_check = new double[M*N];
	//h3_check = new double[M*N];
	// End Debugging variables //

	g->time += 1;										// Must advance time manually here
	// Launch a kernel on the GPU with one thread for each element.
	Update_3H_kernel << <BLK, THD >> >(dev_h1, dev_ch1, dev_h2, dev_ch2, dev_h3, dev_ch3, dev_ez, g->im, g->jm);	// may need to reduce the last two arguments to g->im-1 and g->jm-1
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
	
	// Launch a kernel on the GPU with one thread for each element.
	Update_Ez_kernel << <BLK, THD >> >(dev_ez, dev_ez_float, dev_h1, dev_h2, dev_h3, dev_cez, g->hex_t, g->im, g->jm);	// may need to reduce the last two arguments to g->im-1 and g->jm-1
	checkErrorAfterKernelLaunch();						// Check for any errors launching the kernel
	deviceSyncAfterKernelLaunch();						// Do a device sync 
	
	Source_Update_Kernel << <BLK, THD >> >(dev_ez, dev_ez_float, g->src_i, g->src_j, 0, g->time, factor, 128, g->N_lambda, g->cdtds, 0, g->maxTime, g->im);
}

void resetBeforeExit() {

	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaDeviceReset failed!");
	
}

void pickGPU(int gpuid) {

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(gpuid);
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
}

void checkErrorAfterKernelLaunch() {
	
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
}

void deviceSyncAfterKernelLaunch() {

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel.\n", cudaStatus);
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