// This file belongs to HEXCUDA V2.00

#ifndef _CUDA_FUNCTIONS_H
#define _CUDA_FUNCTIONS_H

//#include "parameters.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "grid_2d_geo.h"
#include <stdio.h>

void update_all_fields_hex_CUDA();
void resetBeforeExit();
void deviceSyncAfterKernelLaunch();
void pickGPU(int gpuid);
void checkErrorAfterKernelLaunch();
void initializeHexGlobalDevicePointers();
int copyHexTMzArraysToDevice();
bool copyHexFieldSnapshotsFromDevice();
bool deallocateHexCudaArrays();			// used to cudaFree() all device arrays
//the following two are leftovers from the cartesian version of this code (2DFDTD CUDA V2.01)
__global__ void HxHyUpdate_Kernel(double *dHx, double *dChxh, double *dChxe, double *dHy, double *dChyh, double *dChye, double *dEz, int M, int N);
__global__ void EzUpdate2D_Kernel(double *dEz, double *dCezh, double *dCeze, float *dImEz, double *dHx, double *dHy, int DIM, int time);
__global__ void Source_Update_Kernel(double *dEz, float *dImEz, int x, int y, int type, int time, double factor, int loc, double ppw, double Sc, int start_time, int stop_time, int m);
// The following are meant for the hexagonal grid
__global__ void Update_3H_kernel(double *dH1, double *dCh1, double *dH2, double *dCh2, double *dH3, double *dCh3, double *dEz, int M, int N);
__global__ void Update_Ez_kernel(double *dEz, float *dEz_float, double *dH1, double *dH2, double *dH3, double *dCez, double hex_t, int M, int N);
__global__ void Generate_Grid_kernel(float4 *dPtr, int mesh_width, int mesh_height);

#endif


