#include "graphics.h"
#include <math.h>

//texture<float4, 2, cudaReadModeElementType> inTex;	// Used for PBO

__global__ void create_Grid_points_only_kernel(float4 *dDptr, uchar4 *cPtr, int width, int height, float quarter_height, float half_width) {
	// This kernel turns the blank vertex spatial array into a properly formatted 
	// array of vertices that can be drawn during the display call.  As the spatial
	// information does not change during the simulation, this kernel should get called
	// only once.  Right now, it is being called every iteration.
	// This kernel generates only the Ez points - the centers of the hexagons
	__shared__ float dev_quarter_height;
	__shared__ float dev_half_width;				// Shared for access speed
	__shared__ float dev_width;
	dev_quarter_height = quarter_height;			// The height of the hexagon
	dev_half_width = half_width;					// The width of the hexagon
	dev_width = 2 * half_width;
	
	/* The following XXXX lines are for the 2-D case */
	int i = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate column
	int j = blockIdx.y * blockDim.y + threadIdx.y;	// Calculate row
	int offset = j * blockDim.x * gridDim.x + i;	// Calculate offset

	/* The following two lines are for the 1-D case */
	//int i = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate column for 1-D case
	//int offset = i;									// Keeping both offset and i for clarity
	
	float u, v, w, z;								// The four components of uv space
    
	v = -1.0f + j * dev_quarter_height;				// The Y-component in uv space
	if (j == 0)										// Are we in row 0?
		u = -1.0f + dev_half_width + i*dev_width;	// Then we start indented (w/2) 
	if (j % 2 == 0)									// Are we in an even row?
		u = -1.0f + dev_half_width + i*dev_width;	// Then we start indented (w/2) 
	if (j == 1)										// Are we in row 1?
		u = -1.0 + i*dev_width;						// Then we start flush (no w/2 offset)
	if (j % 2 != 0)									// Are we in an odd row?
		u = -1.0 + i*dev_width;						// Then start flush (no w/2 offset)
	
	w = 0.0f; z = 0.0f;

	// write output vertex
	if (i < width && j < height) {
		dDptr[offset] = make_float4(u, v, w, 1.0f);
		cPtr[offset].x = 255.f;	// Next 3 lines set the 3 color components
		cPtr[offset].y = 0.f;	// with the color scalar
		cPtr[offset].z = 255.f;
		
	}
	else {
		dDptr[offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		cPtr[offset].x = 255.f;	// Next 3 lines set the 3 color components
		cPtr[offset].y = 255.f;	// with the color scalar
		cPtr[offset].z = 255.f;
	}
}

__global__ void find_min_and_max_on_gpu(int nblocks, float* field, 
										float* minimum_field_value, 
										float* maximum_field_value)
{
	__shared__ float minarr[1024];
	__shared__ float maxarr[1024];

	int i = threadIdx.x;
	int nTotalThreads = blockDim.x;

	minarr[i] = field[i];
	maxarr[i] = minarr[i];
	for (int j = 1; j<nblocks; j++)
	{
		minarr[i + nTotalThreads] = field[i + nTotalThreads*j];
		if (minarr[i] > minarr[i + nTotalThreads])
			minarr[i] = minarr[i + nTotalThreads];

		if (maxarr[i] < minarr[i + nTotalThreads])
			maxarr[i] = minarr[i + nTotalThreads];
		__syncthreads();
	}
	__syncthreads();

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		if (threadIdx.x < halfPoint)
		{
			float temp = minarr[i + halfPoint];

			if (temp < minarr[i]) minarr[i] = temp;

			temp = maxarr[i + halfPoint];
			if (temp > maxarr[i]) maxarr[i] = temp;
		}
		__syncthreads();
		nTotalThreads = (nTotalThreads >> 1);
	}
	if (i == 0)
	{
		minimum_field_value[0] = minarr[0];
		maximum_field_value[0] = maxarr[0];
	}
}

//void createColormapOnGpu()
//{
//	cudaError_t et;
//	et = cudaMemcpyToSymbol(dvrgb, rgb, 256 * sizeof(int), 0, cudaMemcpyHostToDevice);
//}

__global__ void create_hex_image_on_gpu(uchar4 *colorPos, int M_color, int N_color, float* Ez, int M, int N, float minval, float maxval)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate column
	int j = blockIdx.y * blockDim.y + threadIdx.y;	// Calculate row
	int offset = j*M_color + i;						// Calculate offset for colorPos
	float F;										// Color scalar
	
	/*int cind;
	float temp;
	temp = minval;
	int ti = (j + 1)*M + i;
	if (j == M - 1) ti = (j)*M + i;
	F = Ez[ti] - minval;
	cind = floor(255 * F / (maxval - minval));
	if (cind > 255) cind = 255;
	g_odata[ci] = dvrgb[cind];*/

	//if (offset < (M_color*N_color)){				// Make sure we are in range
	if (i < M_color && j < N_color) {				// Check row and column index are in range
		if ((j+1) % 3 == 0) {						// Are we in an Ez row? (Every 3rd row)
			int j_ez = j / 3;						// Calculate j offset for Ez 
			int offset_ez = j_ez*M + i;				// Calculate offset for Ez
			F = (Ez[offset_ez] - minval) /			// Calculate color scalar from Ez
				(maxval - minval);	
			colorPos[offset].x = 255.f *0.5*(F);	// Next 3 lines set the 3 color components
			colorPos[offset].y = 255.f *0.5*(F);	// with the color scalar
			colorPos[offset].z = 255.f *0.5*(F);
		}
		else {
			colorPos[offset].x = 255.f;				// Set vertex to white if it is not an 
			colorPos[offset].y = 255.f;				// Ez point
			colorPos[offset].z = 255.f;
		}
		colorPos[offset].w = 0;						// Not really sure what the w component does
	}	
	else {								// If outside the col or row index, set everything to zero
		colorPos[offset].w = 0.f;
		colorPos[offset].x = 0.f;
		colorPos[offset].y = 0.f;
		colorPos[offset].z = 0.f;
	}
	__syncthreads();
}

void createImageOnGpu()	// argument g_odata is the float Ez field 
{												// array, coming in as a device pointer
	int M_dptr = M + 1;				// Number of horizontal vertices for hexagonal grid
	int N_dptr = 3 * N + 2;			// Number of vertical vertices for hexagonal grid
	int Bx = (M_dptr + (TILE_SIZE - 1)) / TILE_SIZE;		// Calculate CUDA grid dimensions.  
	int By = (N_dptr + (TILE_SIZE - 1)) / TILE_SIZE;		// Block dimension fixed at 32x32 threads
	dim3 BLK(Bx, By, 1);
	dim3 THD(TILE_SIZE, TILE_SIZE, 1);
	
	//dim3 block(TILE_SIZE, TILE_SIZE, 1);
	//dim3 block(5, 14, 1);

	//dim3 grid(M / block.x, N / block.y, 1);
	//dim3 grid(1, 1, 1);

	dim3 gridm = dim3(1, 1, 1);
	dim3 blockm = dim3(TILE_SIZE*TILE_SIZE, 1, 1);
	int  nblocks = gridm.x * gridm.y;
	float minval, maxval;
	float *dvF;
		
	//if (show_Ez) dvF = dev_ez_float; else dvF = dev_hx_float;
	dvF = dev_ez_float;

	find_min_and_max_on_gpu << < gridm, blockm >> >(nblocks, dvF, dvminimum_field_value, dvmaximum_field_value);

	cudaMemcpy(&minval, dvminimum_field_value, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&maxval, dvmaximum_field_value, sizeof(float), cudaMemcpyDeviceToHost);

	if (minval > 0.0) minval = 0.0;
	if (maxval < 0.0) maxval = 0.0;
	if (abs(minval) > maxval) maxval = -minval; else minval = -maxval;
	if (minval < global_min_field) global_min_field = minval;
	if (maxval > global_max_field) global_max_field = maxval;

	//cudaMemcpy(g->ez_float, dvF, g->nCells*sizeof(float), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < g->nCells; i++)
	//	if (g->ez_float[i] > 0.001f || g->ez_float[i] < -0.001f)
	//		printf("ez_float[%i] = %f\n", i, g->ez_float[i]);

	//minval = -1.0;	maxval = 1.0;	global_min_field = -1.0; global_max_field = 1.0;
	//the following kernel now takes a uchar4 array, not uint
	create_hex_image_on_gpu << < BLK,THD >> >(cptr, M_dptr, N_dptr, dvF, M, N, global_min_field, global_max_field);
}

void create_Grid_points_only(float4* dDptr, uchar4 *cPtr)
{
	// This function and kernel get called only once to create the spatial portion
	// of the vertex buffer object.  The colors will be updated seperately each loop.
	int M_dptr = M + 1;							// Number of horizontal vertices for hexagonal grid
	int N_dptr = 3 * N + 2;						// Number of vertical vertices for hexagonal grid
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;	// Just the TILE_SIZE squared for a 1-D kernel

	/* The following seven lines are for the 2-D kernel */
	// int Bx = (M_dptr + (TILE_SIZE - 1)) / TILE_SIZE;		// Calculate CUDA grid dimensions.  
	// int By = (N_dptr + (TILE_SIZE - 1)) / TILE_SIZE;		// Block dimension fixed at 32x32 threads
	// int size = Bx * TILE_SIZE * By * TILE_SIZE;			// Total array size
	// dim3 BLK(Bx, By, 1);
	// dim3 THD(TILE_SIZE, TILE_SIZE, 1);
	// dim3 THD(5, 14, 1);
	// BLK = (1, 1, 1);

	/* The following four lines are for the 1-D case */
	int Bx = (TILE_SQUARED - 1 + M_dptr*N_dptr) / TILE_SQUARED;
	dim3 BLK(Bx, 1, 1);									// Grid-block dimension for the 1-D case
	dim3 THD(TILE_SQUARED, 1, 1);						// Thread-block dimension for the 1-D case
	int size = TILE_SQUARED * Bx;						// Total array size for the 1-D case

	float h = 2.0 / ((float)N - 0.75);					// The height of the hexagon
	float wi = 2.0 / ((float)M + 0.50);					// The width of the hexagon
	float4 *vertex_check;
	vertex_check = new float4[size];

	create_Grid_points_only_kernel << < BLK,THD >> >(dDptr, cPtr, M_dptr, N_dptr, (h/4), (wi/2));		
	
	/* The following four lines are for trouble-shooting purposes */
	/*cudaMemcpy(vertex_check, dDptr, size * sizeof(float4), cudaMemcpyDeviceToHost);
	for (int k = 0; k < size; k++) {
		if (vertex_check[k].x != 0 && vertex_check[k].y != 0) 
			printf("vertex[%i] x = %f and y = %f\n", k, vertex_check[k].x, vertex_check[k].y);
	}*/
}																					// so that OpenGL can have it back.

