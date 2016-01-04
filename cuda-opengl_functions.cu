#include "graphics.h"
#include <math.h>

//texture<float4, 2, cudaReadModeElementType> inTex;	// Used for PBO

__global__ void create_Grid_points_only_kernel_1D(float4 *dDptr, uchar4 *cPtr, int width, int height, float quarter_height, float half_width) {
	
	__shared__ float dev_quarter_height;
	__shared__ float dev_half_width;				// Shared for access speed
	__shared__ float dev_width;
	dev_quarter_height = quarter_height;			// The height of the hexagon
	dev_half_width = half_width;					// The width of the hexagon
	dev_width = 2 * half_width;
	
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D unrolled array
	int j = offset / width;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % width;							// Keeping both offset and i for clarity
    int max_size = width*height;					// Size of the vertex array (actual points)
    float u, v;										// The four components of uv space
		
	v = -1.0f + j * dev_quarter_height;				// The Y-component in uv space
	if (j == 0)										// Are we in row 0?
		u = -1.0f + dev_half_width + i*dev_width;	// Then we start indented (w/2) 
	if (j % 2 == 0)									// Are we in an even row?
		u = -1.0f + dev_half_width + i*dev_width;	// Then we start indented (w/2) 
	if (j == 1)										// Are we in row 1?
		u = -1.0 + i*dev_width;						// Then we start flush (no w/2 offset)
	if (j % 2 != 0)									// Are we in an odd row?
		u = -1.0 + i*dev_width;						// Then start flush (no w/2 offset)
	
	// write output vertex
	if (offset < max_size)
		dDptr[offset] = make_float4(u, v, 0.0f, 1.0f); 
    else 
		dDptr[offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);		
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

__global__ void create_hex_image_on_gpu_kernel_1D(uchar4 *colorPos, int M_color, int N_color, float* Ez, int M, int N, float minval, float maxval)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;	// Calculate linear offset for 1-D case
	int j = offset / M_color;							// Creates a virtual row index for the 1-D case, needed for odd/even row check
	int i = offset % M_color;						// Keeping both offset and i for clarity
	int max_size = M_color*N_color;					// Size of the vertex array (actual points)
	
	int flag_odd = j % 2;
	int flag_even;
	if (flag_odd == 1) flag_even = 0; else flag_even = 1;

	float F;										// Color scalar

	if (offset < max_size) {				        // Check row and column index are in range
		if ((j + 1) % 3 == 0 && i < M) {			// Are we in an Ez row (Every 3rd row) and is i within range?
			int j_ez = j / 3;						// Calculate j offset for Ez 
			int offset_ez = j_ez*M + i;				// Calculate offset for Ez
			F = (Ez[offset_ez-flag_odd] - minval);// Calculate color scalar from Ez
			F = F /	(maxval - minval);
			colorPos[offset].x = 255.f * 0.7 * (F);	// Next 3 lines set the 3 color components
			colorPos[offset].y = 255.f * 0.3 * (F);	// with the color scalar
			colorPos[offset].z = 255.f * 0.5 * (F);
		} else {
			colorPos[offset].x = 0.f;				// Set vertex to white if it is not an 
			colorPos[offset].y = 0.f;				// Ez point
			colorPos[offset].z = 0.f;
		}
		colorPos[offset].w = 0.f;					// Not really sure what the w component does
	} else {								        // If outside the col or row index, set everything to zero
		colorPos[offset].w = 0.f;
		colorPos[offset].x = 0.f;
		colorPos[offset].y = 90.f;
		colorPos[offset].z = 200.f;
	}
	__syncthreads();
}

void createImageOnGpu()	// argument g_odata is the float Ez field 
{									// array, coming in as a device pointer
	int M_dptr = g->im + 1;				// Number of horizontal vertices for hexagonal grid
	int N_dptr = 3 * g->jm + 2;			// Number of vertical vertices for hexagonal grid
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;				// Just the TILE_SIZE squared for a 1-D kernel
	int Bx1 = (TILE_SQUARED - 1 + M_dptr*N_dptr) / TILE_SQUARED;
	dim3 BLK1(Bx1, 1, 1);									// Grid-block dimension for the 1-D case
	dim3 THD1(TILE_SQUARED, 1, 1);							// Thread-block dimension for the 1-D case
	dim3 gridm = dim3(1, 1, 1);
	dim3 blockm = dim3(TILE_SQUARED, 1, 1);
	int  nblocks = gridm.x * gridm.y;
	float minval, maxval, *dvF;
			
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
	    
	minval = -1.0;	maxval = 1.0;	global_min_field = -0.2; global_max_field = 0.2;
	//the following kernel now takes a uchar4 array, not uint
	create_hex_image_on_gpu_kernel_1D << < BLK1, THD1 >> >(cptr, M_dptr, N_dptr, dvF, g->im, g->jm, global_min_field, global_max_field);
}

void create_Grid_points_only(float4* dDptr, uchar4 *cPtr)
{
	// This function and kernel get called only once to create the spatial portion
	// of the vertex buffer object.  The colors will be updated seperately each loop.
	int M_dptr = g->im + 1;										// Number of horizontal vertices for hexagonal grid
	int N_dptr = 3 * g->jm + 2;									// Number of vertical vertices for hexagonal grid
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;				// Just the TILE_SIZE squared for a 1-D kernel
	int screendim = g->im;
	if (g->jm > g->im) screendim = g->jm;

	float h4 = (2.0 / ((float)screendim - 0.75))/4;					// The height of the hexagon
	float wi2 = (2.0 / ((float)screendim + 0.50))/2;				// The width of the hexagon
	
	/* The following four lines are for the 1-D case */
	int Bx1 = (TILE_SQUARED - 1 + M_dptr*N_dptr) / TILE_SQUARED;
	dim3 BLK1(Bx1, 1, 1);									// Grid-block dimension for the 1-D case
	dim3 THD1(TILE_SQUARED, 1, 1);							// Thread-block dimension for the 1-D case
			
	create_Grid_points_only_kernel_1D << < BLK1, THD1 >> >(dDptr, cPtr, M_dptr, N_dptr, h4, wi2);
	
	//int size1 = TILE_SQUARED * Bx1;							// Total array size for the 1-D case
	/* The following four lines are for trouble-shooting purposes */
	/*float4 *vertex_check;
	vertex_check = new float4[size1];
	cudaMemcpy(vertex_check, dDptr, size1 * sizeof(float4), cudaMemcpyDeviceToHost);	
	for (int k = 0; k < size1; k++) {
		if (vertex_check[k].x != 0 && vertex_check[k].y != 0) 
			printf("vertex[%i] x = %f and y = %f\n", k, vertex_check[k].x, vertex_check[k].y);
	}*/
}																					

