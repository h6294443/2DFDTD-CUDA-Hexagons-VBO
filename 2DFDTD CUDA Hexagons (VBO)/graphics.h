#include "parameters.h"
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "grid_2d_geo.h"
#include "global.h"
#include "cuda_functions.h"

// This file belongs to HEXCUDA V2.00

/* The following are functions from graphics.cpp */
void setImageAndWindowSize();			
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
void createVBO(mappedBuffer_t* mbuf);
void deleteVBO(mappedBuffer_t* mbuf);
void initCuda();
void renderCuda(int drawMode);
void Cleanup(int iExitCode);
bool runFdtdWithFieldDisplay(int argc, char** argv);
void runIterationsAndDisplay();			// Used as display callback function in the glutMainLoop()
bool saveSampledFieldsToFile();
void initGL(int argc, char **argv);

//void createColormapOnGpu();
void createImageOnGpu();
void create_Grid_points_only(float4 *dDptr, uchar4 *cPtr);

// Function prototypes for stuff in cuda-opengl_functions.cu
__global__ void create_Grid_points_only_kernel(float4 *dDptr, uchar4 *cptr, int width, int height, float quarter_height, float half_width);
__global__ void find_min_and_max_on_gpu(int nblocks, float* field,
										float* minimum_field_value,
										float* maximum_field_value);
__global__ void create_hex_image_on_gpu(uchar4 *colorPos, int M_color, int N_color, float* Ez, int M, int N, float minval, float maxval);

