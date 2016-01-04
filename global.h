// This file is intended for external variable declarations so that OpenGL functions 
// have access to program variables without directly passing them.

// This file belongs to HEXCUDA V2.00

#pragma once
#include <cstdlib>
#include <cstdio>
#include <helper_math.h>
#ifndef _global_h
#define _global_h
#define USE_CUDA3

#include "grid_2d_geo.h"
#include <GL/glew.h>
#define PI 3.14159265359

typedef struct {						// This structure contains vbo data
	GLuint vbo;
	GLuint typeSize;
#ifdef USE_CUDA3
	struct cudaGraphicsResource *cudaResource;
#else
	void* space;
#endif
} mappedBuffer_t;

// vbo variables
extern mappedBuffer_t vertexVBO;
extern mappedBuffer_t colorVBO;
extern const unsigned int RestartIndex;
extern float4 *dptr;					// The vertex part of the vbo - generated only once
extern uchar4 *cptr;					// The color part of the vbo - generated each time loop
extern uint *iptr;						// Not sure what this is yet.

extern Grid *g;
extern unsigned int *image_data;		// the container for the normalized color field data
extern double *dev_h1;					// Now the global device pointer for field Hx
extern double *dev_ch1;					// Global device pointer for Chxh
extern double *dev_h2;					// Now the global device pointer for field Hy
extern double *dev_ch2;					// Same
extern double *dev_h3;
extern double *dev_ch3;
extern double *dev_ez;					// Now the global device pointer for field Ez
extern double *dev_cez;					// Same
extern float *dev_ez_float;				// Copy of dev_ez but in single precision

// Hexagonal stuff
//extern const double dx;
//extern const double dt;
//extern double Sc;
extern const double c;
extern const double hex_maximum_distance;
//extern const double lambda;
//extern const double N_lambda;
extern int iGLUTWindowHandle;          // handle to the GLUT window

extern float *dvminimum_field_value;	// Both of these are passed to the find-min-max-gpu functions
extern float *dvmaximum_field_value;	// to get proper min/max field values for color-scaling
extern float global_min_field;			// calculated by find_min_max_on_gpu
extern float global_max_field;			// calculated by find_min_max_on_gpu

//extern uint MAX_FPS;

// stuff that used to be in parameters.h
const int maxTime = 600;				// number of time steps         
//extern int M;							// steps in x-direction
//extern int N;							// steps in y-direction
//extern double X;							// domain width x in meters 
//extern double Y;							// domain height y in meters
extern const int TILE_SIZE;				// Tile size, relates closely to size of a block.  
extern const double e0;					// electric permittivity of free space
extern const double u0;					// magnetic permeability of free space
extern const double imp0;				// impedance of free space
extern const int slowdown;
extern const int src_pos_x;
extern const int src_pos_y;
//extern double src_f;

#endif