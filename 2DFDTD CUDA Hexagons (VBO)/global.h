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

typedef struct {						// This structure contains vbo data
	GLuint vbo;
	GLuint typeSize;
#ifdef USE_CUDA3
	struct cudaGraphicsResource *cudaResource;
#else
	void* space;
#endif
} mappedBuffer_t;

//__constant__ unsigned int  dvrgb[256];

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
extern const double dx;
extern const double dt;
extern const double Sc;
extern const double c;
extern const double hex_maximum_distance;
extern const double lambda;
extern const double N_lambda;
extern int iGLUTWindowHandle;          // handle to the GLUT window

extern float *dvminimum_field_value;	// Both of these are passed to the find-min-max-gpu functions
extern float *dvmaximum_field_value;	// to get proper min/max field values for color-scaling
extern float global_min_field;			// calculated by find_min_max_on_gpu
extern float global_max_field;			// calculated by find_min_max_on_gpu

/*
// Note for all the externs declared below:  they have no location in memory until defined somewhere else (or here).  
// Extern <variable type> just declares the variable globally to the program, but it does not exist until
// it has been defined.

//extern float dy;						// differential y-operator
extern float domain_min_x;				// ?
extern float domain_min_y;				// ?
extern float domain_max_x;				// ?
extern float domain_max_y;				// ?

//extern unsigned int rgb[];				// used in createImageOnGpu() in graphics.cpp
extern float* field_data;				// presumably the argument to createImageOnGpu() or something
//extern unsigned int* dvimage_data;

//extern bool show_Ez;					// Used as a flag in visualization
extern int plotting_step;					// Used in IterationAndDisplay; every plotting_step steps arrays will 
// be displayed via OpenGL
extern GLuint pbo_destination;
//extern struct cudaGraphicsResource *cuda_pbo_destination_resource;
//extern GLuint cuda_result_texture;
*/
#endif