// This file belongs to HEXCUDA V2.00

#include "global.h"
float *dev_ez_float;
double *dev_ez;
double *dev_cez;
double *dev_h1;
double *dev_ch1;
double *dev_h2;
double *dev_ch2;
double *dev_h3;
double *dev_ch3;
float *dvminimum_field_value;
float *dvmaximum_field_value;
int plotting_step;
Grid *g = new Grid;

//  Vertex buffer stuff
float4 *dptr;					// The vertex part of the vbo - generated only once
uchar4 *cptr;					// The color part of the vbo - generated each time loop

// Hexagonal stuff
const double c = 299792458.0;
const double e0 = 8.85418782e-12;		// electric permittivity of free space
const double u0 = 4 * PI *1e-7;			// magnetic permeability of free space
const double imp0 = sqrt(u0 / e0);		// impedance of free space
const int TILE_SIZE = 32;
const int slowdown = 25;

const unsigned int RestartIndex = 0xffffffff;		// Used for primitive restart (VBO)
mappedBuffer_t vertexVBO = { NULL, sizeof(float4), NULL };
mappedBuffer_t colorVBO = { NULL, sizeof(uchar4), NULL };
