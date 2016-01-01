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
uint *iptr;						// Not sure what this is yet.

// Hexagonal stuff
const double c = 299792458.0;
double src_f = 1.1e3;                 // Frequency of the source (for a sine or cosine)
const double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
const double dx = 40000;
const double Sc = sqrt(2.f/3.f)-1.f/2.f;
const double dt = Sc*dx/c;
const double N_lambda = lambda / dx;
const double hex_t = dx;

const unsigned int RestartIndex = 0xffffffff;		// Used for primitive restart (VBO)
mappedBuffer_t vertexVBO = { NULL, sizeof(float4), NULL };
mappedBuffer_t colorVBO = { NULL, sizeof(uchar4), NULL };

uint MAX_FPS = 1;