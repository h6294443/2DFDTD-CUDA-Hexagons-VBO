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
const double e0 = 8.85418782e-12;		// electric permittivity of free space
const double u0 = 4 * PI *1e-7;			// magnetic permeability of free space
const double imp0 = sqrt(u0 / e0);		// impedance of free space
const int TILE_SIZE = 32;

const unsigned int RestartIndex = 0xffffffff;		// Used for primitive restart (VBO)
mappedBuffer_t vertexVBO = { NULL, sizeof(float4), NULL };
mappedBuffer_t colorVBO = { NULL, sizeof(uchar4), NULL };




//
//// Step 1: Specify source frequency	
//printf("Enter source frequency in kHz: ");
//scanf_s(" %g", &src_f);
//
//// Step 2: Calculate wavelength
//const double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
//
//// Step 3: Specify Sc:
//printf("\n\nEnter desired Courant number (0.1-0.8): ");
//scanf_s(" %g", &Sc);
//if (Sc < 0.1) Sc = 0.1;
//if (Sc > 0.8) Sc = 0.8;
//
//// Step 4: Specify physical domain size in meters
//printf("\n\nEnter the domain width (X) in meters: ");
//scanf_s(" %g", &X);
//printf("\n\nEnter the domain height (Y) in meters: ");
//scanf_s(" %g", &Y);
//
//// Step 5: Specify desired points-per-wavelength N_lambda
//printf("\n\nEnter points-per-wavelength (can be a float): ");
//scanf_s(" %g", &N_lambda);
//
//// Step 6: Calculate dx (this may not be dx as defined)
//const double dx = lambda / N_lambda;				// This is the largest distance possible within one hexagon - from one point to the opposing pointy point
//
//// Step 7: Calculate dt
//const double dt = Sc*dx / c;
//
//// Step 8: Calculate M and N
//int M = (2 * X) / (sqrt(3.0) * dx);
//int N = (4 * Y) / (3.0 * dx);
//
// Step 8: Specify source position (and soon, type)
const int src_pos_x = (int)(0.15*g->im);
const int src_pos_y = (int)(0.5*g->jm);
//
// Step 9: Specify desired slowdown, if any.
const int slowdown = 25;
//
//
//
//
//
//
