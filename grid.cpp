// This file belongs to HEXCUDA V2.00
#include "global.h"
#include "parameters.h"

void gridInit(Grid *g) {
	g->im = M;
	g->jm = N;
	g->nCells = g->im * g->jm; // this is padded to get H(0,0) values to calculate.  
	// Consequently, Ez(im,:) and Ez(:,jm) values are not 
	// used

	// The following block calculates geometric parameters of each   
	// hexagonal cell.  See grid_2d_geo.h for details on parameters. 
	g->hex_max = dx;
	g->hex_t = 0.5 * g->hex_max;
	g->hex_d = sqrt(0.5) * g->hex_t;
	g->hex_A = sqrt(3.0) / 4 * g->hex_t * g->hex_t * 6;

	g->time = 0;
	g->maxTime = maxTime;
	g->type = twoDGeoTMz;
	g->cdtds = Sc;
	int m, n;

	// Create grid components and coefficients.						 
	g->h1 = new double[g->nCells];
	g->h2 = new double[g->nCells];
	g->h3 = new double[g->nCells];
	g->cez = new double[g->nCells];
	g->ez = new double[g->nCells];
	g->ch1 = new double[g->nCells];
	g->ch2 = new double[g->nCells];
	g->ch3 = new double[g->nCells];
	g->ez_float = new float[g->nCells];
	
	// Set the field and update coefficients for H1-H3
	for (m = 0; m < g->jm; m++) {
		for (n = 0; n < g->im; n++) {
			int offset = n + m * (g->im);
			g->h1[offset] = 0.0;
			g->h2[offset] = 0.0;
			g->h3[offset] = 0.0;
			g->ch1[offset] = dt / (u0 * 2 * g->hex_d);
			g->ch2[offset] = dt / (u0 * 2 * g->hex_d);
			g->ch3[offset] = dt / (u0 * 2 * g->hex_d);
			g->cez[offset] = dt / (e0 * g->hex_A);
			g->ez[offset] = 0;
		}
	}

	printf("Lambda: %g m\n", lambda);
	printf("N-lambda: %g\n", N_lambda);
	printf("dx: %g m\n", dx);
	printf("dt: %g s\n", dt);
	printf("Sc: %g\n", Sc);
}