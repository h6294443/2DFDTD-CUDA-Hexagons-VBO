// This file belongs to HEXCUDA V2.00
#include "global.h"

void gridInit(Grid *g) {
	
	// Step 1: Specify source frequency	
	printf("Enter source frequency in Hz (Engineering/Scientific notation ok): ");
	scanf_s(" %lf", &g->src_f);
	
	// Step 2: Calculate wavelength
	g->Lambda = c / g->src_f;           // Wavelength of the source (for a sine or cosine)

	// Step 3: Specify Sc:
	printf("\n\nEnter desired Courant number (0.1-0.8): ");
	scanf_s(" %lf", &g->cdtds);
	if (g->cdtds < 0.1) g->cdtds = 0.1;
	if (g->cdtds > 0.8) g->cdtds = 0.8;

	// Step 4: Specify physical domain size in meters
	printf("\n\nEnter the domain width (X) in meters: ");
	scanf_s(" %lf", &g->DOMX);
	printf("\n\nEnter the domain height (Y) in meters: ");
	scanf_s(" %lf", &g->DOMY);

	// Step 5: Specify desired points-per-wavelength N_lambda
	printf("\n\nEnter points-per-wavelength (can be a float): ");
	scanf_s(" %lf", &g->N_lambda);

	// Step 6: Calculate dx (this may not be dx as defined)
	g->dx = g->Lambda / g->N_lambda;				// This is the largest distance possible within one hexagon - from one point to the opposing pointy point

	// Step 7: Calculate dt
	g->dt = g->cdtds*g->dx / c;

	// Step 8: Calculate M and N
	g->im = (2 * g->DOMX) / (sqrt(3.0) * g->dx);
	g->jm = (4 * g->DOMY) / (3.0 * g->dx);

	// Step 8: Specify source position (and soon, type)
	//const int src_pos_x = (int)(0.15*M);
	//const int src_pos_y = (int)(0.5*N);
	g->src_x = 0.65 * g->DOMX;
	g->src_y = 0.50 * g->DOMY;

	// Calculate a few more hexagon-specific stuff
	g->hex_max = g->dx;					// largest distance within one single hexagon
	g->hex_t = 0.5 * g->hex_max;		// from center to furthest outside point	
	g->hex_d = sqrt(3.0)/2 * g->hex_t;	// half-width.  Two of them are one flat width
	g->hex_A = sqrt(3.0) / 4 * g->hex_t * g->hex_t * 6;
	g->src_i = (int)(g->src_x / (2 * g->hex_d));	// source position in the Ez array
	g->src_j = (int)((4 * g->src_y) / (3.0 * g->dx));

	// Step 9: Specify desired slowdown, if any.
	//const int slowdown = 25;
	
	printf("\n\nSimulation parameters\n");
	printf("Domain size: %g m by %g m and %i by %i cells (M by N)\n", g->DOMX, g->DOMY, g->im, g->jm);
	printf("Source frequency = %3g Hz, wavelength = %g m, and ppw = %g\n", g->src_f, g->Lambda, g->N_lambda);
	printf("dx = %g m, dt = %g s, Sc = %g \n", g->dx, g->dt, g->cdtds);
	printf("Source is at (%3g, %3g) meters.\n\n", g->src_x, g->src_y);
	
	g->nCells = g->im * g->jm; // this is padded to get H(0,0) values to calculate.  
	g->time = 0;
	g->maxTime = maxTime;
	g->type = twoDGeoTMz;
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

	// The following implements a PEC in the center of the domain, a circle of radius R.
	// Alternatively, specify the center of the circle.
	double r1 = 0.25*g->DOMX/2;
	double r2 = 0.75*g->DOMX/2;
	double xc = g->DOMX / 2;
	double yc = g->DOMY / 2;
	double xcurrent = 0; double ycurrent = 0; double xd = 0; double yd = 0;
	double check = 0;

	printf("\n\nRadius 1 = %g and Radius 2 = %g\n", r1, r2);
	printf("Center position (x,y) = (%3g,%3g)\n", xc, yc);


	// Set the field and update coefficients for H1-H3
	for (m = 0; m < g->jm; m++) {
		for (n = 0; n < g->im; n++) {
			int offset = n + m * (g->im);
						
			//find current x and y position in domain
			xcurrent = n * 2.0 * g->hex_d;
			ycurrent = 0.75 * m * g->dx;
			xd = (xcurrent - xc);
			yd = (ycurrent - yc);
			
			check = sqrt(xd*xd + yd*yd);
			if (check < r1 || check > r2)
				g->cez[offset] = 0.0;
			else
				g->cez[offset] = g->dt / (e0 * g->hex_A);
			
			g->h1[offset] = 0.0;
			g->h2[offset] = 0.0;
			g->h3[offset] = 0.0;
			g->ch1[offset] = g->dt / (u0 * 2 * g->hex_d);
			g->ch2[offset] = g->dt / (u0 * 2 * g->hex_d);
			g->ch3[offset] = g->dt / (u0 * 2 * g->hex_d);
			g->ez[offset] = 0;
					
					
		}
	}

	
}