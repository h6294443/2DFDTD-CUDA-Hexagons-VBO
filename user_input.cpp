/*	This source file belongs to project "Ch8_8.4 (TMz Example)	*/

#include <stdio.h>
#include <stdlib.h>
#include "global.h"

void initialize_parameters(Grid g*) {
	//double src_f;
	//double Sc;
	//double X, Y; 
	//double N_lambda;
	
	// Step 1: Specify source frequency	
	printf("Enter source frequency in kHz: ");
	scanf_s(" %g", &src_f);
	
	// Step 2: Calculate wavelength
	const double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
	
	// Step 3: Specify Sc:
	printf("\n\nEnter desired Courant number (0.1-0.8): ");
	scanf_s(" %g", &Sc);
	if (Sc < 0.1) Sc = 0.1;
	if (Sc > 0.8) Sc = 0.8;
	
	// Step 4: Specify physical domain size in meters
	printf("\n\nEnter the domain width (X) in meters: ");
	scanf_s(" %g", &X);
	printf("\n\nEnter the domain height (Y) in meters: ");
	scanf_s(" %g", &Y);
	
	// Step 5: Specify desired points-per-wavelength N_lambda
	printf("\n\nEnter points-per-wavelength (can be a float): ");
	scanf_s(" %g", &N_lambda);
	
	// Step 6: Calculate dx (this may not be dx as defined)
	const double dx = lambda / N_lambda;				// This is the largest distance possible within one hexagon - from one point to the opposing pointy point
	
	// Step 7: Calculate dt
	const double dt = Sc*dx / c;
	
	// Step 8: Calculate M and N
	int M = (2 * X) / (sqrt(3.0) * dx);
	int N = (4 * Y) / (3.0 * dx);
	
	// Step 8: Specify source position (and soon, type)
	const int src_pos_x = (int)(0.15*M);
	const int src_pos_y = (int)(0.5*N);
	
	// Step 9: Specify desired slowdown, if any.
	const int slowdown = 25;

	// Step 1: Specify source frequency
	//const double src_f = 1.1e7;                 // Frequency of the source (for a sine or cosine)
	// Step 2: Calculate wavelength
	//const double lambda = c / src_f;           // Wavelength of the source (for a sine or cosine)
	// Step 3: Specify Sc:
	//double Sc = sqrt(2.f / 3.f) - 0.15;	// Force 0.1 - 0.8
	// Step 4: Specify physical domain size in meters
	//const double X = 30;
	//const double Y = 30;
	// Step 5: Specify desired points-per-wavelength N_lambda
	//const double N_lambda = 250;
	// Step 6: Calculate dx (this may not be dx as defined)
	//const double dx = lambda / N_lambda;				// This is the largest distance possible within one hexagon - from one point to the opposing pointy point
	// Step 7: Calculate dt
	//const double dt = Sc*dx / c;
	// Step 8: Calculate M and N
	//int M = (2 * X) / (sqrt(3.0) * dx);
	//int N = (4 * Y) / (3.0 * dx);
	// Step 8: Specify source position (and soon, type)
	//const int src_pos_x = (int)(0.15*M);
	//const int src_pos_y = (int)(0.5*N);
	// Step 9: Specify desired slowdown, if any.
	//const int slowdown = 25;




	return;
}

