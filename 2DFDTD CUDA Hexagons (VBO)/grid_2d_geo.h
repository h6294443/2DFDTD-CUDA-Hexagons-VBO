/*---------------------------------------------------------------------------*/
// Implements a 2D Geodesic TMz grid										 //
/*---------------------------------------------------------------------------*/
// This file belongs to HEXCUDA V2.00

#ifndef _GRID_2D_GEO_H
#define _GRID_2D_GEO_H

enum GRIDTYPE { oneDGrid, teZGrid, tmZGrid, twoDGeoTMz };

struct Grid {
	double *h1, *h2, *h3;		// Double-precision pointers for the field update equations
	double *ez, *cez;			// See previous line
	float *ez_float;			// Single-precision e-field.
	int im, jm, nCells;			// x-dim, y-dim, number of field pointers  (any one field)
	double *ch1, *ch2, *ch3;	// this is the magnetic field update coefficient.  Not space dependent in this simulation
	double hex_max;				// maximum distance aross hexagon (vertice to opposing vertice)
	double hex_t;				// length of equilateral triangle, six of which make up the hexagon
	double hex_d;				// flat-to-flat distance of the hexagon. 2*hex_d is the distance between
								// adjacent Ez points
	double hex_A;				// area of the hexagon
	int time, maxTime;
	int type;
	double cdtds;
};

typedef struct Grid Grid;
void gridInit(Grid *g);


#endif