// ____________________________________________________________________	//
// HEXCUDA V2.00 uses the same program stucture as CUDA V2.00 and 2.01. //
// This is necessary to conform to the glutMainLoop program structure.  //
// ____________________________________________________________________	//
// Instead of a pixel-buffer object, HEXCUDA V2.00 generates a vertex-  //
// buffer object that contains the 3D vertices plus color of each grid  //
// point.  The spatial parts do not change during the simulation, only  //
// some vertex colors do (as they represent Ez field intensity).        //
// ____________________________________________________________________	//
// Consequently, the spatial data can be generated only once at the     //
// beginning of the simulation with a dedicated kernel call to initia-  //
// lize the vertex buffer.  After that, it can be passed to a modified  //
// createImageOnGpu function & kernel to update the colors of only those//
// vertices that correspond to Ez field values (the centers of each     //
// hexagon).
// ____________________________________________________________________	//
// After that, the vbo is displayed and swapped to the front buffer.    //
// ____________________________________________________________________	//
/* Matt Engels, October 5, 2015*/

// This file belongs to HEXCUDA V2.00

#include "global.h"
#include <Windows.h>
#include "cuda_functions.h"
#include "graphics.h"
#include <time.h>


int main(int argc, char** argv)
{
	
	gridInit(g);							// Initialize the grid
	initializeHexGlobalDevicePointers();	// Initialize all global dev pointers to zero
	runFdtdWithFieldDisplay(argc, argv);	// Main function that has the call to glutMainLoop()

	
}

