#include "graphics.h"
//HEXCUDA V2.00
unsigned int window_width = 1080;
unsigned int window_height = 1080;
unsigned int image_width = g->im;// +1;
unsigned int image_height = g->jm;// 3 * N + 2;
int drawMode = GL_TRIANGLE_FAN;
int iGLUTWindowHandle = 0;          // handle to the GLUT window
float global_min_field = 1e9;
float global_max_field = -1e9;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;

bool pause_flag = true;	// used for halting field updates (freeze)

void setImageAndWindowSize()
{
	image_width = g->im;
	image_height = g->jm;

	if (g->im > g->jm)
		window_height = window_width*g->jm / g->im;
	else
		window_width = window_height*g->im / g->jm;
}
void idle()
{
	glutPostRedisplay();
}
void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case(27) :
		exit(0);
		break;
	case 'p' :
		pause_flag = !pause_flag;
		break;
	case 'd':
	case 'D':
		switch (drawMode) {
		case GL_POINTS: drawMode = GL_LINE_STRIP; break;
		case GL_LINE_STRIP: drawMode = GL_TRIANGLE_FAN; break;
		default: drawMode = GL_POINTS;
		} break;
	}
	glutPostRedisplay();
}
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}
void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	}
	else if (mouse_buttons & 4) {
		translate_z += dy * 0.01;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}
void reshape(int w, int h)
{
	window_width = w;
	window_height = h;
}

void createVBO(mappedBuffer_t* mbuf)	//void createVBO(GLuint* vbo, unsigned int typeSize)
{
	// create buffer object
	glGenBuffers(1, &(mbuf->vbo));
	glBindBuffer(GL_ARRAY_BUFFER, mbuf->vbo);

	// initialize buffer object
	unsigned int size = (g->im+1) * (3*g->jm + 2) * mbuf->typeSize;		// size is (M+1)(3N+2) for full, 
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);	// padded hex grid
	glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef USE_CUDA3
	cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource), mbuf->vbo,
		cudaGraphicsMapFlagsNone);
#else
	// register buffer object with CUDA
	cudaGLRegisterBufferObject(mbuf->vbo);
#endif
}

void deleteVBO(mappedBuffer_t* mbuf)	
{
	glBindBuffer(1, mbuf->vbo);
	glDeleteBuffers(1, &(mbuf->vbo));

#ifdef USE_CUDA3
	cudaGraphicsUnregisterResource(mbuf->cudaResource);
	mbuf->cudaResource = NULL;
	mbuf->vbo = NULL;
#else
	cudaGLUnregisterBufferObject(mbuf->vbo);
	mbuf->vbo = NULL;
#endif

}
void initCuda()
{
	pickGPU(1);
	createVBO(&vertexVBO);
	createVBO(&colorVBO);		
}

void renderCuda_1D(int drawMode)	// 1D version for 1D grid point kernel
{
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
	
	int M_hex = g->im + 1;
	int N_hex = 3 * g->jm + 2;
	int TILE_SQUARED = TILE_SIZE * TILE_SIZE;				// Just the TILE_SIZE squared for a 1-D kernel
	int Bx1 = (TILE_SQUARED - 1 + M_hex*N_hex) / TILE_SQUARED;
	int size_padded = TILE_SQUARED * Bx1;							// Total array size for the 1-D case
	int size_true = M_hex*N_hex;

	switch (drawMode) {
	case GL_LINE_STRIP:
		for (int i = 0; i < ((g->im + 1)*(3 * g->jm + 2)); i += g->im)
			glDrawArrays(GL_LINE_STRIP, i, size_true);// ((M + 1)*(3 * N + 2)));
		break;
	case GL_TRIANGLE_FAN: {
		static GLuint* qIndices = NULL;
		int size = 9 * g->nCells;							// M * N Ez points.  Then 9 points per Ez point
														// 9 pts = 1 center, 6 periphery, 1 repeat, 1 restart
		if (qIndices == NULL) {							// allocate and assign trianglefan indicies 
			qIndices = (GLuint *)malloc(size*sizeof(GLuint));
			int index = 0;								// Used for offsets
			
			/* Below is the for-loop construct for setting up the index array for the hexagon VBO */
			int mod = 0;
			int k = 0;								// modifier that gets added to center, #3, #6 depending on row
			for (int j = 2; j < N_hex - 1; j = j + 3) {	// Start at 3rd row, step only into Ez rows
				mod = (j + 1) / 3;					// Calculate the integer division result
				k = 1 - mod % 2;					// Calculate the modulus (remainder) - even/odd row of j in multiples of 3

				for (int i = 0; i < g->im; i++) {		// Steps through the columns in the Ez rows
					qIndices[index++] = j*M_hex + i + k;			// Center of hexagon, this is Ez(i,j)
					qIndices[index++] = (j + 1)*M_hex + (i + 1);	// Point #2
					qIndices[index++] = (j + 2)*M_hex + i + k;		// Point #3
					qIndices[index++] = (j + 1)*M_hex + i;			// Point #4
					qIndices[index++] = (j - 1)*M_hex + i;			// Point #5
					qIndices[index++] = (j - 2)*M_hex + i + k;		// Point #6
					qIndices[index++] = (j - 1)*M_hex + (i + 1);	// Point #7
					qIndices[index++] = (j + 1)*M_hex + (i + 1);	// Point #8 (repeat, closes it out)
					qIndices[index++] = RestartIndex;				// Restart primitive
				}
			}
			//for (int z = 0; z < size; z++)
			//	printf("qIndices[%i] = %i\n", z, qIndices[z]);
			//printf("Sizeof(qIndices) = %i", sizeof(qIndices));
		}
		glPrimitiveRestartIndexNV(RestartIndex);
		glEnableClientState(GL_PRIMITIVE_RESTART_NV);
		glDrawElements(GL_TRIANGLE_FAN, size, GL_UNSIGNED_INT, qIndices);
		glDisableClientState(GL_PRIMITIVE_RESTART_NV);
	} break;
	default:
		glDrawArrays(GL_POINTS, 0, size_true);
		break;
	}
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

void Cleanup(int iExitCode)
{
	deleteVBO(&vertexVBO);
	deleteVBO(&colorVBO);
	cudaThreadExit();
	if (iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
	exit(iExitCode);
}

bool runFdtdWithFieldDisplay(int argc, char** argv)	// This is the primary function main() calls 
{													// and it start the glutMainLoop() which calls the display() function
	initGL(argc, argv);
	initCuda();			
	glutDisplayFunc(runIterationsAndDisplay);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);							// picks a GPU, creates both vertex buffers
	//create_Grid_points_only(dptr);

	if (int ret = copyHexTMzArraysToDevice() != 0){	// copy data from CPU RAM to GPU global memory
		if (ret == 1) printf("Memory allocation error in copyTMzArraysToDevice(). \n\n Exiting.\n");
		return 0;
	}
	glutMainLoop();									// This starts the glutMainLoop 
	Cleanup(EXIT_FAILURE);
}

void runIterationsAndDisplay()						// This is the glut display function.  It is called once each
{													// glutMainLoop() iteration.
	if (pause_flag) {
		if (g->time < g->maxTime)
			update_all_fields_hex_CUDA();	// was fdtdIternationsOnGpu()
		else {
			copyHexFieldSnapshotsFromDevice();
			deallocateHexCudaArrays();
			Cleanup(EXIT_SUCCESS);
		}
	}

	uint drawStartTime = clock();	// used in the fps control

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// The following block maps two cudaGraphicsResources, the color and vertex vbo's
	// and gets pointers to them.
	size_t start;
	checkCudaErrors(cudaGraphicsMapResources(1, &vertexVBO.cudaResource, NULL));	// this vertex block needs to go to the create_grid_points function.  It gets called only once
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &start,	// as this buffer does not change once initialized.
		vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &colorVBO.cudaResource, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cptr, &start, colorVBO.cudaResource));
			
	create_Grid_points_only(dptr,cptr);
	createImageOnGpu();		// calls min-max and create_image kernels
	
	// unmap the GL buffer 
	checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource, NULL));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &colorVBO.cudaResource, NULL));

	renderCuda_1D(drawMode);		// Render the graphics to the back buffer
	cudaThreadSynchronize();
	glutSwapBuffers();				// Swap the front and back buffers
	glutPostRedisplay();
	Sleep(slowdown);
		
	//for (int k = 0; k < 8; k++){
	//	//printf("qIndices[%i]:%i\n", k, qIndices[k]);
	//	printf("dPtr[%i] u = %f and v = %f", k, dptr[k], dptr[k]);
	//}

}


/*bool saveSampledFieldsToFile()
{

	/*strcpy(output_file_name, "result_");
	strncat(output_file_name, input_file_name, strlen(input_file_name));

	ofstream output_file;
	output_file.open(output_file_name, ios::out | ios::binary);

	if (!output_file.is_open())
	{
		cout << "File <" << output_file_name << "> can not be opened! \n";
		return false;
	}
	else
	{
		output_file.write((char*)sampled_electric_fields_sampled_value, number_of_sampled_electric_fields*sizeof(float)*number_of_time_steps);
		output_file.write((char*)sampled_magnetic_fields_sampled_value, number_of_sampled_magnetic_fields*sizeof(float)*number_of_time_steps);
		free(sampled_electric_fields_sampled_value);
		free(sampled_magnetic_fields_sampled_value);
	}

	output_file.close();
	return true;
}*/

void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("2D HEX CUDA");
	glutDisplayFunc(runIterationsAndDisplay);		// runIterationAndDisplay is what glutMainLoop() will keep running.
	glutKeyboardFunc(keyboard);						// So it has to contain the FDTD time iteration loop
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mouse);
	glutMotionFunc(motion);
	//glutIdleFunc(idle);

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// projection
	GLfloat aspect_ratio = (GLfloat)window_width / (GLfloat)window_height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, aspect_ratio, 0.5, 45.0);

	//glMatrixMode(GL_PROJECTION);
	//glOrtho(-1, 1, -1, 1, -1, 1);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity;
}

void drawText(float x, float y, void *font, char *string) {
	char *c;
	glRasterPos2f(x, y);

	for (c = string; *c != ' '; c++) {
		glutBitmapCharacter(font, *c);
	}
}
