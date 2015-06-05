/*
Implementation of "d2q9" Lattice Boltzmann, with Bhatnagar-Gross-Krook collision approximation.

Space is discretized on a lattice, and particle distribution functions ("velocities") discretized onto the links
between lattice points, that is, the up/down, left/right and diagonals. We denote these velocities as follows:

   f6  f2  f5
     \  |  /
   f3--f0--f1
     /  |  \
   f7  f4  f8

Define local macroscopic variables: fluid density rho = Sum f_i
                                    velocity u = 1/rho Sum f_i e_i   (ei are unit vectors along the links)

The particle distribution functions are updated each timestep:
   f_i(x+e_i*dt, t+dt) = f_i(x,t) - 1/tau [f_i(x,t) - feq_i(x,t)]

with (BGK) feq_i = omega_i rho(x) [1 + 3/2 e_i.u/c^2 + 9/2 (e_i.u)^2/c^4 - 3/2 u^2/c^2].

The domain can contain walls, and we can implement "bounce back" (no slip) or "reflect" (slip) boundary conditions.

Arrays indexed: (A(0,0) A(0,1) A(0,2) ...
                 A(1,0) A(1,1) A(1,2) ... ) ie, 2nd index runs along contiguous rows.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>		// memcpy
#include <time.h>			// clock_gettime
#include <immintrin.h>	// vector intrinsics
#define __USE_GNU
#include <fenv.h>			// feenableexcept
#include <CL/opencl.h>



// Choose precision and vectorization (here choose a serial header)
//#include "prec_double_avx.h"
//#include "prec_double_sse.h"
//#include "prec_double_serial.h"
//#include "prec_float_avx.h"
//#include "prec_float_sse.h"
//#include "prec_float_serial.h"


#include "runparams.h"


// OpenCL Stuff
int InitialiseCLEnvironment(cl_platform_id*, cl_device_id*, cl_context*, cl_command_queue*, cl_program*);
void CleanUpCLEnvironment(cl_platform_id*, cl_device_id*, cl_context*, cl_command_queue*, cl_program*);

#define MAXDEVICES 2
#define NQUEUES 1
char *kernelFileName = "src/latticeboltzmannkernels.cl";


// Function prototypes
real_t ComputeReynolds(
	const real_t * restrict f,
	const int * restrict walls);

void InitializeArrays(
	real_t * restrict f,
	int * restrict walls);

void PrintLattice(int timeStep, const real_t * restrict f);

double GetWallTime(void);



int main(void)
{
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

	// Allocate memory. Use _mm_malloc() for alignment. Want 32byte alignment for vector instructions.
	real_t * f;
	int * walls;

	int allocSize = NX * NYPADDED * NSPEEDS;
	printf("Lattice Size: %dx%d (%lf.2 MB)\n", NX, NY, (double)(allocSize*sizeof(*f))/1024.0/1024.0);
	f = _mm_malloc(allocSize * sizeof *f, 32);
	walls = _mm_malloc(NX * NYPADDED * sizeof *walls, 32);
	InitializeArrays(f, walls);


	// Set up OpenCL environment
	cl_platform_id		platform[2];
	cl_device_id		device_id[MAXDEVICES];
	cl_context			context;
	cl_command_queue	queue[NQUEUES];
	cl_program			program;
	cl_kernel			ApplySourceKernelA, ApplySourceKernelB, StreamCollideKernelA, StreamCollideKernelB;
	cl_int				err;
	cl_mem 				device_fA, device_fB, device_walls;

	if (InitialiseCLEnvironment(platform, device_id, &context, queue, &program) == EXIT_FAILURE) {
		printf("Error initialising OpenCL environment\n");
		return EXIT_FAILURE;
	}


	//set size of local and global work groups
	size_t globalSize[2], localSize[2];
	localSize[0] = 1;
	localSize[1] = 256;
	globalSize[0] = localSize[0]*((NX-1)/localSize[0])+localSize[0];
	globalSize[1] = localSize[1]*((NY-1)/localSize[1])+localSize[1];
	printf("---OpenCL: Using localSize: %dx%d, globalSize: %dx%d.\n", (int)localSize[0],(int)localSize[1], (int)globalSize[0],(int)globalSize[1]);


	// Create Kernels
	ApplySourceKernelA = clCreateKernel(program, "ApplySourceKernel", &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}
	ApplySourceKernelB = clCreateKernel(program, "ApplySourceKernel", &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}
	StreamCollideKernelA = clCreateKernel(program, "StreamCollideKernel", &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}
	StreamCollideKernelB = clCreateKernel(program, "StreamCollideKernel", &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}


	// Allocate device memory
	size_t sizeBytes = NX*NY*NSPEEDS * sizeof *f;
	device_fA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error allocating memory on device, line %d: %d.\n", __LINE__, err);
		return EXIT_FAILURE;
	}
	device_fB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error allocating memory on device, line %d: %d.\n", __LINE__, err);
		return EXIT_FAILURE;
	}
	sizeBytes = NX*NY * sizeof *walls;
	device_walls = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error allocating memory on device, line %d: %d.\n", __LINE__, err);
		return EXIT_FAILURE;
	}


	// Set kernel arguments
	err  = clSetKernelArg(ApplySourceKernelA, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(ApplySourceKernelA, 1, sizeof(cl_mem), &device_walls);

	err |= clSetKernelArg(ApplySourceKernelB, 0, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(ApplySourceKernelB, 1, sizeof(cl_mem), &device_walls);

	err |= clSetKernelArg(StreamCollideKernelA, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(StreamCollideKernelA, 1, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(StreamCollideKernelA, 2, sizeof(cl_mem), &device_walls);

	err |= clSetKernelArg(StreamCollideKernelB, 0, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(StreamCollideKernelB, 1, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(StreamCollideKernelB, 2, sizeof(cl_mem), &device_walls);
	if (err != CL_SUCCESS) {
		printf("Error in clSetKernelArg, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}


	// Start timing here. We include transfers to and from device in timing for comparison with CPU
	double timeElapsed = GetWallTime();
	err = 0;

	// Copy initial arrays to device:
	err |= clEnqueueWriteBuffer(queue[0], device_fA, CL_TRUE, 0, NX*NY*NSPEEDS*sizeof(real_t), f, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue[0], device_walls, CL_TRUE, 0, NX*NY*sizeof *walls, walls, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error in clEnqueueWriteBuffer, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}

	// Begin timesteps
	for (int n = 0; n < NTIMESTEPS; n+=2) {

		if (n % PRINTSTATSEVERY == 0) {
			clFinish(queue[0]);
			if (n != 0) {
				double complete = (double)n/(double)NTIMESTEPS;
				double secElap = GetWallTime()-timeElapsed;
				double secRem = secElap/complete*(1.0-complete);
				double avgbw = 4.0*n*sizeof(real_t)*NX*NY*NSPEEDS/(GetWallTime()-timeElapsed)/1024/1024/1024;
				printf("%5.2lf%%--Elapsed: %3dm%02ds, Remaining: %3dm%02ds. [Updates/s: %.3le, Update BW: ~%.3lf GB/s, GFLOPs: ~%.3lf]\n",
				       complete*100, (int)secElap/60, (int)secElap%60, (int)secRem/60, (int)secRem%60, n/(double)secElap,
				       avgbw, FLOPPERLATTICEPOINT*NX*NY*n/(double)secElap/1000.0/1000.0/1000.0);
			}
		}
		if (n % SAVELATTICEEVERY == 0) {
			//clEnqueueReadBuffer(queue[0], device_fA, CL_TRUE, 0, NX*NY*NSPEEDS*sizeof(real_t), f, 0, NULL, NULL);
			//PrintLattice(n, f);
		}

		// Do a timestep -- run kernels
		err |= clEnqueueNDRangeKernel(queue[0], ApplySourceKernelA, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error in clEnqueueNDRangeKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
		}
		err |= clEnqueueNDRangeKernel(queue[0], StreamCollideKernelA, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error in clEnqueueNDRangeKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
		}
		err |= clEnqueueNDRangeKernel(queue[0], ApplySourceKernelB, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error in clEnqueueNDRangeKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
		}
		err |= clEnqueueNDRangeKernel(queue[0], StreamCollideKernelB, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error in clEnqueueNDRangeKernel: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
		}
	}
	// Copy array back to host
	clEnqueueReadBuffer(queue[0], device_fA, CL_TRUE, 0, NX*NY*NSPEEDS*sizeof(real_t), f, 0, NULL, NULL);
	clFinish(queue[0]);
	timeElapsed = GetWallTime() - timeElapsed;
	// End iterations

	// print final run stats
	double avgbw = 4.0*NTIMESTEPS*sizeof(real_t)*NX*NY*NSPEEDS/timeElapsed/1024/1024/1024;
	printf("100.0%%--Elapsed: %3dm%02ds,                     [Updates/s: %.3le, Update BW: ~%.3lf GB/s, GFLOPs: ~%.3lf]\n",
	       (int)timeElapsed/60, (int)timeElapsed%60, NTIMESTEPS/timeElapsed, avgbw,
	       FLOPPERLATTICEPOINT*NX*NY*NTIMESTEPS/timeElapsed/1000.0/1000.0/1000.0);
	printf("Time: %lf Re %.10le\n", timeElapsed, ComputeReynolds(f, walls));


	// Free dynamically allocated memory
	_mm_free(f);
	_mm_free(walls);

	return EXIT_SUCCESS;
}



real_t ComputeReynolds(
	const real_t * restrict f,
	const int * restrict walls)
{
	// Compute reynolds number over central column
	int j = (int)((real_t)NY / 2.0);

	real_t u_yTotal = 0;
	int latticePoints = 0;

	for (int i = 0; i < NX; i++) {
		if (walls[I(i,j, 0)] == 0) {
			real_t density = 0.0;
			for (int s = 0; s < NSPEEDS; s++) {
				density += f[I(i,j, s)];
			}
			u_yTotal += (+(f[I(i,j, 5)]+f[I(i,j, 1)]+f[I(i,j, 8)])
			             -(f[I(i,j, 6)]+f[I(i,j, 3)]+f[I(i,j, 7)]))/density;
			latticePoints++;
		}
	}

	real_t viscosity = 1.0/3.0 * (TAU - 1.0/2.0);

	return u_yTotal/(real_t)latticePoints * 10 / viscosity;
}



void InitializeArrays(
	real_t * restrict f,
	int * restrict walls)
{
	const real_t initialf = INITIALDENSITY;

	// It is important that we initialize the arrays in parallel, with the same scheduling as the computation loops.
	// This allows memory to be allocated correctly for NUMA systems, with a "first touch" policy.

//#pragma omp parallel for default(none) shared(walls) schedule(static)
	// Add walls
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			walls[I(i,j, 0)] = 0;
		}
	}
	// barrier
	for (int i = 20; i < 220; i++) {
		walls[I(i,100, 0)] = 1;
		walls[I(i,101, 0)] = 1;
		walls[I(i,102, 0)] = 1;
		walls[I(i,103, 0)] = 1;
		walls[I(i,104, 0)] = 1;
	}
	// edges top and bottom
	for (int j = 0; j < NY; j++) {
		walls[I(0,j, 0)] = 1;
		walls[I(NX-1,j, 0)] = 1;
	}

//#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			f[I(i,j, 0)] = initialf * OMEGA0;
			f[I(i,j, 1)] = initialf * OMEGA14;
			f[I(i,j, 2)] = initialf * OMEGA14;
			f[I(i,j, 3)] = initialf * OMEGA14;
			f[I(i,j, 4)] = initialf * OMEGA14;
			f[I(i,j, 5)] = initialf * OMEGA58;
			f[I(i,j, 6)] = initialf * OMEGA58;
			f[I(i,j, 7)] = initialf * OMEGA58;
			f[I(i,j, 8)] = initialf * OMEGA58;
		}
	}


}



void PrintLattice(int timeStep, const real_t * restrict f)
{
	char filename[100];
	sprintf(filename,"data/%d.csv",timeStep);
	FILE *fp = fopen(filename,"w");

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {

			real_t density = 0;
			for (int s = 0; s < NSPEEDS; s++) {
				density += f[I(i,j, s)];
			}

			real_t u_x = (+(f[I(i,j, 6)]+f[I(i,j, 2)]+f[I(i,j, 5)])
			              -(f[I(i,j, 7)]+f[I(i,j, 4)]+f[I(i,j, 8)]))/density;
			real_t u_y = (+(f[I(i,j, 5)]+f[I(i,j, 1)]+f[I(i,j, 8)])
			              -(f[I(i,j, 6)]+f[I(i,j, 3)]+f[I(i,j, 7)]))/density;

			real_t uSquared = u_x * u_x + u_y * u_y;

			fprintf(fp,"%.10lf",uSquared);
			if (j < NY-1) fprintf(fp,", ");
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}



double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}


// OpenCL functions
int InitialiseCLEnvironment(cl_platform_id *platform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//error flag
	cl_int err;
	char infostring[1024];

	//get kernel from file
	FILE* kernelFile = fopen(kernelFileName, "rb");
	fseek(kernelFile, 0, SEEK_END);
	long fileLength = ftell(kernelFile);
	rewind(kernelFile);
	char *kernelSource = malloc(fileLength*sizeof(char));
	fread(kernelSource, fileLength, sizeof(char), kernelFile);
	fclose(kernelFile);

	//bind to platform
//	printf("Getting CL Platform...\n");
	err = clGetPlatformIDs(2, platform, NULL);
	if (err != CL_SUCCESS) {
		printf("Error in clGetPlatformIDs, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}
	clGetPlatformInfo(platform[0],CL_PLATFORM_VENDOR,sizeof(infostring),infostring,NULL);
	printf("---OpenCL: Platform Vendor: %s\n",infostring);

	//get device ID
//	printf("Getting CL Device IDs...\n");
	cl_uint totalDevices;
//	err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_CPU, MAXDEVICES, device_id, &totalDevices);
	err = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, MAXDEVICES, device_id, &totalDevices);
	if (err != CL_SUCCESS) {
		printf("Error in clGetDeviceIDs, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}
	char deviceName[50];
	for (int i = 0; i < (int) totalDevices; i++) {
		clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
		printf("---OpenCL: Device found: %d. %s\n", i, deviceName);
		cl_ulong maxAlloc;
		clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, NULL);
		printf("---OpenCL: CL_DEVICE_MAX_MEM_ALLOC_SIZE: %lu MB\n", maxAlloc/1024/1024);
	}

	//create a context
//	printf("Creating CL Context...\n");
	*context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateContext, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}

	//create a queue, CHOOSE DEVICE HERE
//	printf("Creating CL Queue...\n");
	*queue = clCreateCommandQueue(*context, device_id[0], 0, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateCommandQueue, line %d.\n", __LINE__);
		return EXIT_FAILURE;
	}

	//create the program with the source above
//	printf("Creating CL Program...\n");
	*program = clCreateProgramWithSource(*context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateProgramWithSource: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}

	//build program executable
//	printf("Building CL Executable...\n");
	err = clBuildProgram(*program, 0, NULL, "-I src/runparams.h", NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error in clBuildProgram: %d, line %d.\n", err, __LINE__);
		char buffer[2000];
		clGetProgramBuildInfo(*program, device_id[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	free(kernelSource);
	return EXIT_SUCCESS;
}


void CleanUpCLEnvironment(cl_platform_id *platform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//release CL resources
	clReleaseProgram(*program);
	clReleaseCommandQueue(*queue);
	clReleaseContext(*context);
}
