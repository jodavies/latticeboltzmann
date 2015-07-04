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
                                    velocity u = 1/rho Sum f_i e_i   (ei are vectors along the links)

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


#include "runparams.h"


// OpenCL Stuff
int InitialiseCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*);
void CleanUpCLEnvironment(cl_platform_id**, cl_device_id***, cl_context*, cl_command_queue*, cl_program*);
void CheckOpenCLError(cl_int err, int line);

#define NQUEUES 1
char *kernelFileName = "src/latticeboltzmannkernels.cl";


// Function prototypes
real_t ComputeReynolds(
	const real_t * restrict f,
	const int * restrict walls);

void InitializeArrays(
	real_t * restrict f,
	int * restrict walls);

void PrintRunStats(int n, double startTime);

void PrintLattice(int timeStep, const real_t * restrict f);
void PrintSquaredDensity(int timeStep, const real_t * restrict printArray);

double GetWallTime(void);



int main(void)
{
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
	setenv("CUDA_CACHE_DISABLE", "1", 1);

	// Allocate memory. Use _mm_malloc() for alignment. Want 32byte alignment for vector instructions.
	real_t * f;
	real_t * printArray;
	int * walls;

	int allocSize = NX * NYPADDED * NSPEEDS;
	printf("Lattice Size: %dx%d (%dx%d) (%lf.2 MB)\n", NX,NY, NX,NYPADDED, (double)(allocSize*sizeof(*f))/1024.0/1024.0);
	f = _mm_malloc(allocSize * sizeof *f, 32);

	allocSize = NX*NYPADDED * sizeof *walls;
	walls = _mm_malloc(allocSize, 32);
	InitializeArrays(f, walls);

	allocSize = NX*NYPADDED * sizeof *printArray;
	printArray = _mm_malloc(allocSize, 32);


	// Set up OpenCL environment
	cl_platform_id    *platform;
	cl_device_id      **device_id;
	cl_context        context;
	cl_command_queue  queue[NQUEUES];
	cl_program        program;
	cl_kernel         ApplySourceKernelA, ApplySourceKernelB, StreamCollideKernelA, StreamCollideKernelB;
	cl_kernel         ComputeSquaredDensityKernelA, ComputeSquaredDensityKernelB;
	cl_int            err;
	cl_mem            device_fA, device_fB, device_walls, device_printArray;

	if (InitialiseCLEnvironment(&platform, &device_id, &context, queue, &program) == EXIT_FAILURE) {
		printf("Error initialising OpenCL environment\n");
		return EXIT_FAILURE;
	}


	//set size of local and global work groups.
	size_t localSize = LOCALSIZE;
	size_t globalSize = localSize*((NX*NY-1)/localSize)+localSize;
	printf("---OpenCL: Using localSize: %d, globalSize: %d.\n", (int)localSize, (int)globalSize);
	size_t localSizeApplySource = LOCALSIZE;
	size_t globalSizeApplySource = localSizeApplySource*((NX-1)/localSizeApplySource)+localSizeApplySource;


	// Create Kernels
	ApplySourceKernelA = clCreateKernel(program, "ApplySourceKernel", &err);
	CheckOpenCLError(err, __LINE__);
	ApplySourceKernelB = clCreateKernel(program, "ApplySourceKernel", &err);
	CheckOpenCLError(err, __LINE__);
	StreamCollideKernelA = clCreateKernel(program, "StreamCollideKernel", &err);
	CheckOpenCLError(err, __LINE__);
	StreamCollideKernelB = clCreateKernel(program, "StreamCollideKernel", &err);
	CheckOpenCLError(err, __LINE__);
	ComputeSquaredDensityKernelA = clCreateKernel(program, "ComputeSquaredDensityKernel", &err);
	CheckOpenCLError(err, __LINE__);
	ComputeSquaredDensityKernelB = clCreateKernel(program, "ComputeSquaredDensityKernel", &err);
	CheckOpenCLError(err, __LINE__);


	// Allocate device memory, including array to store squared density for printing to file
	size_t sizeBytes = NX*NYPADDED*NSPEEDS * sizeof *f;
	device_fA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	CheckOpenCLError(err, __LINE__);
	device_fB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	CheckOpenCLError(err, __LINE__);

	sizeBytes = NX*NYPADDED * sizeof *walls;
	device_walls = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	CheckOpenCLError(err, __LINE__);

	sizeBytes = NX*NYPADDED * sizeof *f;
	device_printArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeBytes, NULL, &err);
	CheckOpenCLError(err, __LINE__);


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

	err |= clSetKernelArg(ComputeSquaredDensityKernelA, 0, sizeof(cl_mem), &device_fA);
	err |= clSetKernelArg(ComputeSquaredDensityKernelA, 1, sizeof(cl_mem), &device_printArray);

	err |= clSetKernelArg(ComputeSquaredDensityKernelB, 0, sizeof(cl_mem), &device_fB);
	err |= clSetKernelArg(ComputeSquaredDensityKernelB, 1, sizeof(cl_mem), &device_printArray);
	CheckOpenCLError(err, __LINE__);


	// Start timing here. We include transfers to and from device in timing for comparison with CPU
	double startTime = GetWallTime();
	err = 0;

	// Copy initial arrays to device:
	err |= clEnqueueWriteBuffer(queue[0], device_fA, CL_TRUE, 0, NX*NYPADDED*NSPEEDS*sizeof *f , f, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue[0], device_walls, CL_TRUE, 0, NX*NYPADDED*sizeof *walls, walls, 0, NULL, NULL);
	CheckOpenCLError(err, __LINE__);

	// Begin timesteps. Open OpenMP parallel region so we can write data to disk in the background
	#pragma omp parallel
	{
		// Only one thread enqueues kernels
		#pragma omp single
		{
			for (int n = 0; n < NTIMESTEPS; n+=2) {

				if (n % PRINTSTATSEVERY == 0) {
					clFinish(queue[0]);
					if (n != 0) {
						PrintRunStats(n, startTime);
					}
				}

#if SAVELATTICE == 1
				if (n % SAVELATTICEEVERY == 0) {
					// compute squared density on device
					err = clEnqueueNDRangeKernel(queue[0], ComputeSquaredDensityKernelA, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
					clFinish(queue[0]);

					// wait for completion of previous PrintLattice. The clEnqueueReadBuffer we make next is NON BLOCKING, but
					// the barrier here guarantees its completion, since the thread waits on the completion of the read before
					// printing the array to file.
					#pragma omp taskwait
					// copy most recent timestep from device, NON-BLOCKING
					cl_event readBufferComplete;
					clEnqueueReadBuffer(queue[0], device_printArray, CL_FALSE, 0, NX*NYPADDED*sizeof(real_t), printArray, 0, NULL, &readBufferComplete);

					// another thread completes the write to disk in the background
					#pragma omp task
					{
						clWaitForEvents(1, &readBufferComplete);
						PrintSquaredDensity(n, printArray);
					}
				}
#endif

				// Do a timestep -- run kernels
				err = clEnqueueNDRangeKernel(queue[0], ApplySourceKernelA, 1, NULL, &globalSizeApplySource, &localSizeApplySource, 0, NULL, NULL);
				err = clEnqueueNDRangeKernel(queue[0], StreamCollideKernelA, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
				err = clEnqueueNDRangeKernel(queue[0], ApplySourceKernelB, 1, NULL, &globalSizeApplySource, &localSizeApplySource, 0, NULL, NULL);
				err = clEnqueueNDRangeKernel(queue[0], StreamCollideKernelB, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

			} // end iterations
		} // end omp single
	} // end omp parallel


	// Copy array back to host
	clFinish(queue[0]);
	err = clEnqueueReadBuffer(queue[0], device_fA, CL_TRUE, 0, NX*NYPADDED*NSPEEDS*sizeof(real_t), f, 0, NULL, NULL);
	CheckOpenCLError(err, __LINE__);

	// Complete queue and stop timer
	clFinish(queue[0]);
	double timeElapsed = GetWallTime() - startTime;


	// print final run stats
	PrintRunStats(NTIMESTEPS, startTime);
	printf("Runtime: %lf Re %.10le\n", timeElapsed, ComputeReynolds(f, walls));


	CleanUpCLEnvironment(&platform, &device_id, &context, queue, &program);
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

	// Add walls, initialize to zero
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			walls[I(i,j, 0)] = 0;
		}
	}
	// edges top and bottom
	for (int j = 0; j < NY; j++) {
		walls[I(0,j, 0)] = 1;
		walls[I(NX-1,j, 0)] = 1;
	}


	// barrier
	for (int i = 99; i < 300; i++) {
		walls[I(i,100, 0)] = 1;
		walls[I(i,101, 0)] = 1;
		walls[I(i,102, 0)] = 1;
		walls[I(i,103, 0)] = 1;
		walls[I(i,104, 0)] = 1;
	}


	// initialize lattice
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			f[I(i,j, 0)] = INITIALDENSITY * OMEGA0;
			f[I(i,j, 1)] = INITIALDENSITY * OMEGA14;
			f[I(i,j, 2)] = INITIALDENSITY * OMEGA14;
			f[I(i,j, 3)] = INITIALDENSITY * OMEGA14;
			f[I(i,j, 4)] = INITIALDENSITY * OMEGA14;
			f[I(i,j, 5)] = INITIALDENSITY * OMEGA58;
			f[I(i,j, 6)] = INITIALDENSITY * OMEGA58;
			f[I(i,j, 7)] = INITIALDENSITY * OMEGA58;
			f[I(i,j, 8)] = INITIALDENSITY * OMEGA58;
		}
	}


}



void PrintLattice(int timeStep, const real_t * restrict f)
{
	char filename[100];
	sprintf(filename,"/mnt/Documents/tmp-data2/%d.csv",timeStep);
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



void PrintSquaredDensity(int timeStep, const real_t * restrict printArray)
{
	char filename[100];
	sprintf(filename,"/mnt/Documents/tmp-data2/%d.csv",timeStep);
	FILE *fp = fopen(filename,"w");

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {

			fprintf(fp,"%.10lf", printArray[I(i,j, 0)]);
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



void PrintRunStats(int n, double startTime)
{
	double complete = (double)n/(double)NTIMESTEPS;
	double timeElap = GetWallTime()-startTime;
	double timeRem = timeElap/complete*(1.0-complete);
	double avgbw = (2.0*n*sizeof(real_t)*NX*NY*NSPEEDS + 2.0*n*sizeof(real_t)*NX*6 + sizeof(int)*NX*NY)
	               /timeElap/1024.0/1024.0/1024.0;
	printf("%5.2lf%%--Elapsed: %3dm%02ds, Remaining: %3dm%02ds. [Updates/s: %.3le, Update BW: ~%.3lf GB/s, GFLOPs: ~%.3lf]\n",
	       complete*100, (int)timeElap/60, (int)timeElap%60, (int)timeRem/60, (int)timeRem%60, n/timeElap,
	       avgbw, FLOPPERLATTICEPOINT*NX*NY*n/timeElap/1000.0/1000.0/1000.0);
}



// OpenCL functions
int InitialiseCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//error flag
	cl_int err;
	char infostring[1024];

	//get kernel from file
	FILE* kernelFile = fopen(kernelFileName, "rb");
	fseek(kernelFile, 0, SEEK_END);
	long fileLength = ftell(kernelFile);
	rewind(kernelFile);
	// allocate space for file + terminating \0
	char *kernelSource = malloc((fileLength+1)*sizeof(char));
	long read = fread(kernelSource, sizeof(char), fileLength, kernelFile);
	kernelSource[fileLength] = '\0';
	if (fileLength != read) printf("Error reading kernel file, line %d\n", __LINE__);
	fclose(kernelFile);

	//get platform and device information
	cl_uint numPlatforms;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	*platform = malloc(numPlatforms * sizeof(cl_platform_id));
	*device_id = malloc(numPlatforms * sizeof(cl_device_id*));
	err |= clGetPlatformIDs(numPlatforms, *platform, NULL);
	CheckOpenCLError(err, __LINE__);

	for (int i = 0; i < numPlatforms; i++) {
		clGetPlatformInfo((*platform)[i], CL_PLATFORM_VENDOR, sizeof(infostring), infostring, NULL);
		printf("\n---OpenCL: Platform Vendor %d: %s\n", i, infostring);

		cl_uint numDevices;
		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		CheckOpenCLError(err, __LINE__);
		(*device_id)[i] = malloc(numDevices * sizeof(cl_device_id));
		err = clGetDeviceIDs((*platform)[i], CL_DEVICE_TYPE_ALL, numDevices, (*device_id)[i], NULL);
		CheckOpenCLError(err, __LINE__);
		for (int j = 0; j < numDevices; j++) {
			char deviceName[200];
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
			printf("---OpenCL:    Device found %d. %s\n", j, deviceName);
			cl_ulong maxAlloc;
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlloc), &maxAlloc, NULL);
			printf("---OpenCL:       CL_DEVICE_MAX_MEM_ALLOC_SIZE: %lu MB\n", maxAlloc/1024/1024);
			cl_uint cacheLineSize;
			clGetDeviceInfo((*device_id)[i][j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cacheLineSize), &cacheLineSize, NULL);
			printf("---OpenCL:       CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %u B\n", cacheLineSize);
		}
	}

	// HERE WE NEED TO HAVE SPECIFIED A PLATFORM AND DEVICE IN runparams.h
	printf("\n---OpenCL: Using platform %d, device %d\n", PLATFORM, DEVICE);
	//create a context
	*context = clCreateContext(NULL, 1, &((*device_id)[PLATFORM][DEVICE]), NULL, NULL, &err);
	CheckOpenCLError(err, __LINE__);
	//create a queue
	*queue = clCreateCommandQueue(*context, (*device_id)[PLATFORM][DEVICE], 0, &err);
	CheckOpenCLError(err, __LINE__);

	//create the program with the source above
//	printf("Creating CL Program...\n");
	*program = clCreateProgramWithSource(*context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error in clCreateProgramWithSource: %d, line %d.\n", err, __LINE__);
		return EXIT_FAILURE;
	}

	//build program executable
//	printf("Building CL Executable...\n");
#if DOUBLEPREC == 1
	err = clBuildProgram(*program, 0, NULL, "-I. -I src", NULL, NULL);
#else
	err = clBuildProgram(*program, 0, NULL, "-I. -I src -cl-single-precision-constant", NULL, NULL);
#endif
	if (err != CL_SUCCESS) {
		printf("Error in clBuildProgram: %d, line %d.\n", err, __LINE__);
		char buffer[5000];
		clGetProgramBuildInfo(*program, (*device_id)[PLATFORM][DEVICE], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	// dump ptx
	size_t binSize;
	clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
	unsigned char *bin = malloc(binSize);
	clGetProgramInfo(*program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
	FILE *fp = fopen("openclPTX.ptx", "wb");
	fwrite(bin, sizeof(char), binSize, fp);
	fclose(fp);
	free(bin);

	free(kernelSource);
	return EXIT_SUCCESS;
}



void CleanUpCLEnvironment(cl_platform_id **platform, cl_device_id ***device_id, cl_context *context, cl_command_queue *queue, cl_program *program)
{
	//release CL resources
	clReleaseProgram(*program);
	clReleaseCommandQueue(*queue);
	clReleaseContext(*context);

	cl_uint numPlatforms;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	for (int i = 0; i < numPlatforms; i++) {
		free((*device_id)[i]);
	}
	free(*platform);
	free(*device_id);
}



void CheckOpenCLError(cl_int err, int line)
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error %d, line %d\n", err, line);
	}
}
