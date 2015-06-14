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
#include <math.h>
#include <assert.h>
#include <mpi.h>


// d2q9 fixed parameters
#define NSPEEDS 9
#define OMEGA0  (4.0/9.0)
#define OMEGA14 (1.0/9.0)
#define OMEGA58 (1.0/36.0)

// Boundary condition. Wrap-around (periodic) or not (fluid flows out of the domain)
#define WRAPAROUND 1

// variable parameters
#define NX 400
#define NY 2000
#define TAU 0.7
#define CSQ (1.0)

#define NTIMESTEPS 10000
#define PRINTSTATSEVERY 1000
#define SAVELATTICE 0
#define SAVELATTICEEVERY 100000
#define ACCEL 0.005
#define INITIALDENSITY 0.1

// MPI
#define FROMMASTER 0
#define IFMASTER if(myRank==0)
#define IFRANK(r) if(myRank==(r))

// Choose precision and vectorization
//#include "prec_double_avx.h"
//#include "prec_double_sse.h"
//#include "prec_double_serial.h"
#include "prec_float_avx.h"
//#include "prec_float_sse.h"
//#include "prec_float_serial.h"



// Macro for array indexing. We need the array stride to be such that we have correct
// alignment for vector loads/stores on rows after the first.
// Additionally, for the MPI version we have two different lattice sizes: the master contains the full lattice with
// NX rows, and ranks all contain just a share of this. We therefore need an indexing macro that takes a rows argument.
#define NYPADDED (ALIGNREQUIREMENT*((NY-1)/ALIGNREQUIREMENT)+ALIGNREQUIREMENT)
#define I(rows, i,j, speed) ((speed)*(rows)*NYPADDED + (i)*NYPADDED + (j))
#define HALOS 2

// Also store the multiple of VECWIDTH below NY, since the vectorized functions need to terminate here,
// possibly with a scalar function cleaning up the "extra".
#define NYVECMAX (VECWIDTH*(NY/VECWIDTH))

// For approximate GFLOPs report: we do ~124 FLOP per lattice point, obtained simply by counting ADD,SUB,MUL,DIV
// intructions in CollideVec.
#define FLOPPERLATTICEPOINT (124.0)



// Function prototypes
void DoTimeStep(
	real_t * restrict fA,
	real_t * restrict fB,
	const char * restrict walls,
	const int * restrict rowBoundaries);

void ApplySource(
	real_t * restrict fAB,
	const char * restrict walls,
	const int * restrict rowBoundaries);

void StreamCollide(
	const int iMin,
	const int iMax,
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls,
	const int * restrict rowBoundaries);

#if defined(AVX) || defined(SSE)
void StreamCollideVec(
	const int iMin,
	const int iMax,
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls,
	const int * restrict rowBoundaries);
#endif

real_t ComputeReynolds(
	const real_t * restrict fM,
	const char * restrict walls);

void InitializeArrays(
	real_t * restrict fM,
	char * restrict walls);

void MPIDistributeInitialConditions(
	real_t * restrict fM,
	real_t * restrict fA,
	char * restrict walls,
	const int * restrict rowBoundaries);

void MPIMasterReceiveLattice(
	real_t * restrict fM,
	real_t * restrict fAB,
	const int * restrict rowBoundaries);

void MPIStartHaloExchange(
	real_t * restrict fAB,
	const int * restrict rowBoundaries,
	MPI_Request * restrict req);

void PrintRunStats(int n, double startTime);

void PrintLattice(int timeStep, const real_t * restrict fM);

double GetWallTime(void);



int main(int argc, char** argv)
{
	int totalRanks, myRank;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
	IFMASTER printf("---MPI: Running with %d ranks\n", totalRanks);

	//feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

	// Compute each rank's share of the lattice. We split along the (shorter) x-direction, since vectorization prefers
	// the longer rows. The first "remainder" ranks get an extra row
	int remainder = NX%totalRanks;
	assert(remainder < totalRanks);
	IFMASTER printf("rem = %d\n", remainder);
	int *rowBoundaries;
	rowBoundaries = malloc((totalRanks+1) * sizeof *rowBoundaries);
	rowBoundaries[0] = 0;
	for (int r = 1; r <= remainder; r++) {
		rowBoundaries[r] = rowBoundaries[r-1] + NX/totalRanks +1;
	}
	for (int r = remainder+1; r < totalRanks; r++) {
		rowBoundaries[r] = rowBoundaries[r-1] + NX/totalRanks;
	}
	rowBoundaries[totalRanks] = NX;

//	IFMASTER {
//		for (int r = 0; r < totalRanks; r++) {
//			printf("rank %d has %d rows (%d,%d)\n",r,rowBoundaries[r+1]-rowBoundaries[r],rowBoundaries[r],rowBoundaries[r+1]);
//		}
//	}


	// Allocate memory. Use _mm_malloc() for alignment. Want 32byte alignment for vector instructions.
	// fM is the master thread's array for initialization and collection of results.
	real_t * fM = NULL;
	real_t * fA, * fB; 
	char * walls;

	// Master allocates an fM large enough for entire lattice.
	// fA and fB are the computation arrays, which contain approx. NX/totalRanks rows. These contain an additional
	// two rows for halo data.
	// The walls array is small, just allocate a full array on all ranks, and give it halo rows to make indexing easier.
	// Ranks will store their walls data at the BEGINNING of this array, ie, starting at row 1.
	int allocSizefM = NX * NYPADDED * NSPEEDS * sizeof *fM;
	int allocSizefAB = (rowBoundaries[myRank+1]-rowBoundaries[myRank] + HALOS) * NYPADDED * NSPEEDS * sizeof *fA;
	int allocSizewalls = (NX + HALOS) * NYPADDED * sizeof *walls;

	IFMASTER printf("Lattice Size: %dx%d (%lf.2 MB)\n", NX, NY, (double)allocSizefM/1024.0/1024.0);
	IFMASTER {
		fM = _mm_malloc(allocSizefM, 32);
	}
	fA = _mm_malloc(allocSizefAB, 32);
	fB = _mm_malloc(allocSizefAB, 32);
	walls = _mm_malloc(allocSizewalls, 32);

	// Master thread initializes fM and walls
	IFMASTER InitializeArrays(fM, walls);

	// Master thread sends initial conditions stored in fM and walls to rank fA
	MPIDistributeInitialConditions(fM, fA, walls, rowBoundaries);
	IFMASTER printf("Initial conditions sent to ranks.\n");


	// Begin iterations
	double startTime = GetWallTime();

	for (int n = 0; n < NTIMESTEPS; n+=2) {
		IFMASTER {
			if (n % PRINTSTATSEVERY == 0) {
				if (n != 0) {
					PrintRunStats(n, startTime);
				}
			}
		}

#if SAVELATTICE == 1
		if (n % SAVELATTICEEVERY == 0) {
			MPIMasterReceiveLattice(fM, fB, rowBoundaries);
			IFMASTER PrintLattice(n, fM);
		}
#endif

		// Do a timestep
		DoTimeStep(fA, fB, walls, rowBoundaries);

	}
	// End iterations

	// Collect final data on master thread. Latest data is contained in fB.
	MPIMasterReceiveLattice(fM, fB, rowBoundaries);

	double timeElapsed = GetWallTime() - startTime;

	// master prints final run stats
	IFMASTER {
		PrintRunStats(NTIMESTEPS, startTime);
		printf("Runtime: %lf Re %.10le\n", timeElapsed, ComputeReynolds(fM, walls));
	}


	// Free dynamically allocated memory
	free(rowBoundaries);
	IFMASTER _mm_free(fM);
	_mm_free(fA);
	_mm_free(fB);
	_mm_free(walls);

	MPI_Finalize();

	return EXIT_SUCCESS;
}



void DoTimeStep(
	real_t * restrict fA,
	real_t * restrict fB,
	const char * restrict walls,
	const int * restrict rowBoundaries)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	// "inner" rows don't border a halo.
	int lowerInnerRow = 2;
	int upperInnerRow = rowBoundaries[myRank+1]-rowBoundaries[myRank]-1;


	ApplySource(fA, walls, rowBoundaries);

	// Start a non-blocking exchange of haloes.
	MPI_Request reqA[12];
	MPIStartHaloExchange(fA, rowBoundaries, reqA);

	// While the transfer completes, we can StreamCollide for the inner rows, that don't border a halo.
#if defined(AVX) || defined(SSE)
	StreamCollideVec(lowerInnerRow, upperInnerRow, 0,NYVECMAX,fA, fB, walls, rowBoundaries);
	StreamCollide(lowerInnerRow, upperInnerRow, NYVECMAX, NY, fA, fB, walls, rowBoundaries);
#else
	StreamCollide(lowerInnerRow, upperInnerRow, 0, NY, fA, fB, walls, rowBoundaries);
#endif

	// Wait for "up" transfer, first 6 requests. Then we can compute the lower row.
	MPI_Waitall(6, reqA, MPI_STATUSES_IGNORE);

	// StreamCollide for lower row bordering halo
#if defined(AVX) || defined(SSE)
	StreamCollideVec(lowerInnerRow-1, lowerInnerRow-1, 0,NYVECMAX,fA, fB, walls, rowBoundaries);
	StreamCollide(lowerInnerRow-1, lowerInnerRow-1, NYVECMAX, NY, fA, fB, walls, rowBoundaries);
#else
	StreamCollide(lowerInnerRow-1, lowerInnerRow-1, 0, NY, fA, fB, walls, rowBoundaries);
#endif

	// Wait for "down" transfer, second 6 requests. Then we can compute the upper row.
	MPI_Waitall(6, &reqA[6], MPI_STATUSES_IGNORE);

	// StreamCollide for lower row bordering halo
#if defined(AVX) || defined(SSE)
	StreamCollideVec(upperInnerRow+1, upperInnerRow+1, 0,NYVECMAX,fA, fB, walls, rowBoundaries);
	StreamCollide(upperInnerRow+1, upperInnerRow+1, NYVECMAX, NY, fA, fB, walls, rowBoundaries);
#else
	StreamCollide(upperInnerRow+1, upperInnerRow+1, 0, NY, fA, fB, walls, rowBoundaries);
#endif




	ApplySource(fB, walls, rowBoundaries);

	// Start a non-blocking exchange of haloes.
	MPI_Request reqB[12];
	MPIStartHaloExchange(fB, rowBoundaries, reqB);

	// While the transfer completes, we can StreamCollide for the inner rows, that don't border a halo.
#if defined(AVX) || defined(SSE)
	StreamCollideVec(lowerInnerRow, upperInnerRow, 0,NYVECMAX,fB, fA, walls, rowBoundaries);
	StreamCollide(lowerInnerRow, upperInnerRow, NYVECMAX, NY, fB, fA, walls, rowBoundaries);
#else
	StreamCollide(lowerInnerRow, upperInnerRow, 0, NY, fB, fA, walls, rowBoundaries);
#endif

	// Wait for "up" transfer, first 6 requests. Then we can compute the lower row.
	MPI_Waitall(6, reqB, MPI_STATUSES_IGNORE);

	// StreamCollide for lower row bordering halo
#if defined(AVX) || defined(SSE)
	StreamCollideVec(lowerInnerRow-1, lowerInnerRow-1, 0,NYVECMAX,fB, fA, walls, rowBoundaries);
	StreamCollide(lowerInnerRow-1, lowerInnerRow-1, NYVECMAX, NY, fB, fA, walls, rowBoundaries);
#else
	StreamCollide(lowerInnerRow-1, lowerInnerRow-1, 0, NY, fB, fA, walls, rowBoundaries);
#endif

	// Wait for "down" transfer, second 6 requests. Then we can compute the upper row.
	MPI_Waitall(6, &reqB[6], MPI_STATUSES_IGNORE);

	// StreamCollide for lower row bordering halo
#if defined(AVX) || defined(SSE)
	StreamCollideVec(upperInnerRow+1, upperInnerRow+1, 0,NYVECMAX,fB, fA, walls, rowBoundaries);
	StreamCollide(upperInnerRow+1, upperInnerRow+1, NYVECMAX, NY, fB, fA, walls, rowBoundaries);
#else
	StreamCollide(upperInnerRow+1, upperInnerRow+1, 0, NY, fB, fA, walls, rowBoundaries);
#endif

}



// Combined stream and collide function, to improve caching. We read from fSrc and write updated values to fDst. This
// function is to be called with the role of fSrc and fDst switched each timestep.
void StreamCollide(
	const int iMin,
	const int iMax,
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls,
	const int * restrict rowBoundaries)
{
	int myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	int rows = rowBoundaries[myRank+1]-rowBoundaries[myRank] + HALOS;

	// Copy to temporary array
	real_t fTmp[NSPEEDS];

	for (int i = iMin; i <= iMax; i++) {
		for (int j = jMin; j < jMax; j++) {

			// pull values from neighbouring lattice points to fTmp
			int x_u = (i + 1);
			int x_d = (i - 1);
			int y_r = (j + 1) % NY;
			int y_l = (j == 0) ? (NY - 1) : (j - 1);
			fTmp[0] = fSrc[I(rows, i  ,j  , 0)];
			fTmp[1] = fSrc[I(rows, i  ,y_l, 1)];
			fTmp[2] = fSrc[I(rows, x_d,j  , 2)];
			fTmp[3] = fSrc[I(rows, i  ,y_r, 3)];
			fTmp[4] = fSrc[I(rows, x_u,j  , 4)];
			fTmp[5] = fSrc[I(rows, x_d,y_l, 5)];
			fTmp[6] = fSrc[I(rows, x_d,y_r, 6)];
			fTmp[7] = fSrc[I(rows, x_u,y_r, 7)];
			fTmp[8] = fSrc[I(rows, x_u,y_l, 8)];

			// bounce-back from wall
			if (walls[I(NX+HALOS, i,j, 0)] == 1) {
				fDst[I(rows, i,j, 1)] = fTmp[3];
				fDst[I(rows, i,j, 2)] = fTmp[4];
				fDst[I(rows, i,j, 3)] = fTmp[1];
				fDst[I(rows, i,j, 4)] = fTmp[2];
				fDst[I(rows, i,j, 5)] = fTmp[7];
				fDst[I(rows, i,j, 6)] = fTmp[8];
				fDst[I(rows, i,j, 7)] = fTmp[5];
				fDst[I(rows, i,j, 8)] = fTmp[6];
			}

			else {
				real_t density = 0;
				for (int s = 0; s < NSPEEDS; s++) {
					density += fTmp[s];
				}

				real_t u_x = (+(fTmp[6]+fTmp[2]+fTmp[5])
				              -(fTmp[7]+fTmp[4]+fTmp[8]))/density;
				real_t u_y = (+(fTmp[5]+fTmp[1]+fTmp[8])
				              -(fTmp[6]+fTmp[3]+fTmp[7]))/density;

				real_t uDotu = u_x * u_x + u_y * u_y;

				// Directional velocity components e_i dot u
				real_t u[NSPEEDS];
				u[1] =     +u_y;
				u[2] = +u_x;
				u[3] =     -u_y;
				u[4] = -u_x;
				u[5] = +u_x+u_y;
				u[6] = +u_x-u_y;
				u[7] = -u_x-u_y;
				u[8] = -u_x+u_y;

				// equilibrium density
				real_t fequ[NSPEEDS];
				fequ[0] = OMEGA0 * density * (1 - 3.0/2.0*uDotu/CSQ);
				fequ[1] = OMEGA14* density * (1 + 3.0*u[1]/CSQ +9.0/2.0*u[1]*u[1]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[2] = OMEGA14* density * (1 + 3.0*u[2]/CSQ +9.0/2.0*u[2]*u[2]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[3] = OMEGA14* density * (1 + 3.0*u[3]/CSQ +9.0/2.0*u[3]*u[3]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[4] = OMEGA14* density * (1 + 3.0*u[4]/CSQ +9.0/2.0*u[4]*u[4]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[5] = OMEGA58* density * (1 + 3.0*u[5]/CSQ +9.0/2.0*u[5]*u[5]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[6] = OMEGA58* density * (1 + 3.0*u[6]/CSQ +9.0/2.0*u[6]*u[6]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[7] = OMEGA58* density * (1 + 3.0*u[7]/CSQ +9.0/2.0*u[7]*u[7]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);
				fequ[8] = OMEGA58* density * (1 + 3.0*u[8]/CSQ +9.0/2.0*u[8]*u[8]/CSQ/CSQ -3.0/2.0*uDotu/CSQ);

				// relaxation:
				for (int s = 0; s < NSPEEDS; s++) {
					fDst[I(rows, i,j, s)] = fTmp[s] + (1.0/TAU)*(fequ[s] - fTmp[s]);
				}
			}

		}
	}

}



#if defined(AVX) || defined(SSE)
void StreamCollideVec(
	const int iMin,
	const int iMax,
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls,
	const int * restrict rowBoundaries)
{
	int myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	int rows = rowBoundaries[myRank+1]-rowBoundaries[myRank] + HALOS;

	const vector_t _one = VECTOR_SET1(1.0);
	const vector_t _three = VECTOR_SET1(3.0);
	const vector_t _half = VECTOR_SET1(0.5);
	const vector_t _threeOtwo = VECTOR_SET1(1.5);

	const vector_t _ICSQ = VECTOR_SET1(1.0/CSQ);
	const vector_t _OMEGA0 = VECTOR_SET1(OMEGA0);
	const vector_t _OMEGA14 = VECTOR_SET1(OMEGA14);
	const vector_t _OMEGA58 = VECTOR_SET1(OMEGA58);
	const vector_t _ITAU = VECTOR_SET1(1.0/TAU);

	// Copy to temporary array
	real_t fTmp[NSPEEDS*VECWIDTH];

	for (int i = iMin; i <= iMax; i++) {
		// compute VECWIDTH lattice points at once
		for (int j = jMin; j < jMax; j+=VECWIDTH) {

			// pull values from neighbouring lattice points to fTmp
			for (int k = 0; k < VECWIDTH; k++) {
				int x_u = (i + 1);
				int x_d = (i - 1);
				int y_r = (j+k + 1) % NY;
				int y_l = (j+k == 0) ? (NY - 1) : (j+k - 1);
				fTmp[VECWIDTH*0+k] = fSrc[I(rows, i  ,j+k, 0)];
				fTmp[VECWIDTH*1+k] = fSrc[I(rows, i  ,y_l, 1)];
				fTmp[VECWIDTH*2+k] = fSrc[I(rows, x_d,j+k, 2)];
				fTmp[VECWIDTH*3+k] = fSrc[I(rows, i  ,y_r, 3)];
				fTmp[VECWIDTH*4+k] = fSrc[I(rows, x_u,j+k, 4)];
				fTmp[VECWIDTH*5+k] = fSrc[I(rows, x_d,y_l, 5)];
				fTmp[VECWIDTH*6+k] = fSrc[I(rows, x_d,y_r, 6)];
				fTmp[VECWIDTH*7+k] = fSrc[I(rows, x_u,y_r, 7)];
				fTmp[VECWIDTH*8+k] = fSrc[I(rows, x_u,y_l, 8)];
			}

			_mm_prefetch(&fSrc[I(rows, i  ,j+VECWIDTH, 0)],0);
			_mm_prefetch(&fSrc[I(rows, i  ,j+VECWIDTH, 1)],0);
			_mm_prefetch(&fSrc[I(rows, i-1,j+VECWIDTH, 2)],0);
			_mm_prefetch(&fSrc[I(rows, i  ,j+VECWIDTH, 3)],0);
			_mm_prefetch(&fSrc[I(rows, i+1,j+VECWIDTH, 4)],0);
			_mm_prefetch(&fSrc[I(rows, i-1,j+VECWIDTH, 5)],0);
			_mm_prefetch(&fSrc[I(rows, i-1,j+VECWIDTH, 6)],0);
			_mm_prefetch(&fSrc[I(rows, i+1,j+VECWIDTH, 7)],0);
			_mm_prefetch(&fSrc[I(rows, i+1,j+VECWIDTH, 8)],0);

			vector_t _density = VECTOR_SET1(0.0);

			vector_t _f0 = VECTOR_LOAD(&fTmp[VECWIDTH*0]);
			vector_t _f1 = VECTOR_LOAD(&fTmp[VECWIDTH*1]);
			vector_t _f2 = VECTOR_LOAD(&fTmp[VECWIDTH*2]);
			vector_t _f3 = VECTOR_LOAD(&fTmp[VECWIDTH*3]);
			vector_t _f4 = VECTOR_LOAD(&fTmp[VECWIDTH*4]);
			vector_t _f5 = VECTOR_LOAD(&fTmp[VECWIDTH*5]);
			vector_t _f6 = VECTOR_LOAD(&fTmp[VECWIDTH*6]);
			vector_t _f7 = VECTOR_LOAD(&fTmp[VECWIDTH*7]);
			vector_t _f8 = VECTOR_LOAD(&fTmp[VECWIDTH*8]);

			_density = VECTOR_ADD(_density, _f0);
			_density = VECTOR_ADD(_density, _f1);
			_density = VECTOR_ADD(_density, _f2);
			_density = VECTOR_ADD(_density, _f3);
			_density = VECTOR_ADD(_density, _f4);
			_density = VECTOR_ADD(_density, _f5);
			_density = VECTOR_ADD(_density, _f6);
			_density = VECTOR_ADD(_density, _f7);
			_density = VECTOR_ADD(_density, _f8);

			vector_t _u_x = VECTOR_ADD(_f6, _f2);
			_u_x = VECTOR_ADD(_u_x, _f5);
			_u_x = VECTOR_SUB(_u_x, _f7);
			_u_x = VECTOR_SUB(_u_x, _f4);
			_u_x = VECTOR_SUB(_u_x, _f8);
			_u_x = VECTOR_DIV(_u_x, _density);

			vector_t _u_y = VECTOR_ADD(_f5, _f1);
			_u_y = VECTOR_ADD(_u_y, _f8);
			_u_y = VECTOR_SUB(_u_y, _f6);
			_u_y = VECTOR_SUB(_u_y, _f3);
			_u_y = VECTOR_SUB(_u_y, _f7);
			_u_y = VECTOR_DIV(_u_y, _density);


			vector_t _uDotuTerm = VECTOR_MUL(_threeOtwo,VECTOR_MUL(_ICSQ,VECTOR_ADD(VECTOR_MUL(_u_x,_u_x),VECTOR_MUL(_u_y,_u_y))));

			// Directional velocity components e_i dot u, multiplied by 3/c^2
			_u_x = VECTOR_MUL(_u_x, VECTOR_MUL(_three, _ICSQ));
			_u_y = VECTOR_MUL(_u_y, VECTOR_MUL(_three, _ICSQ));
			vector_t _u1 = _u_y;
			vector_t _u2 = _u_x;
			vector_t _u3 = VECTOR_MUL(VECTOR_SET1(-1.0),_u_y);
			vector_t _u4 = VECTOR_MUL(VECTOR_SET1(-1.0),_u_x);
			vector_t _u5 = VECTOR_ADD(_u_x,_u_y);
			vector_t _u6 = VECTOR_SUB(_u_x,_u_y);
			vector_t _u7 = VECTOR_MUL(VECTOR_SET1(-1.0),VECTOR_ADD(_u_x,_u_y));
			vector_t _u8 = VECTOR_MUL(VECTOR_SET1(-1.0),VECTOR_SUB(_u_x,_u_y));


			// equilibrium density
			vector_t _fequ0 = VECTOR_SUB(_one, _uDotuTerm);
			vector_t _fequ1 = VECTOR_ADD(_one, VECTOR_ADD(_u1, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u1,_u1)), _uDotuTerm)));
			vector_t _fequ2 = VECTOR_ADD(_one, VECTOR_ADD(_u2, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u2,_u2)), _uDotuTerm)));
			vector_t _fequ3 = VECTOR_ADD(_one, VECTOR_ADD(_u3, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u3,_u3)), _uDotuTerm)));
			vector_t _fequ4 = VECTOR_ADD(_one, VECTOR_ADD(_u4, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u4,_u4)), _uDotuTerm)));
			vector_t _fequ5 = VECTOR_ADD(_one, VECTOR_ADD(_u5, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u5,_u5)), _uDotuTerm)));
			vector_t _fequ6 = VECTOR_ADD(_one, VECTOR_ADD(_u6, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u6,_u6)), _uDotuTerm)));
			vector_t _fequ7 = VECTOR_ADD(_one, VECTOR_ADD(_u7, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u7,_u7)), _uDotuTerm)));
			vector_t _fequ8 = VECTOR_ADD(_one, VECTOR_ADD(_u8, VECTOR_SUB(VECTOR_MUL(_half, VECTOR_MUL(_u8,_u8)), _uDotuTerm)));


			// now multiply by omega and density
			_fequ0 = VECTOR_MUL(_fequ0, VECTOR_MUL(_OMEGA0, _density));
			_fequ1 = VECTOR_MUL(_fequ1, VECTOR_MUL(_OMEGA14, _density));
			_fequ2 = VECTOR_MUL(_fequ2, VECTOR_MUL(_OMEGA14, _density));
			_fequ3 = VECTOR_MUL(_fequ3, VECTOR_MUL(_OMEGA14, _density));
			_fequ4 = VECTOR_MUL(_fequ4, VECTOR_MUL(_OMEGA14, _density));
			_fequ5 = VECTOR_MUL(_fequ5, VECTOR_MUL(_OMEGA58, _density));
			_fequ6 = VECTOR_MUL(_fequ6, VECTOR_MUL(_OMEGA58, _density));
			_fequ7 = VECTOR_MUL(_fequ7, VECTOR_MUL(_OMEGA58, _density));
			_fequ8 = VECTOR_MUL(_fequ8, VECTOR_MUL(_OMEGA58, _density));


			// relaxation: check if cell is blocked!
			int wallsSum = 0;
			for (int k = 0; k < VECWIDTH; k++) {
				wallsSum += walls[I(NX+HALOS, i,j+k, 0)];
			}
			if (wallsSum == 0) {
				VECTOR_STORE(&fDst[I(rows, i,j, 0)], VECTOR_ADD(_f0, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ0,_f0))));
				VECTOR_STORE(&fDst[I(rows, i,j, 1)], VECTOR_ADD(_f1, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ1,_f1))));
				VECTOR_STORE(&fDst[I(rows, i,j, 2)], VECTOR_ADD(_f2, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ2,_f2))));
				VECTOR_STORE(&fDst[I(rows, i,j, 3)], VECTOR_ADD(_f3, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ3,_f3))));
				VECTOR_STORE(&fDst[I(rows, i,j, 4)], VECTOR_ADD(_f4, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ4,_f4))));
				VECTOR_STORE(&fDst[I(rows, i,j, 5)], VECTOR_ADD(_f5, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ5,_f5))));
				VECTOR_STORE(&fDst[I(rows, i,j, 6)], VECTOR_ADD(_f6, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ6,_f6))));
				VECTOR_STORE(&fDst[I(rows, i,j, 7)], VECTOR_ADD(_f7, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ7,_f7))));
				VECTOR_STORE(&fDst[I(rows, i,j, 8)], VECTOR_ADD(_f8, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ8,_f8))));
			}
			else {

				for (int k = 0; k < VECWIDTH; k++) {
					if (walls[I(NX+HALOS, i,j+k, 0)] == 0) {
						fDst[I(rows, i,j+k, 0)] = fTmp[VECWIDTH*0+k] + (1.0/TAU)*( ((real_t*)&_fequ0)[k] - fTmp[VECWIDTH*0+k]);
						fDst[I(rows, i,j+k, 1)] = fTmp[VECWIDTH*1+k] + (1.0/TAU)*( ((real_t*)&_fequ1)[k] - fTmp[VECWIDTH*1+k]);
						fDst[I(rows, i,j+k, 2)] = fTmp[VECWIDTH*2+k] + (1.0/TAU)*( ((real_t*)&_fequ2)[k] - fTmp[VECWIDTH*2+k]);
						fDst[I(rows, i,j+k, 3)] = fTmp[VECWIDTH*3+k] + (1.0/TAU)*( ((real_t*)&_fequ3)[k] - fTmp[VECWIDTH*3+k]);
						fDst[I(rows, i,j+k, 4)] = fTmp[VECWIDTH*4+k] + (1.0/TAU)*( ((real_t*)&_fequ4)[k] - fTmp[VECWIDTH*4+k]);
						fDst[I(rows, i,j+k, 5)] = fTmp[VECWIDTH*5+k] + (1.0/TAU)*( ((real_t*)&_fequ5)[k] - fTmp[VECWIDTH*5+k]);
						fDst[I(rows, i,j+k, 6)] = fTmp[VECWIDTH*6+k] + (1.0/TAU)*( ((real_t*)&_fequ6)[k] - fTmp[VECWIDTH*6+k]);
						fDst[I(rows, i,j+k, 7)] = fTmp[VECWIDTH*7+k] + (1.0/TAU)*( ((real_t*)&_fequ7)[k] - fTmp[VECWIDTH*7+k]);
						fDst[I(rows, i,j+k, 8)] = fTmp[VECWIDTH*8+k] + (1.0/TAU)*( ((real_t*)&_fequ8)[k] - fTmp[VECWIDTH*8+k]);
					}
					else {
						fDst[I(rows, i,j+k, 1)] = fTmp[VECWIDTH*3+k];
						fDst[I(rows, i,j+k, 2)] = fTmp[VECWIDTH*4+k];
						fDst[I(rows, i,j+k, 3)] = fTmp[VECWIDTH*1+k];
						fDst[I(rows, i,j+k, 4)] = fTmp[VECWIDTH*2+k];
						fDst[I(rows, i,j+k, 5)] = fTmp[VECWIDTH*7+k];
						fDst[I(rows, i,j+k, 6)] = fTmp[VECWIDTH*8+k];
						fDst[I(rows, i,j+k, 7)] = fTmp[VECWIDTH*5+k];
						fDst[I(rows, i,j+k, 8)] = fTmp[VECWIDTH*6+k];
					}
				}

			}


		}
	}

}
#endif



void ApplySource(
	real_t * restrict fAB,
	const char * restrict walls,
	const int * restrict rowBoundaries)
{
	int myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// Accelerate the flow so we have a pipe with fluid flowing along it.
	// Increase "incoming" f, decrease "outgoing" (ie, leaving the domain) down the first column

	int rows = rowBoundaries[myRank+1]-rowBoundaries[myRank] + HALOS;

	for (int i = 1; i <= rowBoundaries[myRank+1]-rowBoundaries[myRank]; i++) {

		if (walls[I(NX+HALOS, i,0, 0)] == 0) {
			// check none of the "outgoing" densities will end up negative. f is strictly >= 0!
			if (  (fAB[I(rows, i,0, 6)] - ACCEL*OMEGA58 > 0.0)
			    &&(fAB[I(rows, i,0, 3)] - ACCEL*OMEGA14 > 0.0)
			    &&(fAB[I(rows, i,0, 7)] - ACCEL*OMEGA58 > 0.0) ) {

				fAB[I(rows, i,0, 6)] -= ACCEL*OMEGA58;
				fAB[I(rows, i,0, 3)] -= ACCEL*OMEGA14;
				fAB[I(rows, i,0, 7)] -= ACCEL*OMEGA58;

				fAB[I(rows, i,0, 5)] += ACCEL*OMEGA58;
				fAB[I(rows, i,0, 1)] += ACCEL*OMEGA14;
				fAB[I(rows, i,0, 8)] += ACCEL*OMEGA58;
			}
		}

	}

}



real_t ComputeReynolds(
	const real_t * restrict fM,
	const char * restrict walls)
{
	// Compute reynolds number over central column
	int j = (int)((real_t)NY / 2.0);

	real_t u_yTotal = 0;
	int latticePoints = 0;

	for (int i = 0; i < NX; i++) {
		// WALLS INDEX MUST BE OFFSET HERE.
		if (walls[I(NX+HALOS, i+1,j, 0)] == 0) {
			real_t density = 0.0;
			for (int s = 0; s < NSPEEDS; s++) {
				density += fM[I(NX, i,j, s)];
			}
			u_yTotal += (+(fM[I(NX, i,j, 5)]+fM[I(NX, i,j, 1)]+fM[I(NX, i,j, 8)])
			             -(fM[I(NX, i,j, 6)]+fM[I(NX, i,j, 3)]+fM[I(NX, i,j, 7)]))/density;
			latticePoints++;
		}
	}

	real_t viscosity = 1.0/3.0 * (TAU - 1.0/2.0);

	return u_yTotal/(real_t)latticePoints * 10 / viscosity;
}



void InitializeArrays(
	real_t * restrict fM,
	char * restrict walls)
{
	// Add walls
	for (int i = 1; i <= NX; i++) {
		for (int j = 0; j < NY; j++) {
			walls[I(NX+HALOS, i,j, 0)] = 0;
		}
	}
	// barrier
	for (int i = 21; i <= 220; i++) {
		walls[I(NX+HALOS, i,100, 0)] = 1;
		walls[I(NX+HALOS, i,101, 0)] = 1;
		walls[I(NX+HALOS, i,102, 0)] = 1;
		walls[I(NX+HALOS, i,103, 0)] = 1;
		walls[I(NX+HALOS, i,104, 0)] = 1;
	}
	// edges top and bottom
	for (int j = 0; j < NY; j++) {
		walls[I(NX+HALOS, 1,j, 0)] = 1;
		walls[I(NX+HALOS, NX,j, 0)] = 1;
	}


	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			fM[I(NX, i,j, 0)] = INITIALDENSITY * OMEGA0;
			fM[I(NX, i,j, 1)] = INITIALDENSITY * OMEGA14;
			fM[I(NX, i,j, 2)] = INITIALDENSITY * OMEGA14;
			fM[I(NX, i,j, 3)] = INITIALDENSITY * OMEGA14;
			fM[I(NX, i,j, 4)] = INITIALDENSITY * OMEGA14;
			fM[I(NX, i,j, 5)] = INITIALDENSITY * OMEGA58;
			fM[I(NX, i,j, 6)] = INITIALDENSITY * OMEGA58;
			fM[I(NX, i,j, 7)] = INITIALDENSITY * OMEGA58;
			fM[I(NX, i,j, 8)] = INITIALDENSITY * OMEGA58;
		}
	}


}



void MPIDistributeInitialConditions(
	real_t * restrict fM,
	real_t * restrict fA,
	char * restrict walls,
	const int * restrict rowBoundaries)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);


	MPI_Request *reqf;
	// Make NSPEEDS*totalRanks sends for fM -> fA, including master sending to itself (must use non-blocking send!)
	reqf = malloc(NSPEEDS*totalRanks*sizeof(MPI_Request));


	// fM -> fA
	IFMASTER {
		for (int r = 0; r < totalRanks; r++) {
			int sendRows = rowBoundaries[r+1]-rowBoundaries[r];
			for (int n = 0; n < NSPEEDS; n++) {
				MPI_Isend(&fM[I(NX, rowBoundaries[r],0, n)], sendRows*NYPADDED, MPI_REAL_T, r, r*NSPEEDS+n, MPI_COMM_WORLD, &reqf[r*NSPEEDS+n]);
			}
		}
	}
	// All ranks receive into fA, on row 1! Halos left blank here
	int myRows = rowBoundaries[myRank+1]-rowBoundaries[myRank];
	for (int n = 0; n < NSPEEDS; n++) {
		MPI_Recv(&fA[I(myRows+HALOS, 1,0, n)], myRows*NYPADDED, MPI_REAL_T, FROMMASTER, myRank*NSPEEDS+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}


	// walls -> walls. Master doesn't need to send to itself
	IFMASTER {
		for (int r = 1; r < totalRanks; r++) {
			int sendRows = rowBoundaries[r+1]-rowBoundaries[r];
			MPI_Send(&walls[I(NX+HALOS, rowBoundaries[r],0, 0)], sendRows*NYPADDED, MPI_CHAR, r, 0, MPI_COMM_WORLD);
		}
	}
	// non-master ranks receive into walls at row 1!
	else {
		MPI_Recv(&walls[I(NX+HALOS, 1,0, 0)], myRows*NYPADDED, MPI_CHAR, FROMMASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Master waits on completion
	IFMASTER {
		MPI_Waitall(NSPEEDS*totalRanks, reqf, MPI_STATUSES_IGNORE);
	}


}



void MPIMasterReceiveLattice(
	real_t * restrict fM,
	real_t * restrict fAB,
	const int * restrict rowBoundaries)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	int rows = rowBoundaries[myRank+1]-rowBoundaries[myRank] + HALOS;

	// Blocking version
	// Master receives from all, and copies from its own fB to fA
	IFMASTER {
		for (int r = 1; r < totalRanks; r++) {
			for (int n = 0; n < NSPEEDS; n++) {
				MPI_Recv(&fM[I(NX, rowBoundaries[r],0, n)], (rowBoundaries[r+1]-rowBoundaries[r])*NYPADDED, MPI_REAL_T, r, r*NSPEEDS+n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		memcpy(&fM[I(NX, 0,0, 0)], &fAB[I(rows, 1, 0, 0)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 1)], &fAB[I(rows, 1, 0, 1)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 2)], &fAB[I(rows, 1, 0, 2)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 3)], &fAB[I(rows, 1, 0, 3)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 4)], &fAB[I(rows, 1, 0, 4)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 5)], &fAB[I(rows, 1, 0, 5)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 6)], &fAB[I(rows, 1, 0, 6)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 7)], &fAB[I(rows, 1, 0, 7)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
		memcpy(&fM[I(NX, 0,0, 8)], &fAB[I(rows, 1, 0, 8)], rowBoundaries[1]*NYPADDED*sizeof(real_t));
	}
	// Other ranks send to master from fB
	else {
		for (int n = 0; n < NSPEEDS; n++) {
			MPI_Send(&fAB[I(rows, 1,0, n)], (rowBoundaries[myRank+1]-rowBoundaries[myRank])*NYPADDED, MPI_REAL_T, FROMMASTER, myRank*NSPEEDS+n, MPI_COMM_WORLD);
		}
	}

}



void MPIStartHaloExchange(
	real_t * restrict fAB,
	const int * restrict rowBoundaries,
	MPI_Request * restrict req)
{
	int totalRanks, myRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);

	int rankAbove = myRank+1;
	int rankBelow = myRank-1;
	IFMASTER {
		rankBelow = totalRanks-1;
	}
	IFRANK(totalRanks-1) {
		rankAbove = 0;
	}

	// Determine the rows we have to send and receive into:
	int myRows = rowBoundaries[myRank+1]-rowBoundaries[myRank];
	int lowerHalo = 0;
	int lowerData = 1;
	int upperData = myRows;
	int upperHalo = myRows+1;

	// Store fAB array row count:
	int rowsfAB = myRows + HALOS;


	// These need to be non-blocking as each rank must also receive one of these sends.

	// Send f 6, 2, 5 of upper row to rank above.
	MPI_Isend(&fAB[I(rowsfAB, upperData,0, 6)], NY, MPI_REAL_T, rankAbove, 6, MPI_COMM_WORLD, &req[0]);
	MPI_Isend(&fAB[I(rowsfAB, upperData,0, 2)], NY, MPI_REAL_T, rankAbove, 2, MPI_COMM_WORLD, &req[1]);
	MPI_Isend(&fAB[I(rowsfAB, upperData,0, 5)], NY, MPI_REAL_T, rankAbove, 5, MPI_COMM_WORLD, &req[2]);

	// Receive f 6, 2, 5 from rank below, store in lower halo
	MPI_Irecv(&fAB[I(rowsfAB, lowerHalo,0, 6)], NY, MPI_REAL_T, rankBelow, 6, MPI_COMM_WORLD, &req[3]);
	MPI_Irecv(&fAB[I(rowsfAB, lowerHalo,0, 2)], NY, MPI_REAL_T, rankBelow, 2, MPI_COMM_WORLD, &req[4]);
	MPI_Irecv(&fAB[I(rowsfAB, lowerHalo,0, 5)], NY, MPI_REAL_T, rankBelow, 5, MPI_COMM_WORLD, &req[5]);

	// Send f 7, 4, 8 of lower row to rank below
	MPI_Isend(&fAB[I(rowsfAB, lowerData,0, 7)], NY, MPI_REAL_T, rankBelow, 7, MPI_COMM_WORLD, &req[6]);
	MPI_Isend(&fAB[I(rowsfAB, lowerData,0, 4)], NY, MPI_REAL_T, rankBelow, 4, MPI_COMM_WORLD, &req[7]);
	MPI_Isend(&fAB[I(rowsfAB, lowerData,0, 8)], NY, MPI_REAL_T, rankBelow, 8, MPI_COMM_WORLD, &req[8]);

	// Receive f 7, 4, 8 from rank above, store in upper halo
	MPI_Irecv(&fAB[I(rowsfAB, upperHalo,0, 7)], NY, MPI_REAL_T, rankAbove, 7, MPI_COMM_WORLD, &req[9]);
	MPI_Irecv(&fAB[I(rowsfAB, upperHalo,0, 4)], NY, MPI_REAL_T, rankAbove, 4, MPI_COMM_WORLD, &req[10]);
	MPI_Irecv(&fAB[I(rowsfAB, upperHalo,0, 8)], NY, MPI_REAL_T, rankAbove, 8, MPI_COMM_WORLD, &req[11]);

}



void PrintLattice(
	int timeStep,
	const real_t * restrict fM)
{
	char filename[100];
	sprintf(filename,"data/%d.csv",timeStep);
	FILE *fp = fopen(filename,"w");

	if (fp == NULL) printf("Error opening file %s\n", filename);

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {

			real_t density = 0;
			for (int s = 0; s < NSPEEDS; s++) {
				density += fM[I(NX, i,j, s)];
			}

			real_t u_x = (+(fM[I(NX, i,j, 6)]+fM[I(NX, i,j, 2)]+fM[I(NX, i,j, 5)])
			              -(fM[I(NX, i,j, 7)]+fM[I(NX, i,j, 4)]+fM[I(NX, i,j, 8)]))/density;
			real_t u_y = (+(fM[I(NX, i,j, 5)]+fM[I(NX, i,j, 1)]+fM[I(NX, i,j, 8)])
			              -(fM[I(NX, i,j, 6)]+fM[I(NX, i,j, 3)]+fM[I(NX, i,j, 7)]))/density;

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
