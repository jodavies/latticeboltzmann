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


// d2q9 fixed parameters
#define NSPEEDS 9
#define OMEGA0  (4.0/9.0)
#define OMEGA14 (1.0/9.0)
#define OMEGA58 (1.0/36.0)

// Boundary condition. Wrap-around (periodic) or not (fluid flows out of the domain)
#define WRAPAROUND 1

// variable parameters
#define NX 400
#define NY 2001
#define TAU 0.8
#define CSQ (1.0)

#define NTIMESTEPS 20000
#define PRINTSTATSEVERY 1000
#define SAVELATTICEEVERY 100
#define ACCEL 0.005
#define INITIALDENSITY 0.1


// Choose precision and vectorization
//#include "prec_double_avx.h"
//#include "prec_double_sse.h"
//#include "prec_double_serial.h"
#include "prec_float_avx.h"
//#include "prec_float_sse.h"
//#include "prec_float_serial.h"



// Macro for array indexing. We need the array stride to be such that we have correct
// alignment for vector loads/stores on rows after the first.
#define NYPADDED (ALIGNREQUIREMENT*((NY-1)/ALIGNREQUIREMENT)+ALIGNREQUIREMENT)
#define I(i,j, speed) ((speed)*NX*NYPADDED + (i)*NYPADDED + (j))

// Also store the multiple of VECWIDTH below NY, since the vectorized functions need to terminate here,
// possibly with a scalar function cleaning up the "extra".
#define NYVECMAX (VECWIDTH*(NY/VECWIDTH))

// For approximate GFLOPs report: we do ~124 FLOP per lattice point, obtained simply by counting ADD,SUB,MUL,DIV
// intructions in CollideVec.
#define FLOPPERLATTICEPOINT (124.0)



// Function prototypes
void DoTimeStep(
	real_t * restrict f,
	real_t * restrict fScratch,
	const int * restrict walls);

void ApplySource(
	real_t * restrict f,
	const int * restrict walls);

void Stream(
	const real_t * restrict f,
	real_t * restrict fScratch,
	const int * restrict walls);

void Collide(
	const int jMin,
	const int jMax,
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls);

#if defined(AVX) || defined(SSE)
void CollideVec(
	const int jMin,
	const int jMax,
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls);
#endif

real_t ComputeReynolds(
	const real_t * restrict f,
	const int * restrict walls);

void InitializeArrays(
	real_t * restrict f,
	real_t * restrict fScratch,
	int * restrict walls);

void PrintLattice(int timeStep, const real_t * restrict f);

double GetWallTime(void);



int main(void)
{
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

	// Allocate memory. Use _mm_malloc() for alignment. Want 32byte alignment for vector instructions.
	real_t * f;
	real_t * fScratch;
	int * walls;

	int allocSize = NX * NYPADDED * NSPEEDS;

	f = _mm_malloc(allocSize * sizeof *f, 32);
	fScratch = _mm_malloc(allocSize * sizeof *f, 32);
	walls = _mm_malloc(NX * NYPADDED * sizeof *walls, 32);

	InitializeArrays(f, fScratch, walls);

	printf("Lattice Size: %dx%d (rows padded to %d, vectorized loop running to %d)\n", NX,NY, NYPADDED, NYVECMAX);

	// Begin iterations
	double timeElapsed = GetWallTime();

	for (int n = 0; n < NTIMESTEPS; n++) {

		if (n % PRINTSTATSEVERY == 0) {
			if (n != 0) {
				double complete = (double)n/(double)NTIMESTEPS;
				int secElap = (int)(GetWallTime()-timeElapsed);
				int secRem = (int)(secElap/complete*(1.0-complete));
				// each timestep requires two reads and two writes of the f and fScratch arrays:
				double avgbw = 4.0*n*sizeof(real_t)*NX*NY*NSPEEDS/(GetWallTime()-timeElapsed)/1024/1024/1024;
				printf("%5.2lf%%--Elapsed: %3dm%02ds, Remaining: %3dm%02ds. [Updates/s: %.2le, Update BW: ~%.2lf GB/s, GFLOPs: ~%.2lf]\n",
				       complete*100, secElap/60, secElap%60, secRem/60, secRem%60, n/(double)secElap,
				       avgbw, FLOPPERLATTICEPOINT*NX*NY*n/(double)secElap/1000.0/1000.0/1000.0);
			}
		}
		if (n % SAVELATTICEEVERY == 0) {
			PrintLattice(n, f);
		}

		// Do a timestep
		DoTimeStep(f, fScratch, walls);

	}

	timeElapsed = GetWallTime() - timeElapsed;
	// End iterations

	printf("Time: %lf Re %.10le\n", timeElapsed, ComputeReynolds(f, walls));


	// Free dynamically allocated memory
	_mm_free(f);
	_mm_free(fScratch);
	_mm_free(walls);

	return EXIT_SUCCESS;
}



void DoTimeStep(
	real_t * restrict f,
	real_t * restrict fScratch,
	const int * restrict walls)
{

	ApplySource(f, walls);

	Stream(f, fScratch, walls);

#if defined(AVX) || defined(SSE)
	CollideVec(0, NYVECMAX, f, fScratch, walls);
	Collide(NYVECMAX, NY, f, fScratch, walls);
#else
	Collide(0, NY, f, fScratch, walls);
#endif

}



void Stream(
	const real_t * restrict f,
	real_t * restrict fScratch,
	const int * restrict walls)
{


#ifdef WRAPAROUND
	// Wrap around boundary condition
	// Bottom row
	int i = 0;
	for (int j = 0; j < NY; j++) {

		int x_u = (i + 1) % NX;
		int x_d = (i == 0) ? (NX - 1) : (i - 1);
		int y_r = (j + 1) % NY;
		int y_l = (j == 0) ? (NY - 1) : (j - 1);

		fScratch[I(i  , j  , 0)] = f[I(i,j, 0)];
		fScratch[I(i  , y_r, 1)] = f[I(i,j, 1)];
		fScratch[I(x_u, j  , 2)] = f[I(i,j, 2)];
		fScratch[I(i  , y_l, 3)] = f[I(i,j, 3)];
		fScratch[I(x_d, j  , 4)] = f[I(i,j, 4)];
		fScratch[I(x_u, y_r, 5)] = f[I(i,j, 5)];
		fScratch[I(x_u, y_l, 6)] = f[I(i,j, 6)];
		fScratch[I(x_d, y_l, 7)] = f[I(i,j, 7)];
		fScratch[I(x_d, y_r, 8)] = f[I(i,j, 8)];
	}
	// Top row
	i = NX-1;
	for (int j = 0; j < NY; j++) {

		int x_u = (i + 1) % NX;
		int x_d = (i == 0) ? (NX - 1) : (i - 1);
		int y_r = (j + 1) % NY;
		int y_l = (j == 0) ? (NY - 1) : (j - 1);

		fScratch[I(i  , j  , 0)] = f[I(i,j, 0)];
		fScratch[I(i  , y_r, 1)] = f[I(i,j, 1)];
		fScratch[I(x_u, j  , 2)] = f[I(i,j, 2)];
		fScratch[I(i  , y_l, 3)] = f[I(i,j, 3)];
		fScratch[I(x_d, j  , 4)] = f[I(i,j, 4)];
		fScratch[I(x_u, y_r, 5)] = f[I(i,j, 5)];
		fScratch[I(x_u, y_l, 6)] = f[I(i,j, 6)];
		fScratch[I(x_d, y_l, 7)] = f[I(i,j, 7)];
		fScratch[I(x_d, y_r, 8)] = f[I(i,j, 8)];
	}
	// Left, right columns (excluding top and bottom rows! already done)
	int j = 0;
	for (int i = 1; i < NX-1; i++) {

		int x_u = (i + 1) % NX;
		int x_d = (i == 0) ? (NX - 1) : (i - 1);
		int y_r = (j + 1) % NY;
		int y_l = (j == 0) ? (NY - 1) : (j - 1);

		fScratch[I(i  , j  , 0)] = f[I(i,j, 0)];
		fScratch[I(i  , y_r, 1)] = f[I(i,j, 1)];
		fScratch[I(x_u, j  , 2)] = f[I(i,j, 2)];
		fScratch[I(i  , y_l, 3)] = f[I(i,j, 3)];
		fScratch[I(x_d, j  , 4)] = f[I(i,j, 4)];
		fScratch[I(x_u, y_r, 5)] = f[I(i,j, 5)];
		fScratch[I(x_u, y_l, 6)] = f[I(i,j, 6)];
		fScratch[I(x_d, y_l, 7)] = f[I(i,j, 7)];
		fScratch[I(x_d, y_r, 8)] = f[I(i,j, 8)];
	}
	j = NY-1;
	for (int i = 1; i < NX-1; i++) {

		int x_u = (i + 1) % NX;
		int x_d = (i == 0) ? (NX - 1) : (i - 1);
		int y_r = (j + 1) % NY;
		int y_l = (j == 0) ? (NY - 1) : (j - 1);

		fScratch[I(i  , j  , 0)] = f[I(i,j, 0)];
		fScratch[I(i  , y_r, 1)] = f[I(i,j, 1)];
		fScratch[I(x_u, j  , 2)] = f[I(i,j, 2)];
		fScratch[I(i  , y_l, 3)] = f[I(i,j, 3)];
		fScratch[I(x_d, j  , 4)] = f[I(i,j, 4)];
		fScratch[I(x_u, y_r, 5)] = f[I(i,j, 5)];
		fScratch[I(x_u, y_l, 6)] = f[I(i,j, 6)];
		fScratch[I(x_d, y_l, 7)] = f[I(i,j, 7)];
		fScratch[I(x_d, y_r, 8)] = f[I(i,j, 8)];
	}
#else
	// Non-wraparound BCs
#endif


	// Central region: no BC dependence
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i,1,0)], &f[I(i,1,0)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i,2,1)], &f[I(i,1,1)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i+1,1,2)], &f[I(i,1,2)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i,0,3)], &f[I(i,1,3)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i-1,1,4)], &f[I(i,1,4)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i+1,2,5)], &f[I(i,1,5)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i+1,0,6)], &f[I(i,1,6)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i-1,0,7)], &f[I(i,1,7)], (NY-2)* sizeof *f);
		memcpy(&fScratch[I(i-1,2,8)], &f[I(i,1,8)], (NY-2)* sizeof *f);
	}

}



void Collide(
	const int jMin,
	const int jMax,
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls)
{

#pragma omp parallel for default(none) shared(f,fScratch,walls) schedule(static)
	for (int i = 0; i < NX; i++) {
		for (int j = jMin; j < jMax; j++) {

			// bounce-back from wall
			if (walls[I(i,j, 0)] == 1) {
				f[I(i,j, 1)] = fScratch[I(i,j, 3)];
				f[I(i,j, 2)] = fScratch[I(i,j, 4)];
				f[I(i,j, 3)] = fScratch[I(i,j, 1)];
				f[I(i,j, 4)] = fScratch[I(i,j, 2)];
				f[I(i,j, 5)] = fScratch[I(i,j, 7)];
				f[I(i,j, 6)] = fScratch[I(i,j, 8)];
				f[I(i,j, 7)] = fScratch[I(i,j, 5)];
				f[I(i,j, 8)] = fScratch[I(i,j, 6)];
			}

			else {
				real_t density = 0;
				for (int s = 0; s < NSPEEDS; s++) {
					density += fScratch[I(i,j, s)];
				}

				real_t u_x = (+(fScratch[I(i,j, 6)]+fScratch[I(i,j, 2)]+fScratch[I(i,j, 5)])
				              -(fScratch[I(i,j, 7)]+fScratch[I(i,j, 4)]+fScratch[I(i,j, 8)]))/density;
				real_t u_y = (+(fScratch[I(i,j, 5)]+fScratch[I(i,j, 1)]+fScratch[I(i,j, 8)])
				              -(fScratch[I(i,j, 6)]+fScratch[I(i,j, 3)]+fScratch[I(i,j, 7)]))/density;

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
					f[I(i,j, s)] = fScratch[I(i,j, s)] + (1.0/TAU)*(fequ[s] - fScratch[I(i,j, s)]);
				}
			}

		}
	}


}



// Vectorized version. Only faster is most of the domain is NOT a wall! We compute the relaxation step for every
// lattice point, and then update f depending on whether the points are walls or not.
#if defined(AVX) || defined(SSE)
void CollideVec(
	const int jMin,
	const int jMax,
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls)
{
	const vector_t _one = VECTOR_SET1(1.0);
	const vector_t _three = VECTOR_SET1(3.0);
	const vector_t _half = VECTOR_SET1(0.5);
	const vector_t _threeOtwo = VECTOR_SET1(1.5);

	const vector_t _ICSQ = VECTOR_SET1(1.0/CSQ);
	const vector_t _OMEGA0 = VECTOR_SET1(OMEGA0);
	const vector_t _OMEGA14 = VECTOR_SET1(OMEGA14);
	const vector_t _OMEGA58 = VECTOR_SET1(OMEGA58);

#pragma omp parallel for default(none) shared(f,fScratch,walls) schedule(static)
	for (int i = 0; i < NX; i++) {
		// compute VECWIDTH lattice points at once
		for (int j = jMin; j < jMax; j+=VECWIDTH) {

			vector_t _density = VECTOR_SET1(0.0);

			vector_t _f0 = VECTOR_LOAD(&fScratch[I(i,j, 0)]);
			vector_t _f1 = VECTOR_LOAD(&fScratch[I(i,j, 1)]);
			vector_t _f2 = VECTOR_LOAD(&fScratch[I(i,j, 2)]);
			vector_t _f3 = VECTOR_LOAD(&fScratch[I(i,j, 3)]);
			vector_t _f4 = VECTOR_LOAD(&fScratch[I(i,j, 4)]);
			vector_t _f5 = VECTOR_LOAD(&fScratch[I(i,j, 5)]);
			vector_t _f6 = VECTOR_LOAD(&fScratch[I(i,j, 6)]);
			vector_t _f7 = VECTOR_LOAD(&fScratch[I(i,j, 7)]);
			vector_t _f8 = VECTOR_LOAD(&fScratch[I(i,j, 8)]);

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
				wallsSum += walls[I(i,j+k, 0)];
			}
			if (wallsSum == 0) {
				VECTOR_STORE(&f[I(i,j, 0)], VECTOR_ADD(_f0, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ0,_f0))));
				VECTOR_STORE(&f[I(i,j, 1)], VECTOR_ADD(_f1, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ1,_f1))));
				VECTOR_STORE(&f[I(i,j, 2)], VECTOR_ADD(_f2, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ2,_f2))));
				VECTOR_STORE(&f[I(i,j, 3)], VECTOR_ADD(_f3, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ3,_f3))));
				VECTOR_STORE(&f[I(i,j, 4)], VECTOR_ADD(_f4, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ4,_f4))));
				VECTOR_STORE(&f[I(i,j, 5)], VECTOR_ADD(_f5, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ5,_f5))));
				VECTOR_STORE(&f[I(i,j, 6)], VECTOR_ADD(_f6, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ6,_f6))));
				VECTOR_STORE(&f[I(i,j, 7)], VECTOR_ADD(_f7, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ7,_f7))));
				VECTOR_STORE(&f[I(i,j, 8)], VECTOR_ADD(_f8, VECTOR_MUL(VECTOR_SET1(1.0/TAU),VECTOR_SUB(_fequ8,_f8))));
			}
			else {

				for (int k = 0; k < VECWIDTH; k++) {
					if (walls[I(i,j+k, 0)] == 0) {
						f[I(i,j+k, 0)] = fScratch[I(i,j+k, 0)] + (1.0/TAU)*( ((real_t*)&_fequ0)[k] - fScratch[I(i,j+k, 0)]);
						f[I(i,j+k, 1)] = fScratch[I(i,j+k, 1)] + (1.0/TAU)*( ((real_t*)&_fequ1)[k] - fScratch[I(i,j+k, 1)]);
						f[I(i,j+k, 2)] = fScratch[I(i,j+k, 2)] + (1.0/TAU)*( ((real_t*)&_fequ2)[k] - fScratch[I(i,j+k, 2)]);
						f[I(i,j+k, 3)] = fScratch[I(i,j+k, 3)] + (1.0/TAU)*( ((real_t*)&_fequ3)[k] - fScratch[I(i,j+k, 3)]);
						f[I(i,j+k, 4)] = fScratch[I(i,j+k, 4)] + (1.0/TAU)*( ((real_t*)&_fequ4)[k] - fScratch[I(i,j+k, 4)]);
						f[I(i,j+k, 5)] = fScratch[I(i,j+k, 5)] + (1.0/TAU)*( ((real_t*)&_fequ5)[k] - fScratch[I(i,j+k, 5)]);
						f[I(i,j+k, 6)] = fScratch[I(i,j+k, 6)] + (1.0/TAU)*( ((real_t*)&_fequ6)[k] - fScratch[I(i,j+k, 6)]);
						f[I(i,j+k, 7)] = fScratch[I(i,j+k, 7)] + (1.0/TAU)*( ((real_t*)&_fequ7)[k] - fScratch[I(i,j+k, 7)]);
						f[I(i,j+k, 8)] = fScratch[I(i,j+k, 8)] + (1.0/TAU)*( ((real_t*)&_fequ8)[k] - fScratch[I(i,j+k, 8)]);
					}
					else {
						f[I(i,j+k, 1)] = fScratch[I(i,j+k, 3)];
						f[I(i,j+k, 2)] = fScratch[I(i,j+k, 4)];
						f[I(i,j+k, 3)] = fScratch[I(i,j+k, 1)];
						f[I(i,j+k, 4)] = fScratch[I(i,j+k, 2)];
						f[I(i,j+k, 5)] = fScratch[I(i,j+k, 7)];
						f[I(i,j+k, 6)] = fScratch[I(i,j+k, 8)];
						f[I(i,j+k, 7)] = fScratch[I(i,j+k, 5)];
						f[I(i,j+k, 8)] = fScratch[I(i,j+k, 6)];
					}
				}

			}


		}
	}

}
#endif



void ApplySource(
	real_t * restrict f,
	const int * restrict walls)
{

	// Accelerate the flow so we have a pipe with fluid flowing along it.
	// Increase "incoming" f, decrease "outgoing" (ie, leaving the domain) down the first column
	const real_t accelParameter = ACCEL;
	const int j = 0;
	for (int i = 0; i < NX; i++) {

		if (walls[I(i,0, 0)] == 0) {
			// check none of the "outgoing" densities will end up negative. f is strictly >= 0!
			if (  (f[I(i,j, 6)] - accelParameter*OMEGA58 > 0.0)
			    &&(f[I(i,j, 3)] - accelParameter*OMEGA14 > 0.0)
			    &&(f[I(i,j, 7)] - accelParameter*OMEGA58 > 0.0) ) {

				f[I(i,j, 6)] -= accelParameter*OMEGA58;
				f[I(i,j, 3)] -= accelParameter*OMEGA14;
				f[I(i,j, 7)] -= accelParameter*OMEGA58;

				f[I(i,j, 5)] += accelParameter*OMEGA58;
				f[I(i,j, 1)] += accelParameter*OMEGA14;
				f[I(i,j, 8)] += accelParameter*OMEGA58;
			}

		}
	}

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
	real_t * restrict fScratch,
	int * restrict walls)
{
	const real_t initialf = INITIALDENSITY;

	// It is important that we initialize the arrays in parallel, with the same scheduling as the computation loops.
	// This allows memory to be allocated correctly for NUMA systems, with a "first touch" policy.

#pragma omp parallel for default(none) shared(walls) schedule(static)
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

#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
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

			fScratch[I(i,j, 0)] = 0.0;
			fScratch[I(i,j, 1)] = 0.0;
			fScratch[I(i,j, 2)] = 0.0;
			fScratch[I(i,j, 3)] = 0.0;
			fScratch[I(i,j, 4)] = 0.0;
			fScratch[I(i,j, 5)] = 0.0;
			fScratch[I(i,j, 6)] = 0.0;
			fScratch[I(i,j, 7)] = 0.0;
			fScratch[I(i,j, 8)] = 0.0;
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
