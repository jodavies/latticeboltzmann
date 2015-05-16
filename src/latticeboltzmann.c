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
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define __USE_GNU
#include <fenv.h>



// Choose precision
#define DOUBLEPREC 1
#define MPI_REAL_T MPI_DOUBLE
	typedef double real_t;
//	#define VECWIDTH 1
	#define AVX 1
		#define VECWIDTH 4
//	#define SSE 1
//		#define VECWIDTH 2

//#define SINGLEPREC 1
//	typedef float real_t;
//	#define AVX 1
//		#define VECWIDTH 8
//	#define SSE 1
//		#define VECWIDTH 4
//#define VECWIDTH 1



// d2q9 fixed parameters
#define NSPEEDS 9
#define OMEGA0  (4.0/9.0)
#define OMEGA14 (1.0/9.0)
#define OMEGA58 (1.0/36.0)
#define TAU 0.75
#define CSQ (1.0)

// Boundary condition. Wrap-around (periodic) or not (fluid flows out of the domain)
#define WRAPAROUND 1

// variable parameters
#define NX 400
#define NY 2000
#define NTIMESTEPS 200000
#define PRINTEVERY 1000
#define ACCEL 0.005
#define INITIALDENSITY 0.1

// Macro for array indexing
#define I(i,j, speed) ((speed)*NX*NY + (i)*NY + (j))



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
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls);

void CollideAVX(
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls);

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

	int allocSize = NX * NY * NSPEEDS;

	f = _mm_malloc(allocSize * sizeof *f, 32);
	fScratch = _mm_malloc(allocSize * sizeof *f, 32);
	walls = _mm_malloc(NX * NY * sizeof *walls, 32);

	InitializeArrays(f, fScratch, walls);


	// Begin iterations
	double timeElapsed = GetWallTime();

	for (int n = 0; n < NTIMESTEPS; n++) {
		if (n % PRINTEVERY == 0) {
			PrintLattice(n, f);
			if (n != 0) {
				double complete = (double)n/(double)NTIMESTEPS;
				int secElap = (int)(GetWallTime()-timeElapsed);
				int secRem = (int)(secElap/complete*(1.0-complete));
				double avgbw = n*4*sizeof(real_t)*NX*NY*NSPEEDS/(GetWallTime()-timeElapsed)/1024/1024/1024;
				printf("%6.3lf%% -- Elapsed: %dm%ds, Remaining: %dm%ds.  (Update Bandwidth: ~%.2lf GB/s)\n",
				       complete*100, secElap/60, secElap%60, secRem/60, secRem%60, avgbw);
			}
		}
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

#ifdef AVX
	CollideAVX(f, fScratch, walls);
#else
	Collide(f, fScratch, walls);
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
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i,2,1)], &f[I(i,1,1)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i+1,1,2)], &f[I(i,1,2)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i,0,3)], &f[I(i,1,3)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i-1,1,4)], &f[I(i,1,4)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i+1,2,5)], &f[I(i,1,5)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i+1,0,6)], &f[I(i,1,6)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i-1,0,7)], &f[I(i,1,7)], (NY-2)* sizeof *f);
	}
#pragma omp parallel for default(none) shared(f,fScratch) schedule(static)
	for (int i = 1; i < NX-1; i++) {
		memcpy(&fScratch[I(i-1,2,8)], &f[I(i,1,8)], (NY-2)* sizeof *f);
	}

}



void Collide(
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls)
{

#pragma omp parallel for default(none) shared(f,fScratch,walls) schedule(static)
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {

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
void CollideAVX(
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls)
{
	const __m256d _one = _mm256_set1_pd(1.0);
	const __m256d _three = _mm256_set1_pd(3.0);
	const __m256d _threeOtwo = _mm256_set1_pd(3.0/2.0);
	const __m256d _nineOtwo = _mm256_set1_pd(9.0/2.0);
	const __m256d _ICSQ = _mm256_set1_pd(1.0/CSQ);
	const __m256d _OMEGA0 = _mm256_set1_pd(OMEGA0);
	const __m256d _OMEGA14 = _mm256_set1_pd(OMEGA14);
	const __m256d _OMEGA58 = _mm256_set1_pd(OMEGA58);

#pragma omp parallel for default(none) shared(f,fScratch,walls) schedule(static)
	for (int i = 0; i < NX; i++) {
		// compute VECWIDTH lattice points at once
		for (int j = 0; j < NY; j+=VECWIDTH) {

			__m256d _density = _mm256_set1_pd(0.0);

			__m256d _f0 = _mm256_load_pd(&fScratch[I(i,j, 0)]);
			__m256d _f1 = _mm256_load_pd(&fScratch[I(i,j, 1)]);
			__m256d _f2 = _mm256_load_pd(&fScratch[I(i,j, 2)]);
			__m256d _f3 = _mm256_load_pd(&fScratch[I(i,j, 3)]);
			__m256d _f4 = _mm256_load_pd(&fScratch[I(i,j, 4)]);
			__m256d _f5 = _mm256_load_pd(&fScratch[I(i,j, 5)]);
			__m256d _f6 = _mm256_load_pd(&fScratch[I(i,j, 6)]);
			__m256d _f7 = _mm256_load_pd(&fScratch[I(i,j, 7)]);
			__m256d _f8 = _mm256_load_pd(&fScratch[I(i,j, 8)]);

			_density = _mm256_add_pd(_density, _f0);
			_density = _mm256_add_pd(_density, _f1);
			_density = _mm256_add_pd(_density, _f2);
			_density = _mm256_add_pd(_density, _f3);
			_density = _mm256_add_pd(_density, _f4);
			_density = _mm256_add_pd(_density, _f5);
			_density = _mm256_add_pd(_density, _f6);
			_density = _mm256_add_pd(_density, _f7);
			_density = _mm256_add_pd(_density, _f8);

			__m256d _u_x = _mm256_add_pd(_f6, _f2);
			_u_x = _mm256_add_pd(_u_x, _f5);
			_u_x = _mm256_sub_pd(_u_x, _f7);
			_u_x = _mm256_sub_pd(_u_x, _f4);
			_u_x = _mm256_sub_pd(_u_x, _f8);
			_u_x = _mm256_div_pd(_u_x, _density);

			__m256d _u_y = _mm256_add_pd(_f5, _f1);
			_u_y = _mm256_add_pd(_u_y, _f8);
			_u_y = _mm256_sub_pd(_u_y, _f6);
			_u_y = _mm256_sub_pd(_u_y, _f3);
			_u_y = _mm256_sub_pd(_u_y, _f7);
			_u_y = _mm256_div_pd(_u_y, _density);

			__m256d _uDotu = _mm256_add_pd(_mm256_mul_pd(_u_x,_u_x),_mm256_mul_pd(_u_y,_u_y));

			// Directional velocity components e_i dot u
			__m256d _u1 = _u_y;
			__m256d _u2 = _u_x;
			__m256d _u3 = _mm256_mul_pd(_mm256_set1_pd(-1.0),_u_y);
			__m256d _u4 = _mm256_mul_pd(_mm256_set1_pd(-1.0),_u_x);
			__m256d _u5 = _mm256_add_pd(_u_x,_u_y);
			__m256d _u6 = _mm256_sub_pd(_u_x,_u_y);
			__m256d _u7 = _mm256_mul_pd(_mm256_set1_pd(-1.0),_mm256_add_pd(_u_x,_u_y));
			__m256d _u8 = _mm256_mul_pd(_mm256_set1_pd(-1.0),_mm256_sub_pd(_u_x,_u_y));


			// equilibrium density
			__m256d _fequ0 = _mm256_sub_pd(_one, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ1 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u1,_ICSQ)));
			        _fequ1 = _mm256_add_pd(_fequ1, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u1,_mm256_mul_pd(_u1,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ1 = _mm256_sub_pd(_fequ1, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ2 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u2,_ICSQ)));
			        _fequ2 = _mm256_add_pd(_fequ2, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u2,_mm256_mul_pd(_u2,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ2 = _mm256_sub_pd(_fequ2, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ3 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u3,_ICSQ)));
			        _fequ3 = _mm256_add_pd(_fequ3, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u3,_mm256_mul_pd(_u3,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ3 = _mm256_sub_pd(_fequ3, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ4 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u4,_ICSQ)));
			        _fequ4 = _mm256_add_pd(_fequ4, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u4,_mm256_mul_pd(_u4,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ4 = _mm256_sub_pd(_fequ4, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ5 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u5,_ICSQ)));
			        _fequ5 = _mm256_add_pd(_fequ5, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u5,_mm256_mul_pd(_u5,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ5 = _mm256_sub_pd(_fequ5, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ6 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u6,_ICSQ)));
			        _fequ6 = _mm256_add_pd(_fequ6, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u6,_mm256_mul_pd(_u6,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ6 = _mm256_sub_pd(_fequ6, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ7 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u7,_ICSQ)));
			        _fequ7 = _mm256_add_pd(_fequ7, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u7,_mm256_mul_pd(_u7,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ7 = _mm256_sub_pd(_fequ7, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			__m256d _fequ8 = _mm256_add_pd(_one, _mm256_mul_pd(_three,_mm256_mul_pd(_u8,_ICSQ)));
			        _fequ8 = _mm256_add_pd(_fequ8, _mm256_mul_pd(_nineOtwo,_mm256_mul_pd(_u8,_mm256_mul_pd(_u8,_mm256_mul_pd(_ICSQ,_ICSQ)))));
			        _fequ8 = _mm256_sub_pd(_fequ8, _mm256_mul_pd(_threeOtwo,_mm256_mul_pd(_uDotu,_ICSQ)));
			// now multiply by omega and density
			_fequ0 = _mm256_mul_pd(_fequ0, _mm256_mul_pd(_OMEGA0, _density));
			_fequ1 = _mm256_mul_pd(_fequ1, _mm256_mul_pd(_OMEGA14, _density));
			_fequ2 = _mm256_mul_pd(_fequ2, _mm256_mul_pd(_OMEGA14, _density));
			_fequ3 = _mm256_mul_pd(_fequ3, _mm256_mul_pd(_OMEGA14, _density));
			_fequ4 = _mm256_mul_pd(_fequ4, _mm256_mul_pd(_OMEGA14, _density));
			_fequ5 = _mm256_mul_pd(_fequ5, _mm256_mul_pd(_OMEGA58, _density));
			_fequ6 = _mm256_mul_pd(_fequ6, _mm256_mul_pd(_OMEGA58, _density));
			_fequ7 = _mm256_mul_pd(_fequ7, _mm256_mul_pd(_OMEGA58, _density));
			_fequ8 = _mm256_mul_pd(_fequ8, _mm256_mul_pd(_OMEGA58, _density));


			// relaxation: check if cell is blocked!
			int wallsSum = 0;
			for (int k = 0; k < VECWIDTH; k++) {
				wallsSum += walls[I(i,j+k, 0)];
			}
			if (wallsSum == 0) {
				_mm256_store_pd(&f[I(i,j, 0)], _mm256_add_pd(_f0, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ0,_f0))));
				_mm256_store_pd(&f[I(i,j, 1)], _mm256_add_pd(_f1, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ1,_f1))));
				_mm256_store_pd(&f[I(i,j, 2)], _mm256_add_pd(_f2, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ2,_f2))));
				_mm256_store_pd(&f[I(i,j, 3)], _mm256_add_pd(_f3, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ3,_f3))));
				_mm256_store_pd(&f[I(i,j, 4)], _mm256_add_pd(_f4, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ4,_f4))));
				_mm256_store_pd(&f[I(i,j, 5)], _mm256_add_pd(_f5, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ5,_f5))));
				_mm256_store_pd(&f[I(i,j, 6)], _mm256_add_pd(_f6, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ6,_f6))));
				_mm256_store_pd(&f[I(i,j, 7)], _mm256_add_pd(_f7, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ7,_f7))));
				_mm256_store_pd(&f[I(i,j, 8)], _mm256_add_pd(_f8, _mm256_mul_pd(_mm256_set1_pd(1.0/TAU),_mm256_sub_pd(_fequ8,_f8))));
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
