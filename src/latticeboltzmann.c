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
#define SAVELATTICEEVERY 1000
#define ACCEL 0.005
#define INITIALDENSITY 0.1


// Choose precision and vectorization
#include "prec_double_avx.h"
//#include "prec_double_sse.h"
//#include "prec_double_serial.h"
//#include "prec_float_avx.h"
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
	real_t * restrict fA,
	real_t * restrict fB,
	const char * restrict walls);

void ApplySource(
	real_t * restrict f,
	const char * restrict walls);

void StreamCollide(
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls);

#if defined(AVX) || defined(SSE)
void StreamCollideVec(
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls);
#endif

real_t ComputeReynolds(
	const real_t * restrict f,
	const char * restrict walls);

void InitializeArrays(
	real_t * restrict fA,
	real_t * restrict fB,
	char * restrict walls);

void PrintLattice(int timeStep, const real_t * restrict f);

double GetWallTime(void);



int main(void)
{
	feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

	// Allocate memory. Use _mm_malloc() for alignment. Want 32byte alignment for vector instructions.
	real_t * fA;
	real_t * fB;
	char * walls;

	int allocSize = NX * NYPADDED * NSPEEDS;
	printf("Lattice Size: %dx%d (%lf.2 MB)\n", NX, NY, (double)(allocSize*sizeof(*fA))/1024.0/1024.0);
	fA = _mm_malloc(allocSize * sizeof *fA, 32);
	fB = _mm_malloc(allocSize * sizeof *fB, 32);
	walls = _mm_malloc(NX * NYPADDED * sizeof *walls, 32);

	InitializeArrays(fA, fB, walls);


	// Begin iterations
	double timeElapsed = GetWallTime();

	for (int n = 0; n < NTIMESTEPS; n+=2) {

		if (n % PRINTSTATSEVERY == 0) {
			if (n != 0) {
				double complete = (double)n/(double)NTIMESTEPS;
				int secElap = (int)(GetWallTime()-timeElapsed);
				int secRem = (int)(secElap/complete*(1.0-complete));
				double avgbw = 2.0*n*sizeof(real_t)*NX*NY*NSPEEDS/(GetWallTime()-timeElapsed)/1024/1024/1024;
				printf("%5.2lf%%--Elapsed: %3dm%02ds, Remaining: %3dm%02ds. [Updates/s: %.3le, Update BW: ~%.3lf GB/s, GFLOPs: ~%.3lf]\n",
				       complete*100, secElap/60, secElap%60, secRem/60, secRem%60, n/(double)secElap,
				       avgbw, FLOPPERLATTICEPOINT*NX*NY*n/(double)secElap/1000.0/1000.0/1000.0);
			}
		}
		if (n % SAVELATTICEEVERY == 0) {
			PrintLattice(n, f);
		}

		// Do a timestep
		DoTimeStep(fA, fB, walls);

	}

	timeElapsed = GetWallTime() - timeElapsed;

	// End iterations


	// print final run stats
	double avgbw = 2.0*NTIMESTEPS*sizeof(real_t)*NX*NY*NSPEEDS/timeElapsed/1024/1024/1024;
	printf("100.0%%--Elapsed: %3dm%02ds,                     [Updates/s: %.3le, Update BW: ~%.3lf GB/s, GFLOPs: ~%.3lf]\n",
	       (int)timeElapsed/60, (int)timeElapsed%60, NTIMESTEPS/timeElapsed, avgbw,
	       FLOPPERLATTICEPOINT*NX*NY*NTIMESTEPS/timeElapsed/1000.0/1000.0/1000.0);
	printf("Time: %lf Re %.10le\n", timeElapsed, ComputeReynolds(fA, walls));


	// Free dynamically allocated memory
	_mm_free(fA);
	_mm_free(fB);
	_mm_free(walls);

	return EXIT_SUCCESS;
}



void DoTimeStep(
	real_t * restrict fA,
	real_t * restrict fB,
	const char * restrict walls)
{

	ApplySource(fA, walls);

#if defined(AVX) || defined(SSE)
	StreamCollideVec(0,NYVECMAX,fA, fB, walls);
	StreamCollide(NYVECMAX, NY, fA, fB, walls);
#else
	StreamCollide(0, NY, fA, fB, walls);
#endif

	ApplySource(fB, walls);

#if defined(AVX) || defined(SSE)
	StreamCollideVec(0,NYVECMAX,fB, fA, walls);
	StreamCollide(NYVECMAX, NY, fB, fA, walls);
#else
	StreamCollide(0, NY, fB, fA, walls);
#endif

}



// Combined stream and collide function, to improve caching. We read from fSrc and write updated values to fDst. This
// function is to be called with the role of fSrc and fDst switched each timestep.
void StreamCollide(
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls)
{
	// Copy to temporary array
	real_t fTmp[NSPEEDS];

#pragma omp parallel for default(none) shared(fSrc,fDst,walls) private(fTmp) schedule(static)
	for (int i = 0; i < NX; i++) {
		for (int j = jMin; j < jMax; j++) {

			// pull values from neighbouring lattice points to fTmp
			int x_u = (i + 1) % NX;
			int x_d = (i == 0) ? (NX - 1) : (i - 1);
			int y_r = (j + 1) % NY;
			int y_l = (j == 0) ? (NY - 1) : (j - 1);
			fTmp[0] = fSrc[I(i  ,j  , 0)];
			fTmp[1] = fSrc[I(i  ,y_l, 1)];
			fTmp[2] = fSrc[I(x_d,j  , 2)];
			fTmp[3] = fSrc[I(i  ,y_r, 3)];
			fTmp[4] = fSrc[I(x_u,j  , 4)];
			fTmp[5] = fSrc[I(x_d,y_l, 5)];
			fTmp[6] = fSrc[I(x_d,y_r, 6)];
			fTmp[7] = fSrc[I(x_u,y_r, 7)];
			fTmp[8] = fSrc[I(x_u,y_l, 8)];

			// bounce-back from wall
			if (walls[I(i,j, 0)] == 1) {
				fDst[I(i,j, 1)] = fTmp[3];
				fDst[I(i,j, 2)] = fTmp[4];
				fDst[I(i,j, 3)] = fTmp[1];
				fDst[I(i,j, 4)] = fTmp[2];
				fDst[I(i,j, 5)] = fTmp[7];
				fDst[I(i,j, 6)] = fTmp[8];
				fDst[I(i,j, 7)] = fTmp[5];
				fDst[I(i,j, 8)] = fTmp[6];
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
					fDst[I(i,j, s)] = fTmp[s] + (1.0/TAU)*(fequ[s] - fTmp[s]);
				}
			}

		}
	}

}



#if defined(AVX) || defined(SSE)
void StreamCollideVec(
	const int jMin,
	const int jMax,
	const real_t * restrict fSrc,
	real_t * restrict fDst,
	const char * restrict walls)
{
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

#pragma omp parallel for default(none) shared(fSrc,fDst,walls) private(fTmp) schedule(static)
	for (int i = 0; i < NX; i++) {
		// compute VECWIDTH lattice points at once
		for (int j = jMin; j < jMax; j+=VECWIDTH) {

			// pull values from neighbouring lattice points to fTmp
			for (int k = 0; k < VECWIDTH; k++) {
				int x_u = (i + 1) % NX;
				int x_d = (i == 0) ? (NX - 1) : (i - 1);
				int y_r = (j+k + 1) % NY;
				int y_l = (j+k == 0) ? (NY - 1) : (j+k - 1);
				fTmp[VECWIDTH*0+k] = fSrc[I(i  ,j+k, 0)];
				fTmp[VECWIDTH*1+k] = fSrc[I(i  ,y_l, 1)];
				fTmp[VECWIDTH*2+k] = fSrc[I(x_d,j+k, 2)];
				fTmp[VECWIDTH*3+k] = fSrc[I(i  ,y_r, 3)];
				fTmp[VECWIDTH*4+k] = fSrc[I(x_u,j+k, 4)];
				fTmp[VECWIDTH*5+k] = fSrc[I(x_d,y_l, 5)];
				fTmp[VECWIDTH*6+k] = fSrc[I(x_d,y_r, 6)];
				fTmp[VECWIDTH*7+k] = fSrc[I(x_u,y_r, 7)];
				fTmp[VECWIDTH*8+k] = fSrc[I(x_u,y_l, 8)];
			}

			_mm_prefetch(&fSrc[I(i  ,j+VECWIDTH, 0)],0);
			_mm_prefetch(&fSrc[I(i  ,j+VECWIDTH, 1)],0);
			_mm_prefetch(&fSrc[I(i-1,j+VECWIDTH, 2)],0);
			_mm_prefetch(&fSrc[I(i  ,j+VECWIDTH, 3)],0);
			_mm_prefetch(&fSrc[I(i+1,j+VECWIDTH, 4)],0);
			_mm_prefetch(&fSrc[I(i-1,j+VECWIDTH, 5)],0);
			_mm_prefetch(&fSrc[I(i-1,j+VECWIDTH, 6)],0);
			_mm_prefetch(&fSrc[I(i+1,j+VECWIDTH, 7)],0);
			_mm_prefetch(&fSrc[I(i+1,j+VECWIDTH, 8)],0);

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
				wallsSum += walls[I(i,j+k, 0)];
			}
			if (wallsSum == 0) {
				VECTOR_STORE(&fDst[I(i,j, 0)], VECTOR_ADD(_f0, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ0,_f0))));
				VECTOR_STORE(&fDst[I(i,j, 1)], VECTOR_ADD(_f1, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ1,_f1))));
				VECTOR_STORE(&fDst[I(i,j, 2)], VECTOR_ADD(_f2, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ2,_f2))));
				VECTOR_STORE(&fDst[I(i,j, 3)], VECTOR_ADD(_f3, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ3,_f3))));
				VECTOR_STORE(&fDst[I(i,j, 4)], VECTOR_ADD(_f4, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ4,_f4))));
				VECTOR_STORE(&fDst[I(i,j, 5)], VECTOR_ADD(_f5, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ5,_f5))));
				VECTOR_STORE(&fDst[I(i,j, 6)], VECTOR_ADD(_f6, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ6,_f6))));
				VECTOR_STORE(&fDst[I(i,j, 7)], VECTOR_ADD(_f7, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ7,_f7))));
				VECTOR_STORE(&fDst[I(i,j, 8)], VECTOR_ADD(_f8, VECTOR_MUL(_ITAU,VECTOR_SUB(_fequ8,_f8))));
			}
			else {

				for (int k = 0; k < VECWIDTH; k++) {
					if (walls[I(i,j+k, 0)] == 0) {
						fDst[I(i,j+k, 0)] = fTmp[VECWIDTH*0+k] + (1.0/TAU)*( ((real_t*)&_fequ0)[k] - fTmp[VECWIDTH*0+k]);
						fDst[I(i,j+k, 1)] = fTmp[VECWIDTH*1+k] + (1.0/TAU)*( ((real_t*)&_fequ1)[k] - fTmp[VECWIDTH*1+k]);
						fDst[I(i,j+k, 2)] = fTmp[VECWIDTH*2+k] + (1.0/TAU)*( ((real_t*)&_fequ2)[k] - fTmp[VECWIDTH*2+k]);
						fDst[I(i,j+k, 3)] = fTmp[VECWIDTH*3+k] + (1.0/TAU)*( ((real_t*)&_fequ3)[k] - fTmp[VECWIDTH*3+k]);
						fDst[I(i,j+k, 4)] = fTmp[VECWIDTH*4+k] + (1.0/TAU)*( ((real_t*)&_fequ4)[k] - fTmp[VECWIDTH*4+k]);
						fDst[I(i,j+k, 5)] = fTmp[VECWIDTH*5+k] + (1.0/TAU)*( ((real_t*)&_fequ5)[k] - fTmp[VECWIDTH*5+k]);
						fDst[I(i,j+k, 6)] = fTmp[VECWIDTH*6+k] + (1.0/TAU)*( ((real_t*)&_fequ6)[k] - fTmp[VECWIDTH*6+k]);
						fDst[I(i,j+k, 7)] = fTmp[VECWIDTH*7+k] + (1.0/TAU)*( ((real_t*)&_fequ7)[k] - fTmp[VECWIDTH*7+k]);
						fDst[I(i,j+k, 8)] = fTmp[VECWIDTH*8+k] + (1.0/TAU)*( ((real_t*)&_fequ8)[k] - fTmp[VECWIDTH*8+k]);
					}
					else {
						fDst[I(i,j+k, 1)] = fTmp[VECWIDTH*3+k];
						fDst[I(i,j+k, 2)] = fTmp[VECWIDTH*4+k];
						fDst[I(i,j+k, 3)] = fTmp[VECWIDTH*1+k];
						fDst[I(i,j+k, 4)] = fTmp[VECWIDTH*2+k];
						fDst[I(i,j+k, 5)] = fTmp[VECWIDTH*7+k];
						fDst[I(i,j+k, 6)] = fTmp[VECWIDTH*8+k];
						fDst[I(i,j+k, 7)] = fTmp[VECWIDTH*5+k];
						fDst[I(i,j+k, 8)] = fTmp[VECWIDTH*6+k];
					}
				}

			}


		}
	}

}
#endif



void ApplySource(
	real_t * restrict f,
	const char * restrict walls)
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
	const char * restrict walls)
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
	real_t * restrict fA,
	real_t * restrict fB,
	char * restrict walls)
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

#pragma omp parallel for default(none) shared(fA,fB) schedule(static)
	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			fA[I(i,j, 0)] = initialf * OMEGA0;
			fA[I(i,j, 1)] = initialf * OMEGA14;
			fA[I(i,j, 2)] = initialf * OMEGA14;
			fA[I(i,j, 3)] = initialf * OMEGA14;
			fA[I(i,j, 4)] = initialf * OMEGA14;
			fA[I(i,j, 5)] = initialf * OMEGA58;
			fA[I(i,j, 6)] = initialf * OMEGA58;
			fA[I(i,j, 7)] = initialf * OMEGA58;
			fA[I(i,j, 8)] = initialf * OMEGA58;

			fB[I(i,j, 0)] = 0.0;
			fB[I(i,j, 1)] = 0.0;
			fB[I(i,j, 2)] = 0.0;
			fB[I(i,j, 3)] = 0.0;
			fB[I(i,j, 4)] = 0.0;
			fB[I(i,j, 5)] = 0.0;
			fB[I(i,j, 6)] = 0.0;
			fB[I(i,j, 7)] = 0.0;
			fB[I(i,j, 8)] = 0.0;
		}
	}


}



void PrintLattice(int timeStep, const real_t * restrict f)
{
	char filename[100];
	sprintf(filename,"data/%d.csv",timeStep);
	FILE *fp = fopen(filename,"w");

	if (fp == NULL) printf("Error opening file %s\n", filename);

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
