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
	#define VECWIDTH 1
//	#define AVX 1
//		#define VECWIDTH 4
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
#define TAU 0.7
#define CSQ (1.0)

// Boundary condition. Wrap-around (periodic) or not (fluid flows out of the domain)
#define WRAPAROUND 1

// variable parameters
#define NX 400
#define NY 1200
#define NTIMESTEPS 20000
#define PRINTEVERY 100
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
		if (n % PRINTEVERY == 0) PrintLattice(n, f);
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

	Collide(f, fScratch, walls);

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
	for (int i = 1; i < NX-1; i++) {
		for (int j = 1; j < NY-1; j++) {

			int x_u = (i + 1);
			int x_d = (i - 1);
			int y_r = (j + 1);
			int y_l = (j - 1);

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
	}

}



void Collide(
	real_t * restrict f,
	const real_t * restrict fScratch,
	const int * restrict walls)
{

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
		walls[I(i,50, 0)] = 1;
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
	sprintf(filename,"/home/josh/projects/latticeboltzmann/data/%d.csv",timeStep);
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
