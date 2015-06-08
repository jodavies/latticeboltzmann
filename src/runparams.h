// precision
#define DOUBLEPREC 1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	typedef double real_t;
//#define SINGLEPREC 1
//	typedef float real_t;

// platform and device
#define PLATFORM 0
#define DEVICE 1

// d2q9 fixed parameters
#define NSPEEDS 9
#define OMEGA0  (4.0/9.0)
#define OMEGA14 (1.0/9.0)
#define OMEGA58 (1.0/36.0)

// variable parameters
#define NX 400
#define NY 2000
#define TAU 0.7
#define CSQ (1.0)

#define NTIMESTEPS 10000
#define PRINTSTATSEVERY 1000
#define SAVELATTICEEVERY 300000
#define ACCEL 0.005
#define INITIALDENSITY 0.1


// Macro for array indexing. We need the array stride to be such that we have correct
// alignment for vector loads/stores on rows after the first.
//#define NYPADDED (ALIGNREQUIREMENT*((NY-1)/ALIGNREQUIREMENT)+ALIGNREQUIREMENT)
// align to 32 bytes = 4 doubles or 8 floats
#define NYPADDED NY
#define I(i,j, speed) ((speed)*NX*NYPADDED + (i)*NYPADDED + (j))

// Also store the multiple of VECWIDTH below NY, since the vectorized functions need to terminate here,
// possibly with a scalar function cleaning up the "extra".
//#define NYVECMAX (VECWIDTH*(NY/VECWIDTH))

// For approximate GFLOPs report: we do ~124 FLOP per lattice point, obtained simply by counting ADD,SUB,MUL,DIV
// intructions in CollideVec.
#define FLOPPERLATTICEPOINT (124.0)
