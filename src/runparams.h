// Choose precision: for DP set DOUBLEPREC to 1, otherwise SP
#define DOUBLEPREC 0

// precision related settings
#if DOUBLEPREC == 1
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	typedef double real_t;
	#define ALIGNREQUIREMENT 32
#else
	typedef float real_t;
	#define ALIGNREQUIREMENT 64
#endif

// platform and device
#define PLATFORM 0
#define DEVICE 0
#define LOCALSIZE 128

// run behaviour
#define PRINTSTATSEVERY 1000
#define SAVELATTICE 0
#define SAVELATTICEEVERY 1000

// problem-specific parameters
#define NTIMESTEPS 10000
#define NX 400
#define NY 2000
#define TAU 0.7
#define CSQ 1.0
#define ACCEL 0.005
#define INITIALDENSITY 0.1

// d2q9 fixed parameters
#define NSPEEDS 9
#define OMEGA0  (4.0/9.0)
#define OMEGA14 (1.0/9.0)
#define OMEGA58 (1.0/36.0)

// (memory alignment: GTX960 seems to do best if array rows are 256byte aligned.
//  that is, to the nearest 64 floats or 32 doubles
//  -- this is the alignment used by cudaMalloc, also
//  R9 280X prefers this alignment also)
#define NYPADDED (ALIGNREQUIREMENT*((NY-1)/ALIGNREQUIREMENT)+ALIGNREQUIREMENT)
//#define NYPADDED NY

// Macro for array indexing. Memory is contiguous in the j index.
#define I(i,j, speed) ((speed)*NX*NYPADDED + (i)*NYPADDED + (j))

// For approximate GFLOPs report: we do ~124 FLOP per lattice point, obtained simply by counting ADD,SUB,MUL,DIV
// intructions in CollideVec.
#define FLOPPERLATTICEPOINT (124.0)
