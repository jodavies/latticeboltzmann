#include "src/runparams.h"


__kernel void ApplySourceKernel(__global real_t *f,
                                __global const int *walls)
{
	// fetch ids
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i < NX) {

		if (j == 0) {
			if (walls[I(i,0, 0)] == 0) {
				if (  (f[I(i,0, 6)] - ACCEL*OMEGA58 > 0.0)
					 &&(f[I(i,0, 3)] - ACCEL*OMEGA14 > 0.0)
					 &&(f[I(i,0, 7)] - ACCEL*OMEGA58 > 0.0) ) {

					f[I(i,0, 6)] -= ACCEL*OMEGA58;
					f[I(i,0, 3)] -= ACCEL*OMEGA14;
					f[I(i,0, 7)] -= ACCEL*OMEGA58;

					f[I(i,0, 5)] += ACCEL*OMEGA58;
					f[I(i,0, 1)] += ACCEL*OMEGA14;
					f[I(i,0, 8)] += ACCEL*OMEGA58;
				}

			}
		}

	}

}



__kernel void StreamCollideKernel(__global const real_t *fSrc,
                                  __global real_t *fDst,
                                  __global const int *walls)
{
	// fetch ids
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i < NX) {
	if (j < NY) {

		real_t fTmp[NSPEEDS];

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
