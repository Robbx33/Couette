// Kernels.cuh
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "Constants.h"

__constant__ int d_Cx[Q] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
__constant__ int d_Cy[Q] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
__constant__ double d_w[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

// Kernels
__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e, double *d_h,int offset);
__global__ void computeFeqKernel(double *d_rho,double *d_jx,double *d_jy,double *d_feq);

__global__ void computeCollisionErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff);
__global__ void findMinMaxKernel(double *d_data,double *d_min,double *d_max);
__global__ void renderAndfillKernel(uchar4 *d_color,double *d_positions,double *d_uv,double *d_data,double offset,double scale);

__global__ void collisionKernel(double *d_f,double *d_feq);

__global__ void computeLocalErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset);
__global__ void markEntropyViolationsKernel(double *d_entropy_diff,int *d_violation_mask,int offset);
__global__ void findOmegaEffKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
__global__ void entropicCollisionKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);

__global__ void streamKernel(double *d_f);

#endif
