// Kernels.cuh
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "Constants.h"

// Kernels
__global__ void collisionKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_w,double d_Omega,double d_OmegaPrima);
__global__ void streamKernel(double *d_f,int *d_Cx,int *d_Cy);
/*__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e, double *d_h);
__global__ void computeFeqKernel(double *d_rho,double *d_jx,double *d_jy,double *d_feq);
__global__ void computeErrorsKernel(double* d_rho,double* d_jx,double* d_jy,double* d_rho_e,double* d_h,double* d_feq,double* d_f,double* d_mass_err,double* d_momX_err,double* d_momY_err,double* d_energy_err,double* d_entropy_diff,double* d_max_mass,double* d_max_momX,double* d_max_momY,double* d_max_energy,double* d_min_entropy,double* d_max_entropy);*/

#endif
