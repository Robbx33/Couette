// Kernels.cuh
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "Constants.h"

// Kernels
__global__ void collisionKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_w);
__global__ void streamKernel(double *d_f,int *d_Cx,int *d_Cy);
__global__ void computeMacrosKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e, double *d_h);
__global__ void computeFeqKernel(int *d_Cx,int *d_Cy,double *d_w,double *d_rho,double *d_jx,double *d_jy,double *d_feq);
__global__ void computeErrorsKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff);
__global__ void renderAndfillKernel(uchar4 *d_color,double *d_positions,double *d_uv,double *d_rho);

#endif
