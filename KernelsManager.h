// KernelManager.h
#ifndef KERNELSMANAGER_H
#define KERNELSMANAGER_H

#include "Constants.h"

class KernelsManager{
 private:
  /*int nx,ny;*/
  dim3 blockSize2D,gridSize2D;
  dim3 blockSize3D,gridSize3D;
 public:
  KernelsManager(int Lx,int Ly);
  ~KernelsManager();

  // Launch kernels
  void launchCollision(double *d_f,int *d_Cx,int *d_Cy,double *d_w,double d_Omega,double d_OmegaPrima);
  void launchStream(double *d_f,int *d_Cx,int *d_Cy);
    /*void launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h);
  void launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq);
  void launchComputeErrors(double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_f,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_entropy_diff);

  //Reduction helpers
  void reduceMax(double *d_input,double *d_output,int size);
  void reduceMin(double *d_input,double *d_output,int size);*/
  };

#endif
