// KernelManager.h
#ifndef KERNELSMANAGER_H
#define KERNELSMANAGER_H

#include "Constants.h"
#include "error.h"
#include "Kernels.cuh"
#include "Visualizer.h"

class KernelsManager{
 private:
  Visualizer *viz;
  
  dim3 blockSize2D,gridSize2D;
  dim3 blockSize3D,gridSize3D;

  double *h_min_temp;
  double *h_max_temp;
  double *d_min_temp;
  double *d_max_temp;
 public:
  KernelsManager(Visualizer *Roberto);
  ~KernelsManager(void);

  // Launch kernels
  void launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h);
  void launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq);

  void launchComputeCollisionErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff);
  void findMinMax(double *d_data,double *min_val,double *max_val);
  void launchRenderAndFill(uchar4 *d_color,double *d_positions,double *d_uv,double *d_data,int t);
  
  void launchCollision(double *d_f,double *d_feq);

  void launchComputeLocalErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff);
  void launchMarkEntropyViolationsKernel(double *d_hfhfeq_diff,int *d_violation_mask);
  
  void launchStream(double *d_f);

  
  };

#endif
