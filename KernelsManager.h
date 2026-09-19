// KernelManager.h
#ifndef KERNELSMANAGER_H
#define KERNELSMANAGER_H

#include "Constants.h"
#include "error.h"
#include "Kernels.cuh"

class KernelsManager{
 private:
  dim3 blockSize2D,gridSize2D;
  dim3 blockSize3D,gridSize3D;
 public:
  KernelsManager(void);
  ~KernelsManager(void);

  // Launch kernels
  void launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset);
  void launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq);
  
  void launchComputeCollisionErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff);
  void launchFindMinMax(double *d_data,double *d_min,double *d_max,int offset);
  void launchRenderAndFill(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale);
  
  void launchCollision(double *d_f,double *d_feq,int offset);

  void launchComputeLocalErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_dif,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset);
  void launchMarkEntropyViolations(double *d_energy_diff,int *d_violation_mask,int offset);
  void launchFindOmegaEff(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  void launchEntropicCollision(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  
  void launchStream(double *d_f,int offset);
  };

#endif
