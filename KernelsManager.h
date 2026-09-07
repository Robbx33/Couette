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
  void launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset);
  void launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq);
  
  void launchComputeCollisionErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff);
  void findMinMax(double *d_data,double *min_val,double *max_val);
  void launchRenderAndFill(uchar4 *d_color,double *d_positions,double *d_uv,double *d_data,int t);
  
  void launchCollision(double *d_f,double *d_feq);

  void launchComputeLocalErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_dif,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset);
  void launchMarkEntropyViolations(double *d_energy_diff,int *d_violation_mask,int offset);
  void launchFindOmegaEff(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  void launchEntropicCollision(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  
  void launchStream(double *d_f);

  
  };

#endif
