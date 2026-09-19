// KernelsManager.cu
#include "KernelsManager.h"

KernelsManager::KernelsManager(void){
  // 2D kernel launch configs
  blockSize2D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,1);
  gridSize2D  = dim3((Lx+blockSize2D.x-1)/blockSize2D.x,(Ly+blockSize2D.y-1)/blockSize2D.y,1);
  // 3D kernel launch configs
  blockSize3D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,Q);
  gridSize3D  = dim3((Lx+blockSize3D.x-1)/blockSize3D.x,(Ly+blockSize3D.y-1)/blockSize3D.y,(Q+blockSize3D.z-1)/blockSize3D.z);
}
KernelsManager::~KernelsManager(void){
}

void KernelsManager::launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset){
  computeMacrosKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
void KernelsManager::launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq){
  computeFeqKernel<<<gridSize3D,blockSize3D>>>((double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_feq);
  KERNEL_CHECK();
  SYNC_CHECK();
}
  
void KernelsManager::launchComputeCollisionErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff){
  computeCollisionErrorsKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(double*)d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchFindMinMax(double *d_data,double *d_min,double *d_max,int offset){
  findMinMaxKernel<<<gridSize2D,blockSize2D>>>((double*)d_data,(double*)d_min,(double*)d_max,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchRenderAndFill(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale){
  renderAndfillKernel<<<gridSize2D,blockSize2D>>>((uchar4*)d_color,(double*)vbo_positions_ptr,(double*)vbo_uv_ptr,(double*)d_data,(double)center,(double)scale);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchCollision(double *d_f,double *d_feq,int offset){
  collisionKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(double*)d_feq,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchComputeLocalErrors(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset){
  computeLocalErrorsKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(double*)d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
void KernelsManager::launchMarkEntropyViolations(double *d_entropy_diff,int *d_violation_mask,int offset){
  markEntropyViolationsKernel<<<gridSize2D,blockSize2D>>>((double*)d_entropy_diff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
void KernelsManager::launchFindOmegaEff(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  findOmegaEffKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
void KernelsManager::launchEntropicCollision(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  entropicCollisionKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(double*)d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchStream(double *d_f,int offset){
  streamKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
