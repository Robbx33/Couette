// KernelsManager.cu
#include "KernelsManager.h"

KernelsManager::KernelsManager(){
  // 2D kernel launch configs
  blockSize2D = dim3(4,2,1);
  gridSize2D  = dim3((Lx+blockSize2D.x-1)/blockSize2D.x,(Ly+blockSize2D.y-1)/blockSize2D.y,1);

  // 3D kernel launch configs
  blockSize3D = dim3(4,2,Q);
  gridSize3D  = dim3((Lx+blockSize3D.x-1)/blockSize3D.x,(Ly+blockSize3D.y-1)/blockSize3D.y,(Q+blockSize3D.z-1)/blockSize3D.z);
}

KernelsManager::~KernelsManager(){

}

void KernelsManager::launchCollision(double *d_f,int *d_Cx,int *d_Cy,double *d_w){
  collisionKernel<<<gridSize3D,blockSize3D>>>((double*) d_f,(int*) d_Cx,(int*) d_Cy,(double*) d_w);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchStream(double *d_f,int *d_Cx,int *d_Cy){
  streamKernel<<<gridSize3D,blockSize3D>>>((double*) d_f,(int*) d_Cx,(int*) d_Cy);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchComputeMacros(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h){
  computeMacrosKernel<<<gridSize2D,blockSize2D>>>((double*) d_f,(int*) d_Cx,(int*) d_Cy,(double*) d_rho,(double*) d_jx,(double*) d_jy,(double*) d_rho_e,(double*) d_h);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchComputeFeq(int *d_Cx,int *d_Cy,double *d_w,double *d_rho,double *d_jx,double *d_jy,double *d_feq){
  computeFeqKernel<<<gridSize3D,blockSize3D>>>((int*) d_Cx,(int*) d_Cy,(double*) d_w,(double*) d_rho,(double*) d_jx,(double*) d_jy,(double*) d_feq);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchComputeErrors(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff){
  computeErrorsKernel<<<gridSize2D,blockSize2D>>>((double*) d_f,(int*) d_Cx,(int*) d_Cy,(double*) d_rho,(double*) d_jx,(double*) d_jy,(double*) d_rho_e,(double*) d_h,(double*) d_feq,(double*) d_mass_err,(double*) d_momX_err,(double*) d_momY_err,(double*) d_energy_err,(double*) d_hfhfeq_diff);
  KERNEL_CHECK();
  SYNC_CHECK();
}

void KernelsManager::launchRender(uchar4 *d_buffer,double *d_rho){
  renderKernel<<<gridSize2D,blockSize2D>>>((uchar4*) d_buffer,(double*) d_rho);
  KERNEL_CHECK();
  SYNC_CHECK();
}
