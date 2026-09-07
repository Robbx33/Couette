// KernelsManager.cu
#include "KernelsManager.h"

KernelsManager::KernelsManager(Visualizer *Roberto){
  viz = Roberto;
  // 2D kernel launch configs
  blockSize2D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,1);
  gridSize2D  = dim3((Lx+blockSize2D.x-1)/blockSize2D.x,(Ly+blockSize2D.y-1)/blockSize2D.y,1);
  // 3D kernel launch configs
  blockSize3D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,Q);
  gridSize3D  = dim3((Lx+blockSize3D.x-1)/blockSize3D.x,(Ly+blockSize3D.y-1)/blockSize3D.y,(Q+blockSize3D.z-1)/blockSize3D.z);
  
  h_min_temp = (double*)malloc(gridSize2D.x*gridSize2D.y*sizeof(double));
  h_max_temp = (double*)malloc(gridSize2D.x*gridSize2D.y*sizeof(double));
  CUDA_CHECK(cudaMalloc((void**)&d_min_temp,gridSize2D.x*gridSize2D.y*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_max_temp,gridSize2D.x*gridSize2D.y*sizeof(double)));
}

KernelsManager::~KernelsManager(void){
  free(h_min_temp);
  free(h_max_temp);
  cudaFree(d_min_temp);
  cudaFree(d_max_temp);
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

void KernelsManager::findMinMax(double *d_data,double *min_val,double *max_val){
  findMinMaxKernel<<<gridSize2D,blockSize2D>>>((double*)d_data,(double*)d_min_temp,(double*)d_max_temp);
  KERNEL_CHECK();
  SYNC_CHECK();
  
  CUDA_CHECK(cudaMemcpy((void*)(h_min_temp+0),(const void*)(d_min_temp+0),(size_t)gridSize2D.x*gridSize2D.y*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_max_temp+0),(const void*)(d_max_temp+0),(size_t)gridSize2D.x*gridSize2D.y*sizeof(double),cudaMemcpyDeviceToHost));
  
  *min_val = *(h_min_temp+0);
  *max_val = *(h_max_temp+0);
  for(int idx=0;idx<gridSize2D.x*gridSize2D.y;idx++){
    if(*(h_min_temp+idx)<*min_val){
      *min_val = *(h_min_temp+idx);
    }
    if(*(h_max_temp+idx)>*max_val){
      *max_val = *(h_max_temp+idx);
    }
  }
}

void KernelsManager::launchRenderAndFill(uchar4 *d_color,double *d_positions,double *d_uv,double *d_data,int t){
  double min_val,max_val;
  findMinMax((double*)d_data,(double*)&min_val,(double*)&max_val);
  /*
    offset = the center of your data range
    scale = the radius of your data range (half the width)
    (min_val) → 5.0 (offset) → 7.5 (scale) → 2.5 (max_val) → 10.0
  */
  double offset = (min_val+max_val)/2.0;
  double scale = (max_val-min_val)/2.0;
  if(scale<1e-30){
    scale = 1.0;
  }
  
  viz->copyColorToTexture((uchar4*)d_color);
  viz->mapVBOs((double**)&d_positions,(double**)&d_uv);
  renderAndfillKernel<<<gridSize2D,blockSize2D>>>((uchar4*)d_color,(double*)d_positions,(double*)d_uv,(double*)d_data,(double)offset,(double)scale);
  KERNEL_CHECK();
  SYNC_CHECK();
  viz->unmapVBOs();
  viz->display((int)t,(double)min_val,(double)max_val);
}

void KernelsManager::launchCollision(double *d_f,double *d_feq){
  collisionKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(double*)d_feq);
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

void KernelsManager::launchStream(double *d_f){
  streamKernel<<<gridSize3D,blockSize3D>>>((double*)d_f);
  KERNEL_CHECK();
  SYNC_CHECK();
}
