// Kernels.cu
#include "Kernels.cuh"

// =========================================================================
// COMPUTE MACROS KERNEL
// =========================================================================
__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    double rho=0.0,jx=0.0,jy=0.0,rhoE=0.0,h=0.0;
    
    for(int kaux=0;kaux<Q;kaux++){
      double f = *(d_f+id+kaux*Lx*Ly);
      rho  += f;
      jx   += *(d_Cx+kaux)*f;
      jy   += *(d_Cy+kaux)*f;
      rhoE += 0.5*(d_Cx[kaux]*d_Cx[kaux]+d_Cy[kaux]*d_Cy[kaux])*f;
      if(f > 1e-12){
	h += f*log(f);
      }  
    }
    
    *(d_rho+id) = rho;
    *(d_jx+id)  = jx;
    *(d_jy+id)  = jy;
    *(d_rho_e+id) = rhoE - 0.5*rho*(jx*jx/(rho*rho)+jy*jy/(rho*rho));
    *(d_h+id) = h;
  }
}
// =========================================================================
// COMPUTE FEQ KERNEL
// =========================================================================
__global__ void computeFeqKernel(double *d_rho,double *d_jx,double *d_jy,double *d_feq){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    double rho = *(d_rho+id);
    double jx  = *(d_jx+id);
    double jy  = *(d_jy+id);
    
    *(d_feq+id+k*Lx*Ly) = d_w[k]*rho*(1.0+(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)/c_s2+0.5*(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)*(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)/(c_s2*c_s2)-0.5*(jx*jx/(rho*rho)+jy*jy/(rho*rho))/c_s2);
  }
}

// =========================================================================
// COMPUTE COLLISION ERRORS KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void computeCollisionErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    
    // Calculate feq moments
    double rho_feq=0.0,jx_feq=0.0,jy_feq=0.0,rhoE_feq=0.0;
    double hf=0.0,hfeq=0.0;
    
    for(int kaux=0;kaux<Q;kaux++){
      double f   = *(d_f+id+kaux*Lx*Ly);
      double feq = *(d_feq+id+kaux*Lx*Ly);
      
      rho_feq += feq;
      jx_feq  += *(d_Cx+kaux)*feq;
      jy_feq  += *(d_Cy+kaux)*feq;
      rhoE_feq += 0.5*(d_Cx[kaux]*d_Cx[kaux]+d_Cy[kaux]*d_Cy[kaux])*feq;
      
      if(f > 1e-15 && feq > 1e-15){
	hf   += f*log(f/feq);
	hfeq += feq*log(f/feq);
      }
    }
    
    double energy_feq = rhoE_feq - 0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq));
    
    *(d_mass_err+id) = fabs(*(d_rho+id)-rho_feq);
    *(d_momX_err+id) = fabs(*(d_jx+id)-jx_feq);
    *(d_momY_err+id) = fabs(*(d_jy+id)-jy_feq);
    *(d_energy_err+id) = fabs(*(d_rho_e+id)-energy_feq);
    *(d_hfhfeq_diff+id) = hf-hfeq;
  }
}

__global__ void findMinMaxKernel(double *d_data,double *d_min,double *d_max){
  __shared__ double s_min[THREADS_PER_BLOCK];
  __shared__ double s_max[THREADS_PER_BLOCK];
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int id = i + j*Lx;
  
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  if(i<Lx && j<Ly){
    *(s_min+tid) = *(d_data+id);
    *(s_max+tid) = *(d_data+id);
  }
  else{
    *(s_min+tid) = 1.0e30;
    *(s_max+tid) = -1.0e30;
  }
  __syncthreads();
  
  for(int step=THREADS_PER_BLOCK/2;step>0;step=step/2){
    if(tid<step){
      if(*(s_min+tid+step)<*(s_min+tid)){
	*(s_min+tid)=*(s_min+tid+step);
      }
      if(*(s_max+tid+step)>*(s_max+tid)){
	*(s_max+tid)=*(s_max+tid+step);
      }
    }
    __syncthreads();
  }
  
  if(tid==0){
    int block_id = blockIdx.x + blockIdx.y*gridDim.x;
    *(d_min+block_id) = *(s_min+tid);
    *(d_max+block_id) = *(s_max+tid);
  }
}

// =========================================================================
// FILL VBO KERNEL (For 3D surface)
// =========================================================================
__global__ void renderAndfillKernel(uchar4 *d_color,double *d_positions,double *d_uv,double *d_data,double offset,double scale){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    // normalized maps your entire data range to [-1, 1]:
    double normalized = (*(d_data+id)-offset)/scale;
    
    // Clamp for color
    double val = normalized;
    if(val<-1.0){
      val=-1.0;
    }
    if(val > 1.0){
      val=1.0;
    }  
    // Map to [0,1]
    double t = (val+1.0)*0.5;
    
    // Colormap
    uchar4 color;
    color.w = 255;
    if(t<0.5) {
      float s = t*2.0;
      color.x = 0;
      color.y = (unsigned char)(s*255.0);
      color.z = 255;
    }
    else{
      float s = (t-0.5)*2.0;
      if(s<0.5){
	float u = s*2.0;
	color.x = (unsigned char)(u*255.0);
	color.y = 255;
	color.z = (unsigned char)((1.0-u)*255.0);
      }
      else{
	float u = (s-0.5)*2.0;
	color.x = 255;
	color.y = (unsigned char)((1.0-u)*255.0);
	color.z = 0;
      }
    }
    *(d_color+id) = color;
    
    // Height
    double scale_pos = 0.02;
    double height_scale = 2.0;
    *(d_positions+id*3+0) = (i-Lx/2.0)*scale_pos;
    *(d_positions+id*3+1) = (j-Ly/2.0)*scale_pos;
    *(d_positions+id*3+2) = normalized*height_scale;
    
    *(d_uv+id*2+0) = (double)i/(Lx-1);
    *(d_uv+id*2+1) = (double)j/(Ly-1);
  }
}

// =========================================================================
// COLLISION KERNEL
// =========================================================================
__global__ void collisionKernel(double *d_f,double *d_feq){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    *(d_f+id+(k+Q)*Lx*Ly) = *(d_f+id+k*Lx*Ly)*OmegaPrima + *(d_feq+id+k*Lx*Ly)*Omega; 
  }
}

// =========================================================================
// COMPUTE LOCAL ERRORS KERNEL
// =========================================================================
__global__ void computeLocalErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if(i<Lx && j<Ly){
    int id = i + j*Lx;

    // Read after collision from buffer
    double rho_fcollision=0.0,jx_fcollision=0.0,jy_fcollision=0.0,rhoE_fcollision=0.0;
    double hf=0.0,hfcollision=0.0;
    for(int kaux=0;kaux<Q;kaux++){
      double f = *(d_f+id+kaux*Lx*Ly);
      double fcollision = *(d_f+id+(kaux+Q)*Lx*Ly);
      
      rho_fcollision += fcollision;
      jx_fcollision += *(d_Cx+kaux)*fcollision;
      jy_fcollision += *(d_Cy+kaux)*fcollision;
      rhoE_fcollision += 0.5*(d_Cx[kaux]*d_Cx[kaux]+d_Cy[kaux]*d_Cy[kaux])*fcollision;
      if(f>1e-12){
	hf += f*log(f/fcollision);
	hfcollision += fcollision*log(f/fcollision);
      }
    }

    double rho_e_fcollision = rhoE_fcollision - 0.5*rho_fcollision*(jx_fcollision*jx_fcollision/(rho_fcollision*rho_fcollision)+jy_fcollision*jy_fcollision/(rho_fcollision*rho_fcollision));

    *(d_mass_err+id) = fabs(*(d_rho+id)-rho_fcollision);
    *(d_momX_err+id) = fabs(*(d_jx+id)-jx_fcollision);
    *(d_momY_err+id) = fabs(*(d_jy+id)-jy_fcollision);
    *(d_energy_err+id) = fabs(*(d_rho_e+id)-rho_e_fcollision);
    *(d_hfhfeq_diff+id) = hf-hfcollision;
  }
}

// =========================================================================
// MARK LOCATION OF ENTROPY VIOLATIONS
// =========================================================================
__global__ void markEntropyViolationsKernel(double *d_hfhfeq_diff,int *d_violation_mask){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    *(d_violation_mask+id) = 0;
    if(*(d_hfhfeq_diff+id)<0.0){
      *(d_violation_mask+id) = 1;
    }
  }
}

// =========================================================================
// STREAMING KERNEL
// =========================================================================
__global__ void streamKernel(double *d_f){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int dest_i = (i+d_Cx[k]+Lx)%Lx;
    int dest_j = (j+d_Cy[k]+Ly)%Ly;
    *(d_f+dest_i+dest_j*Lx+k*Lx*Ly) = *(d_f+i+j*Lx+(k+Q)*Lx*Ly);
  }
}
