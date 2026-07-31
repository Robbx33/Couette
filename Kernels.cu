// Kernels.cu
#include "Kernels.cuh"

// =========================================================================
// COLLISION KERNEL
// =========================================================================
__global__ void collisionKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_w){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    double rho=0.0,jx=0.0,jy=0.0;
    for(int kaux=0;kaux<Q;kaux++){
      double f = *(d_f+id+kaux*Lx*Ly);
      rho += f;
      jx += *(d_Cx+kaux)*f;
      jy += *(d_Cy+kaux)*f;      
    }
    
    double feq = d_w[k]*rho*(1.0+(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)/c_s2+0.5*(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)*(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)/(c_s2*c_s2)-0.5*(jx*jx/(rho*rho)+jy*jy/(rho*rho))/c_s2);
    
    *(d_f+id+(k+Q)*Lx*Ly) = *(d_f+id+k*Lx*Ly)*OmegaPrima + Omega*feq; 
  }
}

// =========================================================================
// STREAMING KERNEL
// =========================================================================

__global__ void streamKernel(double *d_f,int *d_Cx,int *d_Cy){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int dest_i = (i+d_Cx[k]+Lx)%Lx;
    int dest_j = (j+d_Cy[k]+Ly)%Ly;
    *(d_f+dest_i+dest_j*Lx+k*Lx*Ly) = *(d_f+i+j*Lx+(k+Q)*Lx*Ly);
  }
}

// =========================================================================
// COMPUTE MACROS KERNEL
// =========================================================================
__global__ void computeMacrosKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h){
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
__global__ void computeFeqKernel(int *d_Cx,int *d_Cy,double *d_w,double *d_rho,double *d_jx,double *d_jy,double *d_feq){
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
// COMPUTE ERRORS KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void computeErrorsKernel(double *d_f,int *d_Cx,int *d_Cy,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_err,double *d_momX_err,double *d_momY_err,double *d_energy_err,double *d_hfhfeq_diff){
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

    // Atomic max/min for diagnostics (use atomicMax/min for doubles)
    // Note: For simplicity, we'll use a reduction approach in the manager
  }
}

// =========================================================================
// RENDER KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void renderKernel(uchar4 *d_texture,double *d_rho){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    double rho = *(d_rho+id);

    // Normalize: 0.98 to 1.01 -> 0 to 1
    double normalized = (rho-0.98)/0.03;//takes rho in range between 0 and 1.
    if(normalized < 0.0){
      normalized = 0.0;//puts it in a safety range
    }
    if(normalized > 1.0){
      normalized = 1.0;//puts it in a safety range
    }

    // Red = high density, Green = low density
    unsigned char red = (unsigned char)(normalized*255.0);
    unsigned char green = (unsigned char)((1.0-normalized)*255.0);

    uchar4 color;
    color.x = red;
    color.y = green;
    color.z = 0;
    color.w = 255;
    *(d_texture+id) = color;
  }
}
