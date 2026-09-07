// Kernels.cu
#include "Kernels.cuh"

// =========================================================================
// COMPUTE MACROS KERNEL
// =========================================================================
__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    double rho=0.0,jx=0.0,jy=0.0,rhoE=0.0,h=0.0;
    
    for(int kaux=0;kaux<Q;kaux++){
      double f = *(d_f+id+(kaux+offset*Q)*Lx*Ly);
      rho  += f;
      jx   += *(d_Cx+kaux)*f;
      jy   += *(d_Cy+kaux)*f;
      rhoE += 0.5*(d_Cx[kaux]*d_Cx[kaux]+d_Cy[kaux]*d_Cy[kaux])*f;
      h += f*log(f/d_w[kaux]);
    }
    
    *(d_rho+id+offset*Lx*Ly) = rho;
    *(d_jx+id+offset*Lx*Ly)  = jx;
    *(d_jy+id+offset*Lx*Ly)  = jy;
    *(d_rho_e+id+offset*Lx*Ly) = rhoE - 0.5*rho*(jx*jx/(rho*rho)+jy*jy/(rho*rho));
    *(d_h+id+offset*Lx*Ly) = h;
  }
}
// =========================================================================
// COMPUTE KBC-FEQ KERNEL
// =========================================================================
__global__ void computeFeqKernel(double *d_rho,double *d_jx,double *d_jy,double *d_feq){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;

  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    double rho = *(d_rho+id);
    double ux = *(d_jx+id)/rho;
    double uy = *(d_jy+id)/rho;

    // ============================================
    // X-dimension contribution
    // ============================================
    double sqrt_x = sqrt(1.0+3.0*ux*ux);
    double term_x1 = 2.0-sqrt_x;
    double term_x2 = (2.0*ux+sqrt_x)/(1.0-ux);
    double term_x;
    if(d_Cx[k]==0){
      term_x=term_x1;
    }
    else{
      if(d_Cx[k]==1){
	term_x=term_x1*term_x2;
      }
      else{
	term_x=term_x1/term_x2;
      }
    }

    // ============================================
    // Y-dimension contribution
    // ============================================
    double sqrt_y = sqrt(1.0+3.0*uy*uy);
    double term_y1 = 2.0-sqrt_y;
    double term_y2 = (2.0*uy+sqrt_y)/(1.0-uy);
    double term_y;
    if(d_Cy[k]==0){
      term_y=term_y1;
    }
    else{
      if(d_Cy[k]==1){
	term_y=term_y1*term_y2;
      }
      else{
	term_y=term_y1/term_y2;
      }
    }

    // ============================================
    // Product and final equilibrium
    // ============================================
    double product = term_x*term_y;
    *(d_feq+id+k*Lx*Ly)=d_w[k]*rho*product;
  }
}

// =========================================================================
// COMPUTE COLLISION ERRORS KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void computeCollisionErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    
    // Calculate feq moments
    double rho_feq=0.0,jx_feq=0.0,jy_feq=0.0,rhoE_feq=0.0,hfeq=0.0;
    
    for(int kaux=0;kaux<Q;kaux++){
      double feq = *(d_feq+id+kaux*Lx*Ly);
      
      rho_feq += feq;
      jx_feq  += *(d_Cx+kaux)*feq;
      jy_feq  += *(d_Cy+kaux)*feq;
      rhoE_feq += 0.5*(d_Cx[kaux]*d_Cx[kaux]+d_Cy[kaux]*d_Cy[kaux])*feq;
      hfeq   += feq*log(feq/d_w[kaux]);
    }
    
    double energy_feq = rhoE_feq - 0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq));
    
    *(d_mass_diff+id) = fabs(rho_feq-*(d_rho+id));
    *(d_momX_diff+id) = fabs(jx_feq-*(d_jx+id));
    *(d_momY_diff+id) = fabs(jy_feq-*(d_jy+id));
    *(d_energy_diff+id) = fabs(energy_feq-*(d_rho_e+id));
    *(d_entropy_diff+id) = hfeq-*(d_h+id);
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
__global__ void computeLocalErrorsKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    *(d_mass_diff+id+offset*Lx*Ly) = fabs(*(d_rho+id+offset*Lx*Ly)-*(d_rho+id));
    *(d_momX_diff+id+offset*Lx*Ly) = fabs(*(d_jx+id+offset*Lx*Ly)-*(d_jx+id));
    *(d_momY_diff+id+offset*Lx*Ly) = fabs(*(d_jy+id+offset*Lx*Ly)-*(d_jy+id));
    *(d_energy_diff+id+offset*Lx*Ly) = fabs(*(d_rho_e+id+offset*Lx*Ly)-*(d_rho_e+id));
    *(d_entropy_diff+id+offset*Lx*Ly) = *(d_h+id+offset*Lx*Ly)-*(d_h+id);
  }
}

// =========================================================================
// MARK LOCATION OF ENTROPY VIOLATIONS
// =========================================================================
__global__ void markEntropyViolationsKernel(double *d_entropy_diff,int *d_violation_mask,int offset){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    *(d_violation_mask+id) = 0;
    if(*(d_entropy_diff+id+offset*Lx*Ly)>0){
      *(d_violation_mask+id) = 1;
    }
  }
}

// =========================================================================
// DEVICE FUNCTION: Compute entropy limit F(omega_eff)
// =========================================================================
__device__ double computeF(double *d_f,double *d_feq,int id,double omega_eff,int offset){
  double sum=0.0;
  for(int kaux=0;kaux<Q;kaux++){
    double f = *(d_f+id+kaux*Lx*Ly);
    double feq = *(d_feq+id+kaux*Lx*Ly);
    double w = d_w[kaux];
    double fnew = f-omega_eff*(f-feq);
    sum += fnew*log(fnew/w)-f*log(f/w);
  }
  return sum;
}

// =========================================================================
// DEVICE FUNCTION: Compute derivate d(F)/d(omega_eff)
// =========================================================================
__device__ double computeDF(double *d_f,double *d_feq,int id,double omega_eff,int offset){
  double sum=0.0;
  for(int kaux=0;kaux<Q;kaux++){
    double f = *(d_f+id+kaux*Lx*Ly);
    double feq = *(d_feq+id+kaux*Lx*Ly);
    double w = d_w[kaux];
    double fnew = f-omega_eff*(f-feq);
    sum -= (f-feq)*(log(fnew/w)+1.0);
  }
  return sum;
}

// =========================================================================
// KERNEL: Find omega_eff using Newton-Bisection hybrid method
// =========================================================================
__global__ void findOmegaEffKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    if(*(d_violation_mask + id)>0){
      // Constants from original code
      const double xacc = 1.0e-7;
      const int MAXIT = 1000;
      
      // Initial bracket [alpha1, alpha2]
      double alpha1 = 1e-15;
      double alpha2 = Omega;
      
      // Evaluate H at bracket endpoints
      double H1 = computeF(d_f,d_feq,id,alpha1,offset);
      double H2 = computeF(d_f,d_feq,id,alpha2,offset);
            
      // If H(alpha1) == 0, return alpha1
      if(fabs(H1)<xacc){
	*(d_omega_eff+id) = alpha1;
	return;
      }
      // If H(alpha2) == 0, return alpha2
      if(fabs(H2)<xacc){
	*(d_omega_eff+id) = alpha2;
	return;
      }
      
      // Initialize bracket bounds (alphal < alphah)
      double alphal, alphah;
      if(H1<0.0){
	alphal = alpha1;
	alphah = alpha2;
      }
      else{
	alphal = alpha2;
	alphah = alpha1;
      }
            
      // Initial guess: midpoint
      double rts = 0.5*(alpha1+alpha2);
      double dalphaold = fabs(alpha2-alpha1);
      double dalpha = dalphaold;
      
      // Compute H and dH at initial guess
      double F1 = computeF(d_f,d_feq,id,rts,offset);
      double F2 = computeDF(d_f,d_feq,id,rts,offset);
      
      for(int iter=0;iter<MAXIT;iter++){
	// Check if Newton step would go outside bracket or if
	// Newton step would be too large (use bisection instead)
	if((((rts-alphah)*F2-F1)*((rts-alphal)*F2-F1)>0.0) || (fabs(2.0*F1)>fabs(dalphaold*F2))){
	  // Use bisection
	  dalphaold = dalpha;
	  dalpha = 0.5*(alphah-alphal);
	  rts = alphal+dalpha;
	  if(alphal==rts){
	    *(d_omega_eff+id) = rts;
	    return;
	  }
	}
	else{
	  // Use Newton
	  dalphaold = dalpha;
	  dalpha = F1/F2;
	  double temp = rts;
	  rts -= dalpha;
	  if(temp==rts){
	    *(d_omega_eff+id) = rts;
	    return;
	  }
	}        
	// Check convergence
	if(fabs(dalpha)<xacc){
	  *(d_omega_eff+id) = rts;
	  return;
	}
	
	// Update function values
	F1 = computeF(d_f,d_feq,id,rts,offset);
	F2 = computeDF(d_f,d_feq,id,rts,offset);
	
	// Update bracket
	if(F1<0.0){
	  alphal = rts;
	}
	else{
	  alphah = rts;
	}
      }
      
      // If max iterations reached, return best guess
      *(d_omega_eff+id) = rts;
    }
    // else{
    //   // If no violation -> use BGK
    //   *(d_omega_eff+id) = Omega;
    // }
  }
}

// =========================================================================
// KERNEL: Entropic collision in the violation cellId
// =========================================================================
__global__ void entropicCollisionKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    if(*(d_violation_mask+id)==1){
      *(d_f+id+(k+offset*Q)*Lx*Ly)=*(d_f+id+k*Lx*Ly)-*(d_omega_eff+id)*(*(d_f+id+k*Lx*Ly)-*(d_feq+id+k*Lx*Ly)); 
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
