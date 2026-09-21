// Brings in the __constant__ lattice arrays (d_Cx, d_Cy, d_w) and the
// forward declarations of the kerneles defined in this file.
#include "Kernels.cuh"

// =========================================================================
// COMPUTE MACROS KERNEL
// =========================================================================
__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset){
  // Compute the macroscopic moments at every lattice site, from the
  // distribution function f at the given offset.
  //
  // One thread per site (i, j). For that site, it loops over the Q
  // directions and accumulates:
  //   rho  = Σ f_k                     (density)
  //   jx   = Σ Cx_k f_k                (x momentum density)
  //   jy   = Σ Cy_k f_k                (y momentum density)
  //   rhoE = Σ 0.5 (Cx_k² + Cy_k²) f_k (total energy density)
  //   h    = Σ f_k ln(f_k / w_k)       (H-function)
  //
  // Then:
  //   rho_e is stored as the internal energy density
  //           rhoE - 0.5 rho (u² + v²)
  //
  // All five results go into their array at the given offset
  // (0 = pre-collision, 1 = post-collision).
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
  // Compute the discrete entropic equilibrium (KBC) at every site and
  // direction, from the current macros rho, jx, jy.
  //
  // One thread per (i, j, k). For that (site, direction):
  //   ux = jx/rho, uy = jy/rho   (local velocity components)
  //
  // The equilibrium is built as a product of two independent factors,
  // one per spatial axis. For each axis the factor depends on the
  // direction's velocity component (0, +1, or -1):
  //
  //   sqrt_α  = sqrt(1 + 3 u_α²)
  //   term_α1 = 2 - sqrt_α
  //   term_α2 = (2 u_α + sqrt_α) / (1 - u_α)
  //
  //   if C_α[k] == 0 : term_α = term_α1
  //   if C_α[k] == 1 : term_α = term_α1 * term_α2
  //   if C_α[k] ==-1 : term_α = term_α1 / term_α2
  //
  // Then:
  //
  //   feq_k = w_k rho term_x term_y
  //
  // Written into d_feq at every (i, j, k).
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;

  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    double rho = *(d_rho+id);
    double ux = *(d_jx+id)/rho;
    double uy = *(d_jy+id)/rho;
    // --------------------------------------------
    // X-dimension contribution
    // --------------------------------------------
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
    // --------------------------------------------
    // Y-dimension contribution
    // --------------------------------------------
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
    // --------------------------------------------
    // Product and final equilibrium
    // -------------------------------------------- 
    *(d_feq+id+k*Lx*Ly)=d_w[k]*rho*term_x*term_y;
  }
}

// =========================================================================
// COMPUTE COLLISION DIFFERENCES KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void computeCollisionDifferencesKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff){
  // Check how close the current state is to equilibrium, per site.
  //
  // One thread per site (i, j). For that site, it recomputes the
  // macroscopic moments of the equilibrium distribution feq:
  //   rho_feq, jx_feq, jy_feq, energy_feq (internal energy), hfeq
  // using the same formulas as computeMacrosKernel but applied to feq.
  //
  // Then it compares them with the current macros (rho, jx, jy, rho_e,
  // h) and stores the differences:
  //   d_mass_diff    |rho   - rho_feq   |
  //   d_momX_diff    |jx    - jx_feq    |
  //   d_momY_diff    |jy    - jy_feq    |
  //   d_energy_diff  |rho_e - energy_feq|
  //   d_entropy_diff   h    - hfeq      
  //
  // The first four should be near zero BECAUSE OF CONSERVATION OF
  // (mass, momentum, internal energy).
  // The entropy difference must be ≥ 0 BECAUSE OF ENTROPY RESTRICTION
  // if NOT, ENTROPY RESTRICTION VIOLATIONS.
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
    
    *(d_mass_diff+id) = fabs(*(d_rho+id)-rho_feq);
    *(d_momX_diff+id) = fabs(*(d_jx+id)-jx_feq);
    *(d_momY_diff+id) = fabs(*(d_jy+id)-jy_feq);
    *(d_energy_diff+id) = fabs(*(d_rho_e+id)-energy_feq);
    *(d_entropy_diff+id) = *(d_h+id)-hfeq;
  }
}
// =========================================================================
// COMPUTE (MIN/MAX)-REDUCTION KERNEL (All diagnostics on GPU)
// =========================================================================
__global__ void findMinMaxKernel(double *d_data,double *d_min,double *d_max,int offset){
  // Reduce d_data (at the given offset) to one partial min/max per block.
  //
  // One thread per site (i, j). Two shared arrays, one slot per thread,
  // hold the running min and max for the block (0 ≤ tid ≤ THREADS_PER_BLOCK).
  // 
  //   1. Each thread loads its own value from d_data into s_min[tid]
  //      and s_max[tid]. Threads outside the grid (i >= Lx or j >= Ly)
  //      load large sentinels so they don't affect the result.
  //
  //   2. Tree reduction: on each pass, the first half of the threads
  //      compares its slot with the slot at tid + step, keeping the
  //      smaller min and the larger max. step halves each pass until
  //      step = 1. __syncthreads separates the passes.
  //
  //   3. Thread 0 writes the block's final min and max into d_min and
  //      d_max, indexed by block id (row-major over the 2D grid).
  //
  // The kernel does not compute the global min/max. It leaves one
  // partial per block, and the caller finishes the reduction on the
  // host after copying the partials back.
  __shared__ double s_min[THREADS_PER_BLOCK];
  __shared__ double s_max[THREADS_PER_BLOCK];
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int id = i + j*Lx;
  
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  if(i<Lx && j<Ly){
    *(s_min+tid) = *(d_data+id+offset*Lx*Ly);
    *(s_max+tid) = *(d_data+id+offset*Lx*Ly);
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
__global__ void renderAndfillKernel(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale){
  // Build the geometry, texture coordinates, and colors of the 3D
  // surface, one thread per site (i, j).
  //
  //   1. Normalize the data value at this site into [-1, 1] using
  //      center and scale, then clamp it to that range. The result is
  //      used both as the height and as the color key.
  //
  //   2. Map the clamped value to [0, 1] (call it t) and pick an RGB
  //      color from a piecewise colormap:
  //        t < 0.5  ->  blue to green
  //        t >= 0.5 ->  green to yellow, then yellow to red
  //      Alpha is always 255.
  //      The color is stored in d_color at the site's linear index.
  //
  //   3. Write the position into the mapped position VBO:
  //        x = (i - Lx/2) * 0.02
  //        y = (j - Ly/2) * 0.02
  //        z = normalized * 2.0        (height)
  //      The 0.02 factor keeps the domain small in world units so it
  //      fits the camera set up in Visualizer.
  //
  //   4. Write the texture coordinates (u, v) into the mapped UV VBO,
  //      running from 0 to 1 across the grid, so the color texture maps
  //      onto the surface in the right orientation.  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    // normalized maps your entire data range to [-1, 1]:
    double normalized = (*(d_data+id)-center)/scale;
    
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
    *(vbo_positions_ptr+id*3+0) = (i-Lx/2.0)*scale_pos;
    *(vbo_positions_ptr+id*3+1) = (j-Ly/2.0)*scale_pos;
    *(vbo_positions_ptr+id*3+2) = normalized*height_scale;
    
    *(vbo_uv_ptr+id*2+0) = (double)i/(Lx-1);
    *(vbo_uv_ptr+id*2+1) = (double)j/(Ly-1);
  }
}

// =========================================================================
// COLLISION KERNEL
// =========================================================================
__global__ void collisionKernel(double *d_f,double *d_feq,int offset){
  // Apply the BGK collision to one (i, j, k) per thread.
  //
  // Reads the current f_k and the equilibrium feq_k at this site and
  // relaxes f_k toward feq_k with the collision frequency Omega,
  // writing the post-collision value into buffer offset=1:
  //
  //   f_new = f * OmegaPrima + feq * Omega
  //         = f * (1 - Omega) + feq * Omega
  //
  // The pre-collision values at buffer 0 are left untouched so they
  // can be used later for the entropy and conservation diagnostics.
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    *(d_f+id+(k+offset*Q)*Lx*Ly) = *(d_f+id+k*Lx*Ly)*OmegaPrima + *(d_feq+id+k*Lx*Ly)*Omega; 
  }
}
// =========================================================================
// COMPUTE LOCAL ERRORS KERNEL
// =========================================================================
__global__ void computeLocalDifferencesKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset){
  // Compute, per site, how much the macroscopic fields changed between
  // the pre-collision state (offset 0) and the pos-collision state (offset=1).
  //
  // One thread per site (i, j). For that site:
  //   d_mass_diff    |rho(offset=0)   - rho(offset=1)  |
  //   d_momX_diff    |jx(offset=0)    - jx(offset=1)   |
  //   d_momY_diff    |jy(offset=0)    - jy(offset=1)   |
  //   d_energy_diff  |rho_e(offset=0) - rho_e(offset=1)|
  //   d_entropy_diff   h(offset=0)    - h(offset=1)      (signed)
  //
  // The first four use absolute values, so they measure the magnitude
  // of the change. The entropy difference keeps its sign, so a negative
  // value means h increased from pre-collision (offset=0) to pos-collision
  // (offset=1) — that is what flags a violation of the entropy restriction.
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    *(d_mass_diff+id+offset*Lx*Ly) = fabs(*(d_rho+id)-*(d_rho+id+offset*Lx*Ly));
    *(d_momX_diff+id+offset*Lx*Ly) = fabs(*(d_jx+id)-*(d_jx+id+offset*Lx*Ly));
    *(d_momY_diff+id+offset*Lx*Ly) = fabs(*(d_jy+id)-*(d_jy+id+offset*Lx*Ly));
    *(d_energy_diff+id+offset*Lx*Ly) = fabs(*(d_rho_e+id)-*(d_rho_e+id+offset*Lx*Ly));
    *(d_entropy_diff+id+offset*Lx*Ly) = *(d_h+id)-*(d_h+id+offset*Lx*Ly);
  }
}
// =========================================================================
// MARK LOCATION OF ENTROPY VIOLATIONS
// =========================================================================
__global__ void markEntropyViolationsKernel(double *d_entropy_diff,int *d_violation_mask,int offset){
  // Flag every site where the entropy restriction is violated.
  //
  // One thread per site (i, j). Reads the entropy difference at the
  // d_entropy_diff(offset=1), which is stored as h(offset=0)-h(offset=1).
  // A negative value means h increased from pre-collision (offset=0) to
  // pos-collision (offset=1), i.e. the entropy went up
  // — that is a violation of the H-theorem.
  //
  // So:
  //   d_violation_mask[id] = 1  if d_entropy_diff < 0
  //   d_violation_mask[id] = 0  otherwise
  //
  // The mask is reset to 0 first, so it is fully overwritten on every
  // call.
  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  
  if(i<Lx && j<Ly){
    int id = i + j*Lx;
    *(d_violation_mask+id) = 0;
    if(*(d_entropy_diff+id+offset*Lx*Ly)<0){
      *(d_violation_mask+id) = 1;
    }
  }
}
// =========================================================================
// DEVICE FUNCTION: Compute entropy limit H(omega_eff)
// =========================================================================
__device__ double computeH(double *d_f,double *d_feq,int id,double omega_eff,int offset){
  // Entropy change caused by a trial collision at one site, with
  // parameter omega_eff.
  //
  // For each direction k, form the post-collision value
  //   fnew = f - omega_eff (f - feq)
  // and accumulate
  //   fnew ln(fnew/w) - f ln(f/w)
  // over the Q directions. The result is H(omega_eff).
  //
  // findOmegaEffKernel calls this to find the omega_eff that makes
  // H(omega_eff) = 0.
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
// DEVICE FUNCTION: Compute derivate d(H)/d(omega_eff)
// =========================================================================
__device__ double computeDH(double *d_f,double *d_feq,int id,double omega_eff,int offset){
  // Derivative of H with respect to omega_eff, at one site.
  //
  // For each direction k, with fnew = f - omega_eff (f - feq), it
  // accumulates
  //   -(f - feq) (ln(fnew/w) + 1)
  // over the Q directions. The result is dH/domega_eff.
  //
  // findOmegaEffKernel uses this for the Newton step when solving
  // H(omega_eff) = 0.
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
  // Find the entropic collision parameter omega_eff at each site that
  // violates the entropy restriction, so that applying the collision
  // with that parameter makes H(omega_eff) = 0 — the collision that
  // exactly saturates the H-theorem instead of exceeding it.
  //
  // One thread per site (i, j). Only flagged sites (d_violation_mask>0)
  // are processed; the rest are skipped.
  //
  // Method: Newton-Bisection hybrid on the scalar function H(omega_eff),
  // with H computed by computeH and dH/domega by computeDH.
  //
  //   - Bracket: [alpha1, alpha2] = [1e-15, Omega], where Omega is the
  //     standard BGK frequency. H is evaluated at both ends.
  //   - If H is already zero at an endpoint, that endpoint is the answer.
  //   - Otherwise the bracket is oriented so H(alphal) < 0 < H(alphah).
  //   - Starting from the midpoint, iterate up to MAXIT times:
  //       * Take a Newton step rts -= F1/F2 when it stays inside the
  //         bracket and is not too large.
  //       * Otherwise take a bisection step to halve the bracket.
  //       * Stop when the step is smaller than xacc, or when the
  //         updated rts equals the previous one (no progress).
  //   - If MAXIT is reached without converging, store the last rts.
  //
  // The converged omega_eff is written into d_omega_eff[id].
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
      double H1 = computeH(d_f,d_feq,id,alpha1,offset);
      double H2 = computeH(d_f,d_feq,id,alpha2,offset);
            
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
      double F1 = computeH(d_f,d_feq,id,rts,offset);
      double F2 = computeDH(d_f,d_feq,id,rts,offset);
      
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
	F1 = computeH(d_f,d_feq,id,rts,offset);
	F2 = computeDH(d_f,d_feq,id,rts,offset);
	
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
  }
}
// =========================================================================
// KERNEL: Entropic collision in the violation cellId
// =========================================================================
__global__ void entropicCollisionKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  // Redo the collision at the sites where the entropy restriction was
  // violated, using the corrected omega_eff instead of the BGK Omega.
  //
  // One thread per (i, j, k). Only sites flagged in d_violation_mask are
  // touched. For those, the new value is
  //
  //   f_new = f - omega_eff (f - feq)
  //
  // with omega_eff coming from findOmegaEffKernel. This is the entropic
  // collision that brings H(omega_eff) to zero, instead of the BGK
  // collision which would push H past zero.
  //
  // Non-flagged sites are left as they are, so the earlier BGK result
  // stands there.
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
__global__ void streamKernel(double *d_f,int offset){
  // Stream the distribution function: move each f_k to the neighbouring
  // site in direction k.
  //
  // One thread per (i, j, k). For direction k, the source value at
  // (i, j) in buffer `offset=1` is written to the destination site
  //
  //   dest_i = (i + Cx[k] + Lx) % Lx
  //   dest_j = (j + Cy[k] + Ly) % Ly
  //
  // in buffer 1 - offset. The "+ Lx" and "+ Ly" before the modulo keep
  // the index positive for negative velocity components.
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int dest_i = (i+d_Cx[k]+Lx)%Lx;
    int dest_j = (j+d_Cy[k]+Ly)%Ly;
    *(d_f+dest_i+dest_j*Lx+k*Lx*Ly) = *(d_f+i+j*Lx+(k+offset*Q)*Lx*Ly);
  }
}
