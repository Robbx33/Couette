// LBM.cu
#include "LBM.h"
#include <iostream>
#include <unistd.h>
using namespace std;

// ===========================================================
// CONSTRUCTOR
// ===========================================================

LATTICEBOLTZMANN::LATTICEBOLTZMANN(KernelsManager *Gargantua){
  km = Gargantua;
  
  // Kill gnuplot ONCE at the very beginning
  system("pkill gnuplot 2>/dev/null");
  usleep(200000);
  
  // Allocate host memory
  h_f     = (double*)malloc(Lx*Ly*Q*sizeof(double));
  h_Cx    = (int*)malloc(Q*sizeof(int));
  h_Cy    = (int*)malloc(Q*sizeof(int));
  h_w     = (double*)malloc(Q*sizeof(double));
  h_rho   = (double*)malloc(Lx*Ly*sizeof(double));
  h_jx    = (double*)malloc(Lx*Ly*sizeof(double));
  h_jy    = (double*)malloc(Lx*Ly*sizeof(double));
  h_rho_e = (double*)malloc(Lx*Ly*sizeof(double));
  h_h     = (double*)malloc(Lx*Ly*sizeof(double));
  h_feq   = (double*)malloc(Lx*Ly*Q*sizeof(double));
  
  // Allocate device memory
  CUDA_CHECK(cudaMalloc((void**)&d_f,Lx*Ly*Q*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_rho,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_jx,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_jy,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_rho_e,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_h,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_feq,Lx*Ly*Q*sizeof(double)));
  
  // Set lattice constants
  *h_Cx     = 0;  *h_Cy     = 0;
  *(h_Cx+1) = 1;  *(h_Cy+1) = 0;
  *(h_Cx+2) = 0;  *(h_Cy+2) = 1;
  *(h_Cx+3) = -1; *(h_Cy+3) = 0;
  *(h_Cx+4) = 0;  *(h_Cy+4) = -1;
  *(h_Cx+5) = 1;  *(h_Cy+5) = 1;
  *(h_Cx+6) = -1; *(h_Cy+6) = 1;
  *(h_Cx+7) = -1; *(h_Cy+7) = -1;
  *(h_Cx+8) = 1;  *(h_Cy+8) = -1;
  
  *h_w = 4.0/9.0;
  *(h_w+1) = *(h_w+2) = *(h_w+3) = *(h_w+4) = 1.0/9.0;
  *(h_w+5) = *(h_w+6) = *(h_w+7) = *(h_w+8) = 1.0/36.0;
    
  // Add Gaussian perturbation using KBC equilibrium
  for(int ix=0;ix<Lx;ix++){
    for(int iy=0;iy<Ly;iy++){
      int idx = ix + iy*Lx;
      double r2 = (ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly);
      double perturbation = amplitude*exp(-r2/(2.0*sigma*sigma));
      
      // Density and velocities with perturbation
      double rho = RHO0+perturbation;
      double ux = UX0;  // or add perturbation to velocity if you want
      double uy = UY0;
      
      // Compute KBC equilibrium for each direction
      for(int iz=0;iz<Q;iz++){
	// X-dimension contribution
	double sqrt_x = sqrt(1.0+3.0*ux*ux);
	double term_x1 = 2.0-sqrt_x;
	double term_x2 = (2.0*ux+sqrt_x)/(1.0-ux);
	double term_x;
	if(h_Cx[iz]==0){
	  term_x=term_x1;
	}
	else{
	  if(h_Cx[iz]==1){
	    term_x=term_x1*term_x2;
	  }
	  else{ 
	    term_x=term_x1/term_x2;
	  }
	}
	
	// Y-dimension contribution
	double sqrt_y = sqrt(1.0+3.0*uy*uy);
	double term_y1 = 2.0-sqrt_y;
	double term_y2 = (2.0*uy+sqrt_y)/(1.0-uy);
      
	double term_y;
	if(h_Cy[iz]==0){
	  term_y=term_y1;
	}
	else{
	  if(h_Cy[iz]==1){
	    term_y=term_y1*term_y2;
	  }
	  else{ 
	    term_y=term_y1/term_y2;
	  }
	}
	
	double product = term_x*term_y;
	
	// KBC equilibrium
	*(h_f+idx+iz*Lx*Ly)=h_w[iz]*rho*product;
      }
    }
  }
  
  // Copy to GPU
  CUDA_CHECK(cudaMemcpy((void*)(d_f+0+0*Lx+0*Lx*Ly),(const void*)(h_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyHostToDevice));
}

// ===========================================================
// DESTRUCTOR
// ===========================================================

LATTICEBOLTZMANN::~LATTICEBOLTZMANN(void){
  free(h_f);
  free(h_Cx);
  free(h_Cy);
  free(h_w);
  free(h_rho);
  free(h_jx);
  free(h_jy);
  free(h_rho_e);
  free(h_h);
  free(h_feq);
  CUDA_CHECK(cudaFree(d_f));
  CUDA_CHECK(cudaFree(d_rho));
  CUDA_CHECK(cudaFree(d_jx));
  CUDA_CHECK(cudaFree(d_jy));
  CUDA_CHECK(cudaFree(d_rho_e));
  CUDA_CHECK(cudaFree(d_h));
  CUDA_CHECK(cudaFree(d_feq));
  //CUDA_CHECK(cudaDeviceReset());
  cout<<"Memory freed (CPU+GPU) and Device Reset."<<endl;
}

void LATTICEBOLTZMANN::computeMacros(int offset){
  km->launchComputeMacros(d_f,d_rho,d_jx,d_jy,d_rho_e,d_h,offset);
}

void LATTICEBOLTZMANN::computeFeq(void){
  km->launchComputeFeq(d_rho,d_jx,d_jy,d_feq);
}

void LATTICEBOLTZMANN::Collision(void){
  km->launchCollision(d_f,d_feq);
}

void LATTICEBOLTZMANN::Stream(void){
  km->launchStream(d_f);
}

