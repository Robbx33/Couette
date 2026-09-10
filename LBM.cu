// LBM.h     -> class declaration
// iostream  -> std::cout for status messages
// unistd.h  -> usleep (pauses after killing old gnuplot processes)
//
// using namespace std lets us write cout instead of std::cout
#include "LBM.h"
#include <iostream>
#include <unistd.h>
using namespace std;

// ===========================================================
// CONSTRUCTOR-LATTICEBOLTZMANN
// ===========================================================
LATTICEBOLTZMANN::LATTICEBOLTZMANN(KernelsManager *Gargantua){
  // Point km at the KernelsManager object passed in.
  km = Gargantua;
  
  // Kill any leftover gnuplot process from a previous run, then wait
  // 200 ms to make sure it's fully gone before continuing.
  system("pkill gnuplot 2>/dev/null");
  usleep(200000);
  
  // Allocate CPU memory for all lattice quantities.
  // Lx*Ly*Q sized arrays: distribution function and equilibrium.
  // Lx*Ly sized arrays: macroscopic fields.
  // Q sized arrays: discrete velocities and weights.
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
  
  // Allocate GPU memory for all lattice quantities. The *2 factor adds an
  // extra buffer per array to hold the post-collision values t+dt (offset 1)
  // alongside the pre-collision ones (offset 0). d_feq is single-buffered.
  CUDA_CHECK(cudaMalloc((void**)&d_f,Lx*Ly*Q*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_rho,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_jx,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_jy,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_rho_e,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_h,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_feq,Lx*Ly*Q*sizeof(double)));
  
  // Set CPU Lattice Boltzmann discrete velocities (D2Q9).
  // The 9 directions: rest (0), the 4 axis-aligned (E, N, W, S),
  // and the 4 diagonals (NE, NW, SW, SE). 
  *h_Cx     = 0;  *h_Cy     = 0;
  *(h_Cx+1) = 1;  *(h_Cy+1) = 0;
  *(h_Cx+2) = 0;  *(h_Cy+2) = 1;
  *(h_Cx+3) = -1; *(h_Cy+3) = 0;
  *(h_Cx+4) = 0;  *(h_Cy+4) = -1;
  *(h_Cx+5) = 1;  *(h_Cy+5) = 1;
  *(h_Cx+6) = -1; *(h_Cy+6) = 1;
  *(h_Cx+7) = -1; *(h_Cy+7) = -1;
  *(h_Cx+8) = 1;  *(h_Cy+8) = -1;
  
  // Set CPU Lattice Boltzmann discrete weights (D2Q9).
  // The 9 weights: rest (0), the 4 axis-aligned (E, N, W, S),
  // and the 4 diagonals (NE, NW, SW, SE)
  *h_w = 4.0/9.0;
  *(h_w+1) = *(h_w+2) = *(h_w+3) = *(h_w+4) = 1.0/9.0;
  *(h_w+5) = *(h_w+6) = *(h_w+7) = *(h_w+8) = 1.0/36.0;
    
  // f start from KBC_feq
  // Unlike the standard Hermite-polynomial BGK equilibrium, this form is
  // derived by minimizing the discrete h-function h = Σ f ln(f/w) under
  // mass and momentum conservation (maximum entropy principle) using
  // Lagrange multipliers for the constrains
  for(int ix=0;ix<Lx;ix++){
    for(int iy=0;iy<Ly;iy++){
      int idx = ix + iy*Lx;
      // Mass in a Gaussian form 
      double r2 = (ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly);
      double perturbation = amplitude*exp(-r2/(2.0*sigma*sigma));
      // Add Gaussian form mass to the initial mass RHO0
      double rho = RHO0+perturbation;
      double ux = UX0;  // or add perturbation to velocity if you want
      double uy = UY0;
      // Compute KBC_feq for each direction 
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
	// f = KBC_feq
	*(h_f+idx+iz*Lx*Ly)=h_w[iz]*rho*term_x*term_y;
      }
    }
  }
  
  // Copy the initial distribution from host (h_f) to device (d_f).
  // Only the first buffer (offset 0) is filled; the second buffer
  // (offset 1, post-collision t+dt) will be written during the simulation.
  CUDA_CHECK(cudaMemcpy((void*)(d_f+0+0*Lx+0*Lx*Ly),(const void*)(h_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyHostToDevice));
}

// ===========================================================
// DESTRUCTOR-LATTICEBOLTZMANN
// ===========================================================
LATTICEBOLTZMANN::~LATTICEBOLTZMANN(void){
  // Liberates CPU memory used by the Lattice Boltzmann quantities
  free((double*)h_f);
  free((int*)h_Cx);
  free((int*)h_Cy);
  free((double*)h_w);
  free((double*)h_rho);
  free((double*)h_jx);
  free((double*)h_jy);
  free((double*)h_rho_e);
  free((double*)h_h);
  free((double*)h_feq);

  // Liberates GPU memory used by the Lattice Boltzmann quantities
  CUDA_CHECK(cudaFree((double*)d_f));
  CUDA_CHECK(cudaFree((double*)d_rho));
  CUDA_CHECK(cudaFree((double*)d_jx));
  CUDA_CHECK(cudaFree((double*)d_jy));
  CUDA_CHECK(cudaFree((double*)d_rho_e));
  CUDA_CHECK(cudaFree((double*)d_h));
  CUDA_CHECK(cudaFree((double*)d_feq));

  // LATTICEBOLTZMANN checkout
  cout<<"Memory freed LATTICEBOLTZMANN(CPU+GPU)."<<endl;
}

// ===========================================================
// computeMacros-LATTICEBOLTZMANN
// ===========================================================
void LATTICEBOLTZMANN::computeMacros(int offset){
  // launches kernel to compute GPU Macros at t(offset=0) or t+dt(offset=1)
  km->launchComputeMacros((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(int)offset);
}

// ===========================================================
// computeFeq-LATTICEBOLTZMANN
// ===========================================================
void LATTICEBOLTZMANN::computeFeq(void){
  // launches kernel to compute GPU feq
  km->launchComputeFeq((double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_feq);
}

// ===========================================================
// Collision-LATTICEBOLTZMANN
// ===========================================================
void LATTICEBOLTZMANN::Collision(void){
  // launches kernel to perform GPU collision
  km->launchCollision((double*)d_f,(double*)d_feq);
}

// ===========================================================
// Stream-LATTICEBOLTZMANN
// ===========================================================
void LATTICEBOLTZMANN::Stream(void){
  // launches kernel to perform GPU stream
  km->launchStream((double*)d_f);
}

