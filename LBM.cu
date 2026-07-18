// LBM.cu
#include "LBM.h"
#include "book.h"
#include <iostream>
#include <unistd.h>
using namespace std;
// ===========================================================
// LATTICE CONSTANTS
// ===========================================================
__constant__ int d_Cx[Q];
__constant__ int d_Cy[Q];
__constant__ double d_w[Q];
__constant__ double d_Omega;
__constant__ double d_OmegaPrima;

const double RHO0=1.0,UX0=0.0,UY0=0.0;
const double nu = 0.15;   
const double Tau = nu/c_s2 + 0.5*dt;

// ===========================================================
// CUDA KERNELS
// ===========================================================

__global__ void collisionKernel(double* d_f){
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
    *(d_f+id+(k+Q)*Lx*Ly) = *(d_f+id+k*Lx*Ly)*d_OmegaPrima + d_Omega*feq;
  }
}

__global__ void streamKernel(double* d_f){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  
  if(i<Lx && j<Ly && k<Q){
    int dest_i = (i+d_Cx[k]+Lx)%Lx;
    int dest_j = (j+d_Cy[k]+Ly)%Ly;
    *(d_f+dest_i+dest_j*Lx+k*Lx*Ly) = *(d_f+i+j*Lx+(k+Q)*Lx*Ly);
  }
}

// ===========================================================
// CONSTRUCTOR
// ===========================================================

LATTICEBOLTZMANN::LATTICEBOLTZMANN() {
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
    HANDLE_ERROR(cudaMalloc((void**)&d_f,Lx*Ly*Q*2*sizeof(double)));

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
    
    h_Omega = dt/Tau;
    h_OmegaPrima = 1.0 - h_Omega;
    
    // Initialize f with equilibrium (rho=1.0, ux=0, uy=0)
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	for(int iz=0;iz<Q;iz++){
	  *(h_f+idx+iz*Lx*Ly) = h_w[iz]*RHO0*(1+(h_Cx[iz]*UX0+h_Cy[iz]*UY0)/c_s2+0.5*((h_Cx[iz]*UX0)*(h_Cx[iz]*UX0)+2*(h_Cx[iz]*UX0)*(h_Cy[iz]*UY0)+(h_Cy[iz]*UY0)*(h_Cy[iz]*UY0))/(c_s2*c_s2)-0.5*(UX0*UX0+UY0*UY0)/c_s2);
	}
      }
    }
    
    // Add Gaussian perturbation
    double amplitude = 0.01;
    double sigma = 10.0;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double r2 = (ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly);
	double perturbation = amplitude*exp(-r2/(2.0*sigma*sigma));
	for(int iz=0;iz<Q;iz++){
	  *(h_f+idx+iz*Lx*Ly) += *(h_w+iz)*perturbation;
	}
      }
    }
    
    // Copy to GPU
    HANDLE_ERROR(cudaMemcpy((void*)(d_f+0+0*Lx+0*Lx*Ly),(const void*)(h_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_Cx+0),(const void*)(h_Cx+0),(size_t)Q*sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_Cy+0),(const void*)(h_Cy+0),(size_t)Q*sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_w+0),(const void*)(h_w+0),(size_t)Q*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)&d_Omega,(const void*)&h_Omega,(size_t)sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)&d_OmegaPrima,(const void*)&h_OmegaPrima,(size_t)sizeof(double)));
    
    // Initialize gnuplot
    system("pkill gnuplot 2>/dev/null");
    usleep(300000);
    gp_pipe = popen("gnuplot -persist","w");
    gp_pipe_error = popen("gnuplot -persist","w");
    
    fprintf(gp_pipe,"set xlabel 'ix'\n");
    fprintf(gp_pipe,"set ylabel 'iy'\n");
    fprintf(gp_pipe,"set zlabel 'Density'\n");
    fprintf(gp_pipe,"set grid\n");
    fprintf(gp_pipe,"set xrange [0:%d]\n", Lx);
    fprintf(gp_pipe,"set yrange [0:%d]\n", Ly);
    fprintf(gp_pipe,"set zrange [0.98:1.015]\n");
    fprintf(gp_pipe,"set cbrange [0.98:1.015]\n");
    fprintf(gp_pipe,"set palette defined (0 '#0000FF', 0.33 '#00FFFF', 0.66 '#FFFF00', 1 '#FF0000')\n");
    fflush(gp_pipe);
    
    // Error window settings
    fprintf(gp_pipe_error,"set xlabel 'ix'\n");
    fprintf(gp_pipe_error,"set ylabel 'iy'\n");
    fprintf(gp_pipe_error,"set zlabel 'Error'\n");
    fprintf(gp_pipe_error,"set grid\n");
    fprintf(gp_pipe_error,"set xrange [0:%d]\n", Lx);
    fprintf(gp_pipe_error,"set yrange [0:%d]\n", Ly);
    fprintf(gp_pipe_error,"set view map\n");
    fprintf(gp_pipe_error,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000')\n");
    fflush(gp_pipe_error);
    
    cout<<"Memory allocated (CPU+GPU)."<<endl;
}

// ===========================================================
// DESTRUCTOR
// ===========================================================

LATTICEBOLTZMANN::~LATTICEBOLTZMANN(){
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
  pclose(gp_pipe);
  pclose(gp_pipe_error);
  HANDLE_ERROR(cudaFree(d_f));
  HANDLE_ERROR(cudaDeviceReset());
  cout<<"Memory freed (CPU+GPU) and Device Reset."<<endl;
}

// ===========================================================
// LBM METHODS
// ===========================================================

void LATTICEBOLTZMANN::Choque(int t){
  int dimx=4,dimy=2,dimz=Q;
  dim3 block(dimx,dimy,dimz);
  dim3 grid((Lx+block.x-1)/block.x,(Ly+block.y-1)/block.y,(Q+block.z-1)/block.z);
  collisionKernel<<<grid, block>>>((double*)d_f);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void LATTICEBOLTZMANN::Adveccion(){
  int dimx=4,dimy=2,dimz=Q;
  dim3 block(dimx,dimy,dimz);
  dim3 grid((Lx+block.x-1)/block.x,(Ly+block.y-1)/block.y,(Q+block.z-1)/block.z);
  streamKernel<<<grid, block>>>((double*)d_f);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void LATTICEBOLTZMANN::copyBack(){
  HANDLE_ERROR(cudaMemcpy((void*)(h_f+0+0*Lx+0*Lx*Ly),(const void*)(d_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyDeviceToHost));
}

void LATTICEBOLTZMANN::calcularMacros(){
  for(int ix=0;ix<Lx;ix++){
    for(int iy=0;iy<Ly;iy++){
      int idx = ix + iy*Lx;
      double rho=0.0,jx=0.0,jy=0.0,rhoE=0.0,h=0.0;
      for(int iz=0;iz<Q;iz++){
	double f = *(h_f+idx+iz*Lx*Ly);
	rho  += f;
	jx   += *(h_Cx+iz)*f;
	jy   += *(h_Cy+iz)*f;
	rhoE += 0.5*(h_Cx[iz]*h_Cx[iz]+h_Cy[iz]*h_Cy[iz])*f;
	if(f > 1e-12){
	  h += f*log(f);
	}
      }
      *(h_rho+idx)   = rho;
      *(h_jx+idx)    = jx;
      *(h_jy+idx)    = jy;
      *(h_rho_e+idx) = rhoE-0.5*rho*(jx*jx/(rho*rho)+jy*jy/(rho*rho));
      *(h_h+idx)     = h;
    }
  }
}

void LATTICEBOLTZMANN::calcularFeq(){
  for(int ix=0;ix<Lx;ix++){
    for(int iy=0;iy<Ly;iy++){
      int idx = ix + iy*Lx;
      double rho = *(h_rho+idx);
      double jx  = *(h_jx+idx);
      double jy  = *(h_jy+idx);
      for(int iz=0;iz<Q;iz++){
	*(h_feq+idx+iz*Lx*Ly) = h_w[iz]*rho*(1.0+(h_Cx[iz]*jx/rho+h_Cy[iz]*jy/rho)/c_s2+0.5*(h_Cx[iz]*jx/rho+h_Cy[iz]*jy/rho)*(h_Cx[iz]*jx/rho+h_Cy[iz]*jy/rho)/(c_s2*c_s2)-0.5*(jx*jx/(rho*rho)+jy*jy/(rho*rho))/c_s2);
      }
    }
  }
}

void LATTICEBOLTZMANN::dibuje3D(int t){
  FILE *tmp = fopen("density.dat","w");
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      fprintf(tmp,"%d %d %f\n",ix,iy,*(h_rho+idx));
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
  
  fprintf(gp_pipe,"set title 'Gaussian Pulse - t=%d'\n",t);
  fprintf(gp_pipe,"splot 'density.dat' with pm3d\n");
  fflush(gp_pipe);
  usleep(2000000);
}

