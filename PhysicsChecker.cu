// PhysicsChecker.cu
#include "PhysicsChecker.h"

PHYSICSCHECKER::PHYSICSCHECKER(void){
  // Allocate CPU error arrays
  h_mass_err = (double*)malloc(Lx*Ly*sizeof(double));
  h_momX_err = (double*)malloc(Lx*Ly*sizeof(double));
  h_momY_err = (double*)malloc(Lx*Ly*sizeof(double));
  h_energy_err = (double*)malloc(Lx*Ly*sizeof(double));
  h_hfhfeq_diff = (double*)malloc(Lx*Ly*sizeof(double));
  
  // Allocate GPU error arrays
  CUDA_CHECK(cudaMalloc((void**)&d_mass_err,Lx*Ly*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momX_err,Lx*Ly*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momY_err,Lx*Ly*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_energy_err,Lx*Ly*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_hfhfeq_diff,Lx*Ly*sizeof(double)));
}

PHYSICSCHECKER::~PHYSICSCHECKER(void){
  // Free host error arrays
  free(h_mass_err);
  free(h_momX_err);
  free(h_momY_err);
  free(h_energy_err);
  free(h_hfhfeq_diff);

  // Free GPU error arrays
  CUDA_CHECK(cudaFree(d_mass_err));
  CUDA_CHECK(cudaFree(d_momX_err));
  CUDA_CHECK(cudaFree(d_momY_err));
  CUDA_CHECK(cudaFree(d_energy_err));
  CUDA_CHECK(cudaFree(d_hfhfeq_diff));
}
