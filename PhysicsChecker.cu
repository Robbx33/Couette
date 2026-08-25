// PhysicsChecker.cu
#include "PhysicsChecker.h"

PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua){
  lbm = Gauss;
  km = Gargantua;
  
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

  CUDA_CHECK(cudaMalloc((void**)&d_violation_mask,Lx*Ly*sizeof(int)));

  // Kill old gnuplot processes BEFORE opening new pipe
  system("pkill -f gnuplot 2>/dev/null");
  usleep(200000);
  
  gp_pipe = popen("gnuplot -persist", "w");
  data_file = fopen("physics_errors.dat", "w");
  first_call = 1;

  fprintf(gp_pipe,"set title 'Physics Errors vs time'\n");
  fprintf(gp_pipe,"set xlabel 'time step'\n");
  fprintf(gp_pipe,"set ylabel 'Error'\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set logscale y\n");
  fflush(gp_pipe);
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

  CUDA_CHECK(cudaFree(d_violation_mask));

  //fclose(data_file);
  //pclose(gp_pipe);

  // Close gnuplot properly
  if(gp_pipe){
    fprintf(gp_pipe, "exit\n");
    fflush(gp_pipe);
    pclose(gp_pipe);
    gp_pipe = NULL;
  }
  if(data_file){
    fclose(data_file);
    data_file = NULL;
  }
  
  // Force kill any remaining gnuplot processes
  system("pkill -f 'gnuplot.*persist' 2>/dev/null");
}

void PHYSICSCHECKER::doubleCheckPhysics(int t){
  //km->launchComputeErrors(lbm->d_f,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_err,d_momX_err,d_momY_err,d_energy_err,d_hfhfeq_diff);
  
  fprintf(data_file, " %d", t);
  
  double min_val, max_val;    
  km->findMinMax(d_mass_err, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_momX_err, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_momY_err, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_energy_err, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_hfhfeq_diff, &min_val, &max_val);
  fprintf(data_file, " %e %e\n", min_val, max_val);
  
  fflush(data_file);
  
  if(first_call){
    fprintf(gp_pipe, "plot 'physics_errors.dat' using 1:2 w lp title 'mass_min', '' using 1:3 w lp title 'mass_max', ");
    fprintf(gp_pipe, "'' using 1:4 w lp title 'momX_min', '' using 1:5 w lp title 'momX_max', ");
    fprintf(gp_pipe, "'' using 1:6 w lp title 'momY_min', '' using 1:7 w lp title 'momY_max', ");
    fprintf(gp_pipe, "'' using 1:8 w lp title 'energy_min', '' using 1:9 w lp title 'energy_max', ");
    fprintf(gp_pipe, "'' using 1:10 w lp title 'entropy_min', '' using 1:11 w lp title 'entropy_max'\n");
    fflush(gp_pipe);
    first_call = 0;
  }
  else{
    fprintf(gp_pipe, "replot\n");
    fflush(gp_pipe);
  }
}

void PHYSICSCHECKER::markEntropyViolations(int t){
  km->launchMarkEntropyViolationsKernel(d_hfhfeq_diff,d_violation_mask);
}
