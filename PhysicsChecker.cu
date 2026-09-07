// PhysicsChecker.cu
#include "PhysicsChecker.h"

PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua){
  lbm = Gauss;
  km = Gargantua;
  
  // Allocate CPU error arrays
  h_mass_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_momX_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_momY_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_energy_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_entropy_diff = (double*)malloc(Lx*Ly*sizeof(double));

  h_violation_mask = (int*)malloc(Lx*Ly*sizeof(int));
  
  // Allocate GPU error arrays
  CUDA_CHECK(cudaMalloc((void**)&d_mass_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momX_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momY_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_energy_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_entropy_diff,Lx*Ly*2*sizeof(double)));

  CUDA_CHECK(cudaMalloc((void**)&d_violation_mask,Lx*Ly*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_omega_eff,Lx*Ly*sizeof(double)));

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
  free(h_mass_diff);
  free(h_momX_diff);
  free(h_momY_diff);
  free(h_energy_diff);
  free(h_entropy_diff);

  free(h_violation_mask);

  // Free GPU error arrays
  CUDA_CHECK(cudaFree(d_mass_diff));
  CUDA_CHECK(cudaFree(d_momX_diff));
  CUDA_CHECK(cudaFree(d_momY_diff));
  CUDA_CHECK(cudaFree(d_energy_diff));
  CUDA_CHECK(cudaFree(d_entropy_diff));
  CUDA_CHECK(cudaFree(d_violation_mask));
  CUDA_CHECK(cudaFree(d_omega_eff));
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
  km->findMinMax(d_mass_diff, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_momX_diff, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_momY_diff, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_energy_diff, &min_val, &max_val);
  fprintf(data_file, " %e %e", min_val, max_val);
  
  km->findMinMax(d_entropy_diff, &min_val,&max_val);
  fprintf(data_file, " %e %e\n", fabs(min_val),fabs(max_val));
  
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

int PHYSICSCHECKER::countViolations(void){
  CUDA_CHECK(cudaMemcpy((void*)(h_violation_mask+0+0*Lx),(const void*)(d_violation_mask+0+0*Lx),(size_t)Lx*Ly*sizeof(int),cudaMemcpyDeviceToHost));
  int count = 0;
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      if(*(h_violation_mask+idx)>0){
	count++;
      }
    }
  }
  return count;
}

void PHYSICSCHECKER::applyELBM(int offset){
  // Find omega_eff
  km->launchFindOmegaEff(lbm->d_f,lbm->d_feq,d_omega_eff,d_violation_mask,offset);

  // Apply collision
  km->launchEntropicCollision(lbm->d_f,lbm->d_feq,d_omega_eff,d_violation_mask,offset);
}

void PHYSICSCHECKER::entropyLocalConservation(int t,int offset){
  // Compute post-collision macros
  lbm->computeMacros(offset);
  
  // Compute local errors and mark violations
  km->launchComputeLocalErrors(lbm->d_f,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_diff,d_momX_diff,d_momY_diff,d_energy_diff,d_entropy_diff,offset);
  km->launchMarkEntropyViolations(d_entropy_diff, d_violation_mask, 1);
  int before = countViolations();
  if(before>0){
    // Triggers Entropic Lattice Boltzmann
    applyELBM(offset);
    // Recompute to verify
    lbm->computeMacros(offset);
    km->launchComputeLocalErrors(lbm->d_f,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_diff,d_momX_diff,d_momY_diff,d_energy_diff,d_entropy_diff,offset);
    km->launchMarkEntropyViolations(d_entropy_diff, d_violation_mask, 1);
    int after = countViolations();
    printf("t=%d: %d → %d\n",t,before,after);
  }
  else{
    printf("t=%d: 0 violations\n",t);
  }
}
