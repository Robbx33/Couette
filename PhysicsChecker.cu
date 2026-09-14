// PhysicsChecker.h     -> class declaration
// iostream  -> std::cout for status messages
// using namespace std lets us write cout instead of std::cout
#include "PhysicsChecker.h"
#include <iostream>
using namespace std;

// ===========================================================
// CONSTRUCTOR-PHYSICSCHECKER
// ===========================================================
PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua){  
  // Point lbm & km at the KernelsManager object passed in.
  lbm = Gauss;
  km = Gargantua;
  
  // On the CPU and GPU
  // Checks for mass, momentum, internal energy conservation and entropy restriction
  // Σ Ωᵢ(f(x,t)) = 0                 
  // Σ C_iα.Ωᵢ(f(x,t)) = 0       
  // Σ C_iα.C_iα.Ωᵢ(f(x,t)) = 0 
  // Σ Ln(f_i(x,t)).Ωᵢ(f(x,t)) ≤  0
  // i.e.
  // Σ f_i(x,t) - Σ f_i^eq(x,t) = 0
  // Σ C_iα.f_i(x,t) - Σ C_iα.f_i^eq(x,t) = 0
  // Σ C_iα.C_iα.f_i(x,t) - Σ C_iα.C_iα.f_i^eq(x,t) = 
  // Σ f_i(x,t).Ln(f_i(x,t)/w_i) - Σ f_i^eq(x,t).Ln(f_i^eq(x,t)/w_i) ≤  0
  // which means
  // ρ(x,t) - ρ^eq(x,t) = 0
  // ρ(x,t).u_α(x,t) - ρ^eq(x,t).u_α^eq(x,t) = 0
  // ρ(x,t)E(x,t) - ρ^eq(x,t)E^eq(x,t) = 0
  // h(x,t) - h^eq(x,t) ≤ 0
  h_mass_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_momX_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_momY_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_energy_diff = (double*)malloc(Lx*Ly*sizeof(double));
  h_entropy_diff = (double*)malloc(Lx*Ly*sizeof(double));
  // On the GPU buffer area
  // Checks for mass, momentum, internal energy conservation and entropy restriction all at the same point of space x
  // (ρ(x,t+dt)-ρ(x,t))/dt + 0 = 0
  // (ρ(x,t+dt).u_α(x,t+dt)-ρ(x,t).u_α(x,t))/dt + 0 = F_α
  // (ρ(x,t+dt).e(x,t+dt)-ρ(x,t).e(x,t))/dt + 0 = 0
  // (h(x,t+dt)-h(x,t))/dt + 0  ≤  0
  // i.e.
  // ρ(x,t+dt) - ρ(x,t) = 0
  // ρ(x,t+dt).u_α(x,t+dt) - ρ(x,t).u_α(x,t) = F_α 
  // ρ(x,t+dt)e(x,t+dt) - ρ(x,t)e(x,t) = 0
  // h(x,t+dt) - h(x,t) ≤ 0
  // which means
  // Σ f_i(x,t+dt) - Σ f_i(x,t) = 0
  // Σ C_iα.f_i(x,t+dt) - Σ C_iα.f_i(x,t) = F_α
  // Σ C_iα.C_iα.f_i(x,t+dt) - Σ C_iα.C_iα.f_i(x,t) = 0 
  // Σ f_i(x,t+dt).Ln(f_i(x,t+dt)/w_i) - Σ f_i(x,t).Ln(f_i(x,t)/w_i) ≤  0
  CUDA_CHECK(cudaMalloc((void**)&d_mass_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momX_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momY_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_energy_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_entropy_diff,Lx*Ly*2*sizeof(double)));

  // On the CPU and GPU. Not all the position on space will be use, just the
  // ones where there are entropy restriction violations
  h_violation_mask = (int*)malloc(Lx*Ly*sizeof(int));  
  CUDA_CHECK(cudaMalloc((void**)&d_violation_mask,Lx*Ly*sizeof(int)));

  // On the GPU. There will be a correction of ω by ω_eff(x,t) on those
  // places where there were entropy restriction violations
  CUDA_CHECK(cudaMalloc((void**)&d_omega_eff,Lx*Ly*sizeof(double)));

  // Kill any leftover gnuplot process from a previous run, then wait
  // 200 ms to make sure it's fully gone before continuing.
  system("pkill -f gnuplot 2>/dev/null");
  usleep(200000);

  // Open a pipe to gnuplot and a file to log the physics errors.
  // The pipe lets us send plot commands directly from C++.
  // The data file holds the numeric values that gnuplot reads.
  // first_call flags the very first plot so gnuplot runs 'plot' once,
  // then 'replot' on every later timestep.
  gp_pipe = popen("gnuplot -persist", "w");
  data_file = fopen("conservation&(-restriction)_quantities.dat","w");
  first_call = 1;

  // Send initial plot configuration to gnuplot through the pipe:
  // title, axis labels, grid, and a logarithmic y-axis (errors span
  // many orders of magnitude, so log scale makes them readable).
  // fflush pushes the commands to gnuplot immediately instead of
  // waiting for the buffer to fill up
  fprintf(gp_pipe,"set title 'conservation and (-restriction) quantities vs time'\n");
  fprintf(gp_pipe,"set xlabel 'time step'\n");
  fprintf(gp_pipe,"set ylabel 'Error'\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set logscale y\n");
  fflush(gp_pipe);
}

// ===========================================================
// DESTRUCTOR-PHYSICSCHECKER
// ===========================================================
PHYSICSCHECKER::~PHYSICSCHECKER(void){
  // Liberates CPU and GPU memory used by to store Conservative & Restrictive
  // quantities
  free(h_mass_diff);
  free(h_momX_diff);
  free(h_momY_diff);
  free(h_energy_diff);
  free(h_entropy_diff);
  CUDA_CHECK(cudaFree(d_mass_diff));
  CUDA_CHECK(cudaFree(d_momX_diff));
  CUDA_CHECK(cudaFree(d_momY_diff));
  CUDA_CHECK(cudaFree(d_energy_diff));
  CUDA_CHECK(cudaFree(d_entropy_diff));
  
  // Liberates CPU and GPU memory used to mark entropy violations
  free(h_violation_mask);
  CUDA_CHECK(cudaFree(d_violation_mask));

  // Liberates GPU memory used to store roots of F(omega_eff)=0
  CUDA_CHECK(cudaFree(d_omega_eff));

  // Close the gnuplot pipe and the data file cleanly.
  // 'exit' tells gnuplot to quit; fflush makes sure the command is
  // sent; pclose waits for gnuplot to terminate and closes the pipe.
  // fclose flushes and closes the data file.
  fprintf(gp_pipe, "exit\n");
  fflush(gp_pipe);
  pclose(gp_pipe);
  fclose(data_file);

  // PHYSICSCHECKER checkout
  cout<<"Memory freed PHYSICSCHECKER(CPU+GPU)."<<endl;
}

// ===========================================================
// doubleCheckPhysics-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::doubleCheckPhysics(int t,int offset){
  //km->launchComputeErrors(lbm->d_f,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_err,d_momX_err,d_momY_err,d_energy_err,d_hfhfeq_diff);


  // Write one row of diagnostics to the data file for this time step.
  // For each macroscopic error array (mass, x-momentum, y-momentum,
  // energy, entropy), find its min and max and print them.
  // The row format is:
  //   t  mass_min mass_max  momX_min momX_max  momY_min momY_max
  //      energy_min energy_max  entropy_min entropy_max
  // fflush ensures the file is up to date before gnuplot reads it
  double min_val, max_val;
  fprintf((FILE*)data_file," %d",(int)t);
  km->findMinMax(d_mass_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  km->findMinMax(d_momX_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  km->findMinMax(d_momY_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  km->findMinMax(d_energy_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  km->findMinMax((double*)d_entropy_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e\n",(double)min_val,(double)max_val);
  fflush((FILE*)data_file);
  
  // Tell gnuplot how to display the data file.
  //
  // On the very first call, send the full 'plot' command: column 1 is
  // the time step; the remaining columns come in pairs (min, max) for
  // each macroscopic error (mass, x-momentum, y-momentum, energy,
  // entropy). Each pair is drawn as a line with point markers ('w lp')
  // and labeled in the legend. After that, first_call is set to 0.
  //
  // On later calls, just send 'replot', which redraws the same plot
  // with the updated data file.
  if(first_call==1){
    fprintf((FILE*)gp_pipe,"plot 'conservation&(-restriction)_quantities.dat' using 1:2 w lp title 'mass_min', '' using 1:3 w lp title 'mass_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:4 w lp title 'momX_min', '' using 1:5 w lp title 'momX_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:6 w lp title 'momY_min', '' using 1:7 w lp title 'momY_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:8 w lp title 'energy_min', '' using 1:9 w lp title 'energy_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:10 w lp title 'entropy_min', '' using 1:11 w lp title 'entropy_max'\n");
    fflush((FILE*)gp_pipe);
    first_call = 0;
  }
  else{
    fprintf((FILE*)gp_pipe,"replot\n");
    fflush((FILE*)gp_pipe);
  }
}

// ===========================================================
// countViolations-PHYSICSCHECKER
// ===========================================================
int PHYSICSCHECKER::countViolations(void){
  // Count how many lattice sites have a non-zero violation flag.
  // Copies the violation mask from GPU to host, then scans all Lx*Ly
  // entries and tallies the ones set to 1. Used to report how many
  // cells violate the entropy condition before and after applying ELBM.
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

// ===========================================================
// conservationRestrictionViolations-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::conservationRestrictionViolations(int t,int offset){  
  // Compute conservation and restriction differences  
  km->launchComputeLocalErrors(lbm->d_f,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_diff,d_momX_diff,d_momY_diff,d_energy_diff,d_entropy_diff,offset);

  // Mark entropy violations
  km->launchMarkEntropyViolations((double*)d_entropy_diff,(int*)d_violation_mask,(int)offset);
}

// ===========================================================
// applyELBM-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::applyELBM(int offset){
  // Find omega_eff
  km->launchFindOmegaEff((double*)lbm->d_f,(double*)lbm->d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);

  // Apply collision
  km->launchEntropicCollision((double*)lbm->d_f,(double*)lbm->d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);

  // Compute post-collision macros
  km->launchComputeMacros((double*)lbm->d_f,(double*)lbm->d_rho,(double*)lbm->d_jx,(double*)lbm->d_jy,(double*)lbm->d_rho_e,(double*)lbm->d_h,(int)offset);
}
