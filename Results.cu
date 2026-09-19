// Results.h     -> class declaration
// iostream  -> std::cout for status messages
// using namespace std lets us write cout instead of std::cout
#include "Results.h"
#include <iostream>
using namespace std;

// ===========================================================
// CONSTRUCTOR-RESULTS
// ===========================================================
RESULTS::RESULTS(LATTICEBOLTZMANN *Gauss,PHYSICSCHECKER *Noah,KernelsManager *Gargantua){
  // Point lbm at the begining of LATTICEBOLTZMANN class making it equal
  // to its respective object passed in. pck points to the begining of the
  // class PHYSICSCHECKER and km points to the KernelsManager class
  lbm = Gauss;
  pck = Noah;
  km = Gargantua;

  // Scratch buffers for findMinMax. The reduction works in two stages:
  // first the kernel writes one partial min/max per block, then the CPU
  // finishes the reduction over those partials. So we need one slot per
  // block on both the device and the host.
  //
  //   grid_size_reduce        number of blocks in the 2D launch, i.e.
  //                           ceil(Lx / THREADS_PER_BLOCK_X) * ceil(Ly / THREADS_PER_BLOCK_Y)
  //   h_min_temp, h_max_temp  host copies of the per-block partials,
  //                           for the final reduction loop on the CPU
  //   d_min_temp, d_max_temp  device buffers that receive the per-block
  //                           partials from findMinMaxKernel
  int gx = (Lx+THREADS_PER_BLOCK_X-1)/THREADS_PER_BLOCK_X;
  int gy = (Ly+THREADS_PER_BLOCK_Y-1)/THREADS_PER_BLOCK_Y;
  grid_size_reduce = gx*gy;
  h_min_temp = (double*)malloc(grid_size_reduce*sizeof(double));
  h_max_temp = (double*)malloc(grid_size_reduce*sizeof(double));
  CUDA_CHECK(cudaMalloc((void**)&d_min_temp,grid_size_reduce*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_max_temp,grid_size_reduce*sizeof(double)));

  // Open the data file that gnuplot reads for the conservation and
  // restriction plot. writeConservationREstriction appends one row per
  // timestep; Visualizer::plotConservationRestriction then tells gnuplot
  // to draw it. Kept open for the whole run and closed in the destructor.
  data_file = fopen("conservation&restriction_quantities.dat","w");
}

// ===========================================================
// DESTRUCTOR-RESULTS
// ===========================================================
RESULTS::~RESULTS(){
  // Release the scratch buffers allocated in the constructor for the
  // two-stage min/max reduction (per-block partials on the device and
  // their host copies).  
  free(h_min_temp);
  free(h_max_temp);
  CUDA_CHECK(cudaFree(d_min_temp));
  CUDA_CHECK(cudaFree(d_max_temp));
  
  // RESULTS checkout
  cout<<"Memory freed RESULTS(CPU+GPU)."<<endl;
}

// ===========================================================
// findMinMax-RESULTS
// ===========================================================
void RESULTS::findMinMax(double *d_data,double *min_val,double *max_val,int offset){
  // Compute the minimum and maximum of d_data over the whole grid at
  // the given offset (buffer). Runs in two stages:
  //
  //   1. km->launchFindMinMax launches findMinMaxKernel, which reduces
  //      within each block and writes one partial min/max per block
  //      into d_min_temp / d_max_temp.
  //   2. The per-block partials are copied to the host and reduced
  //      again in a plain loop, giving the global min/max.
  //
  // Used by writeConservationRestriction for the diagnostics rows, and
  // by Visualizer for the color and height scaling of the 3D surface.
  km->launchFindMinMax((double*)d_data,(double*)d_min_temp,(double*)d_max_temp,(int)offset);
  CUDA_CHECK(cudaMemcpy((void*)(h_min_temp+0),(const void*)(d_min_temp+0),(size_t)grid_size_reduce*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_max_temp+0),(const void*)(d_max_temp+0),(size_t)grid_size_reduce*sizeof(double),cudaMemcpyDeviceToHost));

  *min_val = *(h_min_temp+0);
  *max_val = *(h_max_temp+0);
  for(int idx=0;idx<grid_size_reduce;idx++){
    if(*(h_min_temp+idx)<*min_val){
      *min_val = *(h_min_temp+idx);
    }
    if(*(h_max_temp+idx)>*max_val){
      *max_val = *(h_max_temp+idx);
    }
  }
}

// ===========================================================
// writeConservationRestriction-RESULTS
// ===========================================================
void RESULTS::writeConservationRestriction(int t, int offset){
  // Write one row of diagnostics to the data file for this time step.
  // For each macroscopic error array (mass, x-momentum, y-momentum,
  // energy, entropy), find its min and max and print them.
  // The row format is:
  //   t  mass_min mass_max  momX_min momX_max  momY_min momY_max
  //      energy_min energy_max  entropy_min entropy_max
  // fflush ensures the file is up to date before gnuplot reads it
  double min_val, max_val;
  fprintf((FILE*)data_file," %d",(int)t);
  findMinMax((double*)pck->d_mass_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  findMinMax((double*)pck->d_momX_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  findMinMax((double*)pck->d_momY_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  findMinMax((double*)pck->d_energy_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e",(double)min_val,(double)max_val);
  findMinMax((double*)pck->d_entropy_diff,(double*)&min_val,(double*)&max_val,(int)offset);
  fprintf((FILE*)data_file," %e %e\n",(double)min_val,(double)max_val);
  fflush((FILE*)data_file);
}

/*double RESULTS::analytical_rho(int ix,int iy,int t){
  double sigma2 = sigma*sigma + 2.0*nu*t;
  return RHO0 + amplitude*(sigma*sigma/sigma2)*exp(-((ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly))/(2.0*sigma2));
  }*/

// ===========================================================
// writeMacroData-RESULTS
// ===========================================================
 void RESULTS::writeMacroData(const char *filename,double *data){
   // Write one row of the domain to a .dat file for gnuplot.
   // Takes a horizontal cut at iy=Ly/2 and writes, for each ix,
   // a line "ix <value>".Called once per macro field by write Macros
   FILE *tmp = fopen(filename,"w"); 
   int iy=0.5*Ly;
   for(int ix=0;ix<Lx;ix++){
     int idx = ix + iy*Lx;
     fprintf(tmp,"%d %e\n",ix,*(data+idx));
   }
   fclose(tmp);
 }

// ===========================================================
// writeMacros-RESULTS
// ===========================================================
void RESULTS::writeMacros(int t){
  // Dump the current macroscopic fields to a disk for plotting.
  // Copies rho jx, jy, rho_e, and h from GPU (d_*) to host (h_*).
  // then writes each one to its own .dat file via writeMacroData.
  // Visualizer::plotMacros calls this before telling gnuplot to plot,
  // so gnuplot always reads fresh data.
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_rho+0+0*Lx),(const void*)(lbm->d_rho+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_jx+0+0*Lx),(const void*)(lbm->d_jx+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_jy+0+0*Lx),(const void*)(lbm->d_jy+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_rho_e+0+0*Lx),(const void*)(lbm->d_rho_e+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_h+0+0*Lx),(const void*)(lbm->d_h+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));

  writeMacroData("rho_2D.dat",lbm->h_rho);
  writeMacroData("jx_2D.dat",lbm->h_jx);
  writeMacroData("jy_2D.dat",lbm->h_jy);
  writeMacroData("rho_e_2D.dat",lbm->h_rho_e);
  writeMacroData("h_2D.dat",lbm->h_h);
}


  
