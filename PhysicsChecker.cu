// PhysicsChecker.cu
#include "PhysicsChecker.h"
using namespace std;

PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Noah,KernelsManager *kernels){
  lbm = Noah;
  km = kernels;

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
  
  // Initialize gnuplot pipes
  gp_pipe_mass_error = popen("gnuplot -persist","w");
  gp_pipe_momX_error = popen("gnuplot -persist","w");
  gp_pipe_momY_error = popen("gnuplot -persist","w");
  gp_pipe_energy_error = popen("gnuplot -persist","w");
  gp_pipe_hfhfeq_diff = popen("gnuplot -persist","w");

  // Configure each plot
  configureGnuplotPipe(gp_pipe_mass_error,"mass Error Collision");
  configureGnuplotPipe(gp_pipe_momX_error,"momentum X Error Collision");
  configureGnuplotPipe(gp_pipe_momY_error,"momentum Y Error Collision");
  configureGnuplotPipe(gp_pipe_energy_error,"internal energy Error Collision");
  configureGnuplotPipe(gp_pipe_hfhfeq_diff,"sum[f*log(f/feq)]-sum[feq*Log(f/feq)] Error Collision");
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

  // close gnuplot pipes
  pclose(gp_pipe_mass_error);
  pclose(gp_pipe_momX_error);
  pclose(gp_pipe_momY_error);
  pclose(gp_pipe_energy_error);
  pclose(gp_pipe_hfhfeq_diff);
}

void PHYSICSCHECKER::configureGnuplotPipe(FILE *gp_pipe,const char *title){
  fprintf(gp_pipe,"set xlabel 'ix'\n");
  fprintf(gp_pipe,"set ylabel 'iy'\n");
  fprintf(gp_pipe,"set zlabel 'Error'\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe,"set hidden3d\n");
  fprintf(gp_pipe,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fprintf(gp_pipe,"set title '%s'\n",title);
  fflush(gp_pipe);
}

void PHYSICSCHECKER::computeErrorStats(double *array,double &max_val,double &min_val){
  max_val = 0.0;
  min_val = 0.0;
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      if(*(array+idx)>max_val){
	max_val = *(array+idx);
      }
      if(*(array+idx)<min_val){
	min_val = *(array+idx);
      }
    }
  }
}

void PHYSICSCHECKER::writeErrorData(const char *filename,double *data){
  FILE *tmp = fopen(filename,"w");
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      fprintf(tmp,"%d %d %e\n",ix,iy,*(data+idx));
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
}

void PHYSICSCHECKER::plotCollisionConservation_RestrictionErrorsGPU(int t){
  // 1. Compute errors on GPU
  km->launchComputeErrors(lbm->d_f,lbm->d_Cx,lbm->d_Cy,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_err,d_momX_err,d_momY_err,d_energy_err,d_hfhfeq_diff);

  // 2. Copy error arrays to CPU
  CUDA_CHECK(cudaMemcpy((void*)(h_mass_err+0+0*Lx),(const void*)(d_mass_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_momX_err+0+0*Lx),(const void*)(d_momX_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_momY_err+0+0*Lx),(const void*)(d_momY_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_energy_err+0+0*Lx),(const void*)(d_energy_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_hfhfeq_diff+0+0*Lx),(const void*)(d_hfhfeq_diff+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));

  // 3. Compute max/min values
  double max_mass,min_mass;
  double max_momX,min_momX;
  double max_momY,min_momY;
  double max_energy,min_energy;
  double max_hfhfeq,min_hfhfeq;

  computeErrorStats(h_mass_err,max_mass,min_mass);
  computeErrorStats(h_momX_err,max_momX,min_momX);
  computeErrorStats(h_momY_err,max_momY,min_momY);
  computeErrorStats(h_energy_err,max_energy,min_energy);
  computeErrorStats(h_hfhfeq_diff,max_hfhfeq,min_hfhfeq);

  // 4. Write data files (sampled every 2 points)
  writeErrorData("mass_error_3D.dat",h_mass_err);
  writeErrorData("momX_error_3D.dat",h_momX_err);
  writeErrorData("momY_error_3D.dat",h_momY_err);
  writeErrorData("energy_error_3D.dat",h_energy_err);
  writeErrorData("hfhfeq_diff_3D.dat",h_hfhfeq_diff);
  
  // 5. Plot mass error in its own window
  fprintf(gp_pipe_mass_error,"set title 'Collision Mass Conservation Error - t=%d (max=%e) (min=%e)'\n",t,max_mass,min_mass);
  fprintf(gp_pipe_mass_error,"splot 'mass_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_mass_error); 
  // Plot momentum X error in its own window
  fprintf(gp_pipe_momX_error,"set title '|Collision Momentum X Error| - t=%d (max=%e) (min=%e)'\n",t,max_momX,min_momX);
  fprintf(gp_pipe_momX_error,"splot 'momX_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momX_error);
  // Plot momentum Y error in its own window
  fprintf(gp_pipe_momY_error,"set title '|Collision Momentum Y Error| - t=%d (max=%e) (min=%e)'\n",t,max_momY,min_momY);
  fprintf(gp_pipe_momY_error,"splot 'momY_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momY_error);
  // internal energy error
  fprintf(gp_pipe_energy_error,"set title 'Collision Internal Energy Error - t=%d (max=%e) (min=%e)'\n", t,max_energy,min_energy);
  fprintf(gp_pipe_energy_error,"splot 'energy_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_energy_error);
  // Entropy Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)]
  fprintf(gp_pipe_hfhfeq_diff,"set title 'Collision h-function Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)] - t=%d (max=%e, min=%e)'\n", t,max_hfhfeq,min_hfhfeq);
  fprintf(gp_pipe_hfhfeq_diff,"splot 'hfhfeq_diff_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_hfhfeq_diff);
  
  usleep(2000000);
}

/*void PHYSICSCHECKER::printSummary(){
  // Print one-line summary of current diagnostics
  // You can implement this to show max values
  cout<<"Diagnostics OK"<<endl;
}

void PHYSICSCHECKER::printReport(){
  cout<<"Collision Mass Conservation: OK"<<endl;
  cout<<"Collision Momentum X Conservation: OK"<<endl;
  cout<<"Collision Momentum Y Conservation: OK"<<endl;
  cout<<"Collision Energy Conservation: OK"<<endl;
  cout<<"Collision Entropy Restriction: OK"<<endl;
  }*/
