// PhysicsChecker.cu
#include "PhysicsChecker.h"
#include "LBM.h"
#include "Constants.h"
#include "error.h"
#include <unistd.h>
#include <iostream>
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
  
  /*system("pkill gnuplot 2>/dev/null");
    usleep(300000);*/
  
  gp_pipe_mass_error = popen("gnuplot -persist","w");
  gp_pipe_momX_error = popen("gnuplot -persist","w");
  gp_pipe_momY_error = popen("gnuplot -persist","w");
  gp_pipe_energy_error = popen("gnuplot -persist","w");
  gp_pipe_hfhfeq_diff = popen("gnuplot -persist","w");
  // Configure gp_pipe_mass
  fprintf(gp_pipe_mass_error,"set xlabel 'ix'\n");
  fprintf(gp_pipe_mass_error,"set ylabel 'iy'\n");
  fprintf(gp_pipe_mass_error,"set zlabel 'Error'\n");
  fprintf(gp_pipe_mass_error,"set grid\n");
  fprintf(gp_pipe_mass_error,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_mass_error,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_mass_error,"set hidden3d\n");
  fprintf(gp_pipe_mass_error,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_mass_error);
  // Configure gp_pipe_momX
  fprintf(gp_pipe_momX_error,"set xlabel 'ix'\n");
  fprintf(gp_pipe_momX_error,"set ylabel 'iy'\n");
  fprintf(gp_pipe_momX_error,"set zlabel 'Error'\n");
  fprintf(gp_pipe_momX_error,"set grid\n");
  fprintf(gp_pipe_momX_error,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_momX_error,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_momX_error,"set hidden3d\n");
  fprintf(gp_pipe_momX_error,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_momX_error);
  // Configure gp_pipe_momY
  fprintf(gp_pipe_momY_error,"set xlabel 'ix'\n");
  fprintf(gp_pipe_momY_error,"set ylabel 'iy'\n");
  fprintf(gp_pipe_momY_error,"set zlabel 'Error'\n");
  fprintf(gp_pipe_momY_error,"set grid\n");
  fprintf(gp_pipe_momY_error,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_momY_error,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_momY_error,"set hidden3d\n");
  fprintf(gp_pipe_momY_error,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_momY_error);
  // Configure gp_pipe_energy (internal energy error)
  fprintf(gp_pipe_energy_error,"set xlabel 'ix'\n");
  fprintf(gp_pipe_energy_error,"set ylabel 'iy'\n");
  fprintf(gp_pipe_energy_error,"set zlabel 'Error'\n");
  fprintf(gp_pipe_energy_error,"set grid\n");
  fprintf(gp_pipe_energy_error,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_energy_error,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_energy_error,"set hidden3d\n");
  fprintf(gp_pipe_energy_error,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_energy_error);
  // Configure gp_pipe_entropy (internal energy error)
  fprintf(gp_pipe_hfhfeq_diff,"set xlabel 'ix'\n");
  fprintf(gp_pipe_hfhfeq_diff,"set ylabel 'iy'\n");
  fprintf(gp_pipe_hfhfeq_diff,"set zlabel 'H(f)-H(feq)'\n");
  fprintf(gp_pipe_hfhfeq_diff,"set grid\n");
  fprintf(gp_pipe_hfhfeq_diff,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_hfhfeq_diff,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_hfhfeq_diff,"set hidden3d\n");
  fprintf(gp_pipe_hfhfeq_diff,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_hfhfeq_diff);
}

PHYSICSCHECKER::~PHYSICSCHECKER(){
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

void PHYSICSCHECKER::plotCollisionConservation_RestrictionErrorsCPU(int t){
  double max_mass_error   = 0.0;
  double max_momX_error   = 0.0;
  double max_momY_error   = 0.0;
  double max_energy_error = 0.0;
  double min_hfhfeq       = 0.0;
  double max_hfhfeq       = 0.0;
  
  // Write mass, momentum, energy errors (sampled every 2 points)
  FILE *tmp_mass_error   = fopen("mass_error_3D.dat","w");
  FILE *tmp_momX_error   = fopen("momX_error_3D.dat","w");
  FILE *tmp_momY_error   = fopen("momY_error_3D.dat","w");
  FILE *tmp_energy_error = fopen("energy_error_3D.dat","w");
  FILE *tmp_hfhfeq_diff  = fopen("hfhfeq_diff_3D.dat","w");
  
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      
      double rho_f     = *(lbm->h_rho+idx);
      double rho_feq   = 0.0;
      double jx_f      = *(lbm->h_jx+idx);
      double jx_feq    = 0.0;
      double jy_f      = *(lbm->h_jy+idx);
      double jy_feq    = 0.0;
      double rhoE_f    = *(lbm->h_rho_e+idx);
      double rhoE_feq  = 0.0;
      
      //Entropy: sum f*log(f/feq) >= sum feq*log(f/feq)
      double hf   = 0.0;
      double hfeq = 0.0;
      
      for(int iz=0;iz<Q;iz++){
	rho_feq  += *(lbm->h_feq+idx+iz*Lx*Ly);
	jx_feq   += *(lbm->h_Cx+iz)*(*(lbm->h_feq+idx+iz*Lx*Ly));
	jy_feq   += *(lbm->h_Cy+iz)*(*(lbm->h_feq+idx+iz*Lx*Ly));
	rhoE_feq += 0.5 * (*(lbm->h_Cx+iz)*(*(lbm->h_Cx+iz)) + *(lbm->h_Cy+iz)*(*(lbm->h_Cy+iz))) * (*(lbm->h_feq+idx+iz*Lx*Ly));
	
	// Entropy terms
	if(*(lbm->h_f+idx+iz*Lx*Ly) > 1e-15 && *(lbm->h_feq+idx+iz*Lx*Ly) > 1e-15){
	  hf   += *(lbm->h_f+idx+iz*Lx*Ly)*(log(*(lbm->h_f+idx+iz*Lx*Ly)/(*(lbm->h_feq+idx+iz*Lx*Ly))));
	  hfeq += *(lbm->h_feq+idx+iz*Lx*Ly)*(log(*(lbm->h_f+idx+iz*Lx*Ly)/(*(lbm->h_feq+idx+iz*Lx*Ly)))); 
	}
      }
      double energy_feq = rhoE_feq-0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq));
      
      double mass_err    = fabs(rho_f-rho_feq);
      double momX_err    = fabs(jx_f-jx_feq);
      double momY_err    = fabs(jy_f-jy_feq);
      double energy_err  = fabs(rhoE_f-energy_feq);
      double hfhfeq_diff = hf-hfeq;// Should be >=0
      if(mass_err > max_mass_error){
	max_mass_error = mass_err;
      }
      if(momX_err > max_momX_error){
	max_momX_error = momX_err;
      }
      if(momY_err > max_momY_error){
	max_momY_error = momY_err;
      }
      if(energy_err > max_energy_error){
	max_energy_error = energy_err;
      }
      if(hfhfeq_diff > max_hfhfeq){
	max_hfhfeq = hfhfeq_diff;
      }
      if(hfhfeq_diff < min_hfhfeq){
	min_hfhfeq = hfhfeq_diff;
      }
      fprintf(tmp_mass_error,"%d %d %e\n",ix,iy,mass_err);
      fprintf(tmp_momX_error,"%d %d %e\n",ix,iy,momX_err);
      fprintf(tmp_momY_error,"%d %d %e\n",ix,iy,momY_err);
      fprintf(tmp_energy_error,"%d %d %e\n",ix,iy,energy_err);
      fprintf(tmp_hfhfeq_diff,"%d %d %e\n",ix,iy,hfhfeq_diff);
    }
    fprintf(tmp_mass_error,"\n");
    fprintf(tmp_momX_error,"\n");
    fprintf(tmp_momY_error,"\n");
    fprintf(tmp_energy_error,"\n");
    fprintf(tmp_hfhfeq_diff,"\n");
  }
  fclose(tmp_mass_error);
  fclose(tmp_momX_error);
  fclose(tmp_momY_error);
  fclose(tmp_energy_error);
  fclose(tmp_hfhfeq_diff);
 
  // Plot mass error in its own window
  fprintf(gp_pipe_mass_error,"set title 'Collision Mass Conservation Error - t=%d (max=%e)'\n",t,max_mass_error);
  fprintf(gp_pipe_mass_error,"splot 'mass_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_mass_error); 
  
  // Plot momentum X error in its own window
  fprintf(gp_pipe_momX_error,"set title '|Collision Momentum X Error| - t=%d (max=%e)'\n",t,max_momX_error);
  fprintf(gp_pipe_momX_error,"splot 'momX_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momX_error);
  
  // Plot momentum Y error in its own window
  fprintf(gp_pipe_momY_error,"set title '|Collision Momentum Y Error| - t=%d (max=%e)'\n",t,max_momY_error);
  fprintf(gp_pipe_momY_error,"splot 'momY_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momY_error);
  
  // Internal energy error
  fprintf(gp_pipe_energy_error,"set title 'Collision Internal Energy Error - t=%d (max=%e)'\n", t, max_energy_error);
  fprintf(gp_pipe_energy_error,"splot 'energy_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_energy_error);
  
  // Entropy Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)]
  fprintf(gp_pipe_hfhfeq_diff,"set title 'Collision h-function Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)] - t=%d (min=%e, max=%e)'\n", t, min_hfhfeq, max_hfhfeq);
  fprintf(gp_pipe_hfhfeq_diff,"splot 'hfhfeq_diff_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_hfhfeq_diff);
  
  usleep(2000000);
}

void PHYSICSCHECKER::plotCollisionConservation_RestrictionErrorsGPU(int t){
  // Compute errors on GPU
  km->launchComputeErrors(lbm->d_f,lbm->d_Cx,lbm->d_Cy,lbm->d_rho,lbm->d_jx,lbm->d_jy,lbm->d_rho_e,lbm->d_h,lbm->d_feq,d_mass_err,d_momX_err,d_momY_err,d_energy_err,d_hfhfeq_diff);

  // Copy error arrays to CPU
  CUDA_CHECK(cudaMemcpy((void*)(h_mass_err+0+0*Lx),(const void*)(d_mass_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_momX_err+0+0*Lx),(const void*)(d_momX_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_momY_err+0+0*Lx),(const void*)(d_momY_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_energy_err+0+0*Lx),(const void*)(d_energy_err+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(h_hfhfeq_diff+0+0*Lx),(const void*)(d_hfhfeq_diff+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));

  // Write data files (sampled every 2 points)
  FILE *tmp_mass_error = fopen("mass_error_3D.dat","w");
  FILE *tmp_momX_error = fopen("momX_error_3D.dat","w");
  FILE *tmp_momY_error = fopen("momY_error_3D.dat","w");
  FILE *tmp_energy_error = fopen("energy_error_3D.dat","w");
  FILE *tmp_hfhfeq_diff = fopen("hfhfeq_diff_3D.dat","w");

  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      fprintf(tmp_mass_error,"%d %d %e\n",ix,iy,*(h_mass_err+idx));
      fprintf(tmp_momX_error,"%d %d %e\n",ix,iy,*(h_momX_err+idx));
      fprintf(tmp_momY_error,"%d %d %e\n",ix,iy,*(h_momY_err+idx));
      fprintf(tmp_energy_error,"%d %d %e\n",ix,iy,*(h_energy_err+idx));
      fprintf(tmp_hfhfeq_diff,"%d %d %e\n",ix,iy,*(h_hfhfeq_diff+idx));
    }
    fprintf(tmp_mass_error,"\n");
    fprintf(tmp_momX_error,"\n");
    fprintf(tmp_momY_error,"\n");
    fprintf(tmp_energy_error,"\n");
    fprintf(tmp_hfhfeq_diff,"\n");
  }
  fclose(tmp_mass_error);
  fclose(tmp_momX_error);
  fclose(tmp_momY_error);
  fclose(tmp_energy_error);
  fclose(tmp_hfhfeq_diff);
  
  // Plot mass error in its own window
  fprintf(gp_pipe_mass_error,"set title 'Collision Mass Conservation Error - t=%d (max=)'\n",t);
  fprintf(gp_pipe_mass_error,"splot 'mass_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_mass_error); 
  
  // Plot momentum X error in its own window
  fprintf(gp_pipe_momX_error,"set title '|Collision Momentum X Error| - t=%d (max=)'\n",t);
  fprintf(gp_pipe_momX_error,"splot 'momX_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momX_error);
  
  // Plot momentum Y error in its own window
  fprintf(gp_pipe_momY_error,"set title '|Collision Momentum Y Error| - t=%d (max=)'\n",t);
  fprintf(gp_pipe_momY_error,"splot 'momY_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momY_error);
  
  // Internal energy error
  fprintf(gp_pipe_energy_error,"set title 'Collision Internal Energy Error - t=%d (max=)'\n", t);
  fprintf(gp_pipe_energy_error,"splot 'energy_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_energy_error);
  
  // Entropy Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)]
  fprintf(gp_pipe_hfhfeq_diff,"set title 'Collision h-function Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)] - t=%d (min=, max=)'\n", t);
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
