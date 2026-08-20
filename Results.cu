// Results.cu
#include "Results.h"

RESULTS::RESULTS(LATTICEBOLTZMANN *Gauss){
  lbm = Gauss;
  
  gp_pipe_rho = popen("gnuplot -persist","w");
  gp_pipe_jx = popen("gnuplot -persist","w");
  gp_pipe_jy = popen("gnuplot -persist","w");
  gp_pipe_rho_e = popen("gnuplot -persist","w");
  gp_pipe_h = popen("gnuplot -persist","w");

  configureGnuplotPipe(gp_pipe_rho,"rho");
  configureGnuplotPipe(gp_pipe_jx,"jx");
  configureGnuplotPipe(gp_pipe_jy,"jy");
  configureGnuplotPipe(gp_pipe_rho_e,"rho_e");
  configureGnuplotPipe(gp_pipe_h,"h");
}

RESULTS::~RESULTS(){
  // close gnuplot pipes
  pclose(gp_pipe_rho);
  pclose(gp_pipe_jx);
  pclose(gp_pipe_jy);
  pclose(gp_pipe_rho_e);
  pclose(gp_pipe_h);
}

void RESULTS::configureGnuplotPipe(FILE *gp_pipe,const char *title){
  fprintf(gp_pipe,"set xlabel 'ix'\n");
  fprintf(gp_pipe,"set ylabel ' '\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe,"set title '%s'\n",title);
  fflush(gp_pipe);
}

/*double RESULTS::analytical_rho(int ix,int iy,int t){
  double sigma2 = sigma*sigma + 2.0*nu*t;
  return RHO0 + amplitude*(sigma*sigma/sigma2)*exp(-((ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly))/(2.0*sigma2));
  }*/

 void RESULTS::writeMacrosData(const char *filename,double *data){
   FILE *tmp = fopen(filename,"w"); 
   int iy=0.5*Ly;
   for(int ix=0;ix<Lx;ix++){
     int idx = ix + iy*Lx;
     fprintf(tmp,"%d %e\n",ix,*(data+idx));
   }
   fclose(tmp);
 }

void RESULTS::plotMacros(int t){
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_rho+0+0*Lx),(const void*)(lbm->d_rho+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_jx+0+0*Lx),(const void*)(lbm->d_jx+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_jy+0+0*Lx),(const void*)(lbm->d_jy+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_rho_e+0+0*Lx),(const void*)(lbm->d_rho_e+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy((void*)(lbm->h_h+0+0*Lx),(const void*)(lbm->d_h+0+0*Lx),(size_t)Lx*Ly*sizeof(double),cudaMemcpyDeviceToHost));

  writeMacrosData("rho_2D.dat",lbm->h_rho);
  writeMacrosData("jx_2D.dat",lbm->h_jx);
  writeMacrosData("jy_2D.dat",lbm->h_jy);
  writeMacrosData("rho_e_2D.dat",lbm->h_rho_e);
  writeMacrosData("h_2D.dat",lbm->h_h);

  fprintf(gp_pipe_rho,"set title 'rho - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_rho,"plot 'rho_2D.dat' with lines lw 0.2\n");
  fflush(gp_pipe_rho); 

  fprintf(gp_pipe_jx,"set title 'jx - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_jx,"plot 'jx_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_jx);

  fprintf(gp_pipe_jy,"set title 'jy - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_jy,"plot 'jy_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_jy);

  fprintf(gp_pipe_rho_e,"set title 'rho_e - t=%d (max=) (min=)'\n", t);
  fprintf(gp_pipe_rho_e,"plot 'rho_e_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_rho_e);

  fprintf(gp_pipe_h,"set title 'h - t=%d (max=, min=)'\n", t);
  fprintf(gp_pipe_h,"plot 'h_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_h);
  
  usleep(1000000);
}


  
