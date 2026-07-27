// Results.cu
#include "Results.h"
#include "LBM.h"
#include "Constants.h"
#include <unistd.h>
#include <cmath>

RESULTS::RESULTS(LATTICEBOLTZMANN *Noah){
  lbm = Noah;

  /*system("pkill gnuplot 2>/dev/null");
    usleep(300000);*/
  
  // Gnuplot pipes
  gp_pipe_density  = popen("gnuplot -persist","w");
  gp_pipe_ux       = popen("gnuplot -persist","w");
  gp_pipe_uy       = popen("gnuplot -persist","w");
  gp_pipe_energy   = popen("gnuplot -persist","w");
  gp_pipe_h        = popen("gnuplot -persist","w");
  // Configure density plot
  fprintf(gp_pipe_density,"set xlabel 'ix'\n");
  fprintf(gp_pipe_density,"set ylabel 'iy'\n");
  fprintf(gp_pipe_density,"set zlabel 'Density'\n");
  fprintf(gp_pipe_density,"set grid\n");
  fprintf(gp_pipe_density,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_density,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_density,"set hidden3d\n");
  fprintf(gp_pipe_density,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000', 1 '#FF0000')\n");
  fflush(gp_pipe_density);
  
  // Configure ux plot
  fprintf(gp_pipe_ux,"set xlabel 'ix'\n");
  fprintf(gp_pipe_ux,"set ylabel 'iy'\n");
  fprintf(gp_pipe_ux,"set zlabel 'ux'\n");
  fprintf(gp_pipe_ux,"set grid\n");
  fprintf(gp_pipe_ux,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_ux,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_ux,"set hidden3d\n");
  fprintf(gp_pipe_ux,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000')\n");
  fflush(gp_pipe_ux);

  // Configure uy plot
  fprintf(gp_pipe_uy,"set xlabel 'ix'\n");
  fprintf(gp_pipe_uy,"set ylabel 'iy'\n");
  fprintf(gp_pipe_uy,"set zlabel 'uy'\n");
  fprintf(gp_pipe_uy,"set grid\n");
  fprintf(gp_pipe_uy,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_uy,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_uy,"set hidden3d\n");
  fprintf(gp_pipe_uy,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000')\n");
  fflush(gp_pipe_uy);
  
  // Configure energy plot
  fprintf(gp_pipe_energy,"set xlabel 'ix'\n");
  fprintf(gp_pipe_energy,"set ylabel 'iy'\n");
  fprintf(gp_pipe_energy,"set zlabel 'Internal Energy'\n");
  fprintf(gp_pipe_energy,"set grid\n");
  fprintf(gp_pipe_energy,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_energy,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_energy,"set hidden3d\n");
  fprintf(gp_pipe_energy,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000')\n");
  fflush(gp_pipe_energy);

    // Configure energy plot
  fprintf(gp_pipe_h,"set xlabel 'ix'\n");
  fprintf(gp_pipe_h,"set ylabel 'iy'\n");
  fprintf(gp_pipe_h,"set zlabel 'h'\n");
  fprintf(gp_pipe_h,"set grid\n");
  fprintf(gp_pipe_h,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_h,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_h,"set hidden3d\n");
  fprintf(gp_pipe_h,"set palette defined (0 '#0000FF', 0.5 '#00FF00', 1 '#FF0000')\n");
  fflush(gp_pipe_h);
  
  // Set file names
  snprintf(density_file,sizeof(density_file),"density_3D.dat");
  snprintf(ux_file,sizeof(ux_file),"Ux_velocity_3D.dat");
  snprintf(uy_file,sizeof(uy_file),"Uy_velocity_3D.dat");
  snprintf(energy_file,sizeof(energy_file),"energy_3D.dat");
  snprintf(h_file,sizeof(h_file),"h_function_3D.dat");
}

RESULTS::~RESULTS(){
  pclose(gp_pipe_density);
  pclose(gp_pipe_ux);
  pclose(gp_pipe_uy);
  pclose(gp_pipe_energy);
  pclose(gp_pipe_h);
}

/*void RESULTS::plotDensity(int t){
  FILE *tmp = fopen(density_file,"w");
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      fprintf(tmp,"%d %d %f\n",ix,iy,*(lbm->h_rho+idx));
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
  
  fprintf(gp_pipe_density, "set title 'Density - t=%d'\n",t);
  fprintf(gp_pipe_density, "splot '%s' with points pt 5 ps 0.5 palette\n",density_file);
  fflush(gp_pipe_density);
}

void RESULTS::plotUxVelocity(int t){
  FILE *tmp = fopen(ux_file,"w");
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      double ux = *(lbm->h_jx+idx)/(*(lbm->h_rho+idx));
      fprintf(tmp,"%d %d %f\n",ix,iy,ux);
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
  
  fprintf(gp_pipe_ux,"set title 'Ux Velocity Components - t=%d'\n",t);
  fprintf(gp_pipe_ux,"splot '%s' with points pt 5 ps 0.5 palette\n",ux_file);
  fflush(gp_pipe_ux);
}

void RESULTS::plotUyVelocity(int t){
  FILE *tmp = fopen(uy_file,"w");
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      double uy = *(lbm->h_jy+idx)/(*(lbm->h_rho+idx));
      fprintf(tmp,"%d %d %f\n",ix,iy,uy);
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
  
  fprintf(gp_pipe_uy,"set title 'Uy Velocity Components - t=%d'\n",t);
  fprintf(gp_pipe_uy,"splot '%s' with points pt 5 ps 0.5 palette\n",uy_file);
  fflush(gp_pipe_uy);
}

void RESULTS::plotEnergy(int t){
  FILE *tmp = fopen(energy_file,"w");
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      fprintf(tmp,"%d %d %f\n",ix,iy,*(lbm->h_rho_e+idx));
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
  
  fprintf(gp_pipe_energy,"set title 'Internal Energy - t=%d'\n",t);
  fprintf(gp_pipe_energy,"splot '%s' with points pt 5 ps 0.5 palette\n",energy_file);
  fflush(gp_pipe_energy);
  }*/

void RESULTS::plotAll(int t){
  FILE *tmp_density = fopen(density_file,"w");
  FILE *tmp_ux = fopen(ux_file,"w");
  FILE *tmp_uy = fopen(uy_file,"w");
  FILE *tmp_energy = fopen(energy_file,"w");
  FILE *tmp_h = fopen(h_file,"w");

  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      double rho = *(lbm->h_rho+idx);

      // Density
      fprintf(tmp_density,"%d %d %f\n",ix,iy,rho);

      // Velocity components
      double ux = *(lbm->h_jx+idx)/rho;
      double uy = *(lbm->h_jy+idx)/rho;
      fprintf(tmp_ux,"%d %d %f\n",ix,iy,ux);
      fprintf(tmp_uy,"%d %d %f\n",ix,iy,uy);

      // Internal energy
      fprintf(tmp_energy,"%d %d %f\n",ix,iy,*(lbm->h_rho_e+idx));

      // h = f*log(f)
      double h = 0.0;
      for(int iz=0;iz<Q;iz++){
	double f = *(lbm->h_f+idx+iz*Lx*Ly);
	if(f > 1e-15){
	  h += f*log(f);
	}
      }
      fprintf(tmp_h,"%d %d %f\n",ix,iy,h);
    }
    fprintf(tmp_density,"\n");
    fprintf(tmp_ux,"\n");
    fprintf(tmp_uy,"\n");
    fprintf(tmp_energy,"\n");
    fprintf(tmp_h,"\n");
  }
  fclose(tmp_density);
  fclose(tmp_ux);
  fclose(tmp_uy);
  fclose(tmp_energy);
  fclose(tmp_h);

  // Plot all five
  fprintf(gp_pipe_density,"set title 'Density - t=%d'\n",t);
  fprintf(gp_pipe_density,"splot '%s' with points pt 5 ps 0.5 palette\n",density_file);
  fflush(gp_pipe_density);

  fprintf(gp_pipe_ux,"set title 'Ux Velocity Components - t=%d'\n",t);
  fprintf(gp_pipe_ux,"splot '%s' with points pt 5 ps 0.5 palette\n",ux_file);
  fflush(gp_pipe_ux);
  
  fprintf(gp_pipe_uy,"set title 'Uy Velocity Components - t=%d'\n",t);
  fprintf(gp_pipe_uy,"splot '%s' with points pt 5 ps 0.5 palette\n",uy_file);
  fflush(gp_pipe_uy);

  fprintf(gp_pipe_energy,"set title 'Internal Energy - t=%d'\n",t);
  fprintf(gp_pipe_energy,"splot '%s' with points pt 5 ps 0.5 palette\n",energy_file);
  fflush(gp_pipe_energy);

  fprintf(gp_pipe_h,"set title 'function h=f*log(f) - t=%d'\n",t);
  fprintf(gp_pipe_h,"splot '%s' with points pt 5 ps 0.5 palette\n",h_file);
  fflush(gp_pipe_h);


  
  /*plotDensity((int) t);
  plotUxVelocity((int) t);
  plotUyVelocity((int) t);
  plotEnergy((int) t);
  plotEntropy((int)t);*/
  usleep(2000000);
}

void RESULTS::saveDensity(const char* filename,int t){
  FILE *tmp = fopen(filename,"w");
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      fprintf(tmp,"%d %d %f\n",ix,iy,*(lbm->h_rho+idx));
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
}

void RESULTS::saveVelocity(const char* filename,int t){
  FILE *tmp = fopen(filename,"w");
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      double ux = *(lbm->h_jx+idx)/(*(lbm->h_rho+idx));
      double uy = *(lbm->h_jy+idx)/(*(lbm->h_rho+idx));
      fprintf(tmp,"%d %d %f %f\n",ix,iy,ux,uy);
    }
    fprintf(tmp,"\n");
  }
  fclose(tmp);
}

void RESULTS::saveAll(const char* basename,int t){
  char filename[256];
  snprintf(filename, sizeof(filename),"%s_density_%d.dat",basename,t);
  saveDensity(filename,t);
  snprintf(filename, sizeof(filename),"%s_velocity_%d.dat",basename,t);
  saveVelocity(filename,t);
}
