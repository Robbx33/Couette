// Results.h
#ifndef RESULTS_H
#define RESULTS_H

#include "LBM.h"
#include <unistd.h>

//class LATTICEBOLTZMANN;

class RESULTS{
 private:
  LATTICEBOLTZMANN *lbm;

  // Gnuplot pipes
  FILE *gp_pipe_density;
  FILE *gp_pipe_ux;
  FILE *gp_pipe_uy;
  FILE *gp_pipe_energy;
  FILE *gp_pipe_h;

  // File names
  char density_file[256];
  char ux_file[256];
  char uy_file[256];
  char energy_file[256];
  char h_file[256];
  
 public:
  RESULTS(LATTICEBOLTZMANN *Noah);
  ~RESULTS();

  // Plot functions
  /*void plotDensity(int t);
  void plotUxVelocity(int t);
  void plotUyVelocity(int t);
  void plotEnergy(int t);*/
  void plotAll(int t);

  // Save functions
  void saveDensity(const char* filename,int t);
  void saveVelocity(const char* filename,int t);
  void saveAll(const char* basename,int t);
  
};

#endif
