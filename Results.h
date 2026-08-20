// Results.h
#ifndef RESULTS_H
#define RESULTS_H

#include "LBM.h"
#include <unistd.h>

class RESULTS{
 private:
  LATTICEBOLTZMANN *lbm;

  FILE *gp_pipe_rho;
  FILE *gp_pipe_jx;
  FILE *gp_pipe_jy;
  FILE *gp_pipe_rho_e;
  FILE *gp_pipe_h;
 public:
  RESULTS(LATTICEBOLTZMANN *Gauss);
  ~RESULTS();
  
  // Plot functions
  void configureGnuplotPipe(FILE *gp_pipe,const char *title);
  //double analytical_rho(int ix,int iy,int t);
  void writeMacrosData(const char *filename,double *data);
  void plotMacros(int t);

};

#endif
