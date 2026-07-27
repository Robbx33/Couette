// PhysicsChecker.h
#ifndef PHYSICSCHECKER_H
#define PHYSICSCHECKER_H

#include "LBM.h"
#include "KernelsManager.h"
#include <cstdio>

//class LATTICEBOLTZMANN;

class PHYSICSCHECKER{
private:
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;
  
  double *h_mass_err,*h_momX_err,*h_momY_err,*h_energy_err,*h_hfhfeq_diff;
  double *d_mass_err,*d_momX_err,*d_momY_err,*d_energy_err,*d_hfhfeq_diff;
  
  FILE *gp_pipe_mass_error;
  FILE *gp_pipe_momX_error;
  FILE *gp_pipe_momY_error;
  FILE *gp_pipe_energy_error;
  FILE *gp_pipe_hfhfeq_diff;
  
public:
  PHYSICSCHECKER(LATTICEBOLTZMANN *Noah,KernelsManager *kernels);
  ~PHYSICSCHECKER();
  void plotCollisionConservation_RestrictionErrorsCPU(int t);
  void plotCollisionConservation_RestrictionErrorsGPU(int t);
  /*void printSummary();
    void printReport();*/
};

#endif
