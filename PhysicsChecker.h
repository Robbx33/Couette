// PhysicsChecker.h
#ifndef PHYSICSCHECKER_H
#define PHYSICSCHECKER_H

#include "LBM.h"
#include "KernelsManager.h"
#include <unistd.h>

class PHYSICSCHECKER{
private:
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;
  
  double *h_mass_err,*h_momX_err,*h_momY_err,*h_energy_err,*h_hfhfeq_diff;
  double *d_mass_err,*d_momX_err,*d_momY_err,*d_energy_err,*d_hfhfeq_diff;

  int *d_violation_mask;
  
  FILE *gp_pipe;
  FILE *data_file;
  int first_call;
public:
  PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua);
  ~PHYSICSCHECKER(void);

  //  void writeToFile(int t,const char *name,double min_val,double max_val);
  //void updatePlot();
  void doubleCheckPhysics(int t);
  void markEntropyViolations(int t);
  
  double *get_d_mass_err(void){return d_mass_err;}
  double *get_d_momX_err(void){return d_momX_err;}
  double *get_d_momY_err(void){return d_momY_err;}
  double *get_d_energy_err(void){return d_energy_err;}
  double *get_d_hfhfeq_diff(void){return d_hfhfeq_diff;}
};

#endif
