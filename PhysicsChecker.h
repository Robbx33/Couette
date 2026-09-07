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
  
  double *h_mass_diff,*h_momX_diff,*h_momY_diff,*h_energy_diff,*h_entropy_diff;
  double *d_mass_diff,*d_momX_diff,*d_momY_diff,*d_energy_diff,*d_entropy_diff;

  int *h_violation_mask;
  int *d_violation_mask;
  double *d_omega_eff;
  
  FILE *gp_pipe;
  FILE *data_file;
  int first_call;
public:
  PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua);
  ~PHYSICSCHECKER(void);

  void entropyLocalConservation(int t,int offset);
  
  void doubleCheckPhysics(int t);
  int countViolations(void);  
  void applyELBM(int offset);
  
  double *get_d_mass_diff(void){return d_mass_diff;}
  double *get_d_momX_diff(void){return d_momX_diff;}
  double *get_d_momY_diff(void){return d_momY_diff;}
  double *get_d_energy_diff(void){return d_energy_diff;}
  double *get_d_entropy_diff(void){return d_entropy_diff;}
};

#endif
