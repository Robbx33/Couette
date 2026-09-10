// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  PHYSICSCHECKER_H is not defined -> enter, read the class,
//                       and #define PHYSICSCHECKER_H to mark it as "already read".
//   - Any later inclusion: PHYSICSCHECKER_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
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
