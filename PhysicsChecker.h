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

// LBM.h            -> LATTICEBOLTZMANN class (method implementations)
// KernelsManager.h -> KernelsManager methods used inside this file
// unistd.h         -> usleep (pauses after killing old gnuplot processes)
#include "LBM.h"
#include "KernelsManager.h"
#include <unistd.h>

class PHYSICSCHECKER{
private:
  // On CPU and GPU. Checks for mass, momentum, internal energy conservation and
  // entropy restriction
  double *h_mass_diff,*h_momX_diff,*h_momY_diff,*h_energy_diff,*h_entropy_diff;
  double *d_mass_diff,*d_momX_diff,*d_momY_diff,*d_energy_diff,*d_entropy_diff;

  // On CPU and GPU. Marks the points where there has been a violation of the
  // entropy restriction 
  int *h_violation_mask;
  int *d_violation_mask;

  // On GPU. Changes the particles time without been on a collision τ by τ_eff(x,t)
  // to avoid breaking the entropy restriction  
  double *d_omega_eff;

  // Gnuplot output for the conservation diagnostics
  //
  // gp_pipe    pipe to a gnuplot process; commands are written to it
  //            via fprintf and gnuplot draws the plots
  // data_file  plain text file holding the numeric values (min/max of
  //            each conserved quantity per time step); gnuplot reads it
  // first_call flag: 1 on the first diagnose() call so gnuplot receives
  //            'plot' once, then 0 so later calls send 'replot'
  FILE *gp_pipe;
  FILE *data_file;
  int first_call;

  // Pointers to the other two objects this class works with.
  //
  // lbm points to the Gauss object that points to the start
  //     of the LATTICEBOLTZMANN start of the class
  // km  points to the Gargantua object that points to the start
  //     of the KernelsManager start of the class
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;
public:
  // Lifecycle of the PHYSICSCHECKER object: setup and cleanup.
  // The constructor takes a pointer to the KernelsManager (used to launch
  // all GPU kernels) and other pointer to the LATTICEBOLTZMANN. The
  // destructor frees host and device memory
  PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua);
  ~PHYSICSCHECKER(void);

  // doubleCheckPhysics(t,offset)             writes the min/max of each
  //                                          conservation difference to the
  //                                          data file and updates the gnuplot
  //                                          window
  // countViolations()                        returns how many sites have
  //                                          the entropy-violation flag set
  // conservationRestrictionViolations(t,off) checks entropy after the collision
  //                                          and marks all the ubication where 
  //                                          there is an entropy violation
  // applyELBM(offset)                        finds omega_eff per site and
  //                                          applies the entropic collision
  void doubleCheckPhysics(int t,int offset);
  int countViolations(void);
  void conservationRestrictionViolations(int t,int offset);  
  void applyELBM(int offset);

  // Public accessors for the device arrays, so main() can pass them as
  // arguments to the KernelsManager 
  double *get_d_mass_diff(void){return d_mass_diff;}
  double *get_d_momX_diff(void){return d_momX_diff;}
  double *get_d_momY_diff(void){return d_momY_diff;}
  double *get_d_energy_diff(void){return d_energy_diff;}
  double *get_d_entropy_diff(void){return d_entropy_diff;}
};

#endif
