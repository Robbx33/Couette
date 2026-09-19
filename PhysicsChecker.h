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
#include "LBM.h"

class PHYSICSCHECKER{
private:
  // On GPU. Checks for mass, momentum, internal energy conservation and
  // entropy restriction
  double *d_mass_diff,*d_momX_diff,*d_momY_diff,*d_energy_diff,*d_entropy_diff;

  // On CPU and GPU. Marks the points where there has been a violation of the
  // entropy restriction 
  int *h_violation_mask;
  int *d_violation_mask;

  // On GPU. Changes the particles time without been on a collision τ by τ_eff(x,t)
  // to avoid breaking the entropy restriction  
  double *d_omega_eff;

  // Pointers to the other two objects this class works with.
  //
  // lbm points to the Gauss object that points to the start
  //     of the LATTICEBOLTZMANN start of the class
  // km  points to the Gargantua object that points to the start
  //     of the KernelsManager start of the class
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;

  // Friend classes can access the private members of PHYSICSCHECKER.
  // Visualizer needs direct access to the device arrays
  // (d_mass_diff, d_momX_diff, ...) for plotting.
  // RESULTS needs direct acces to the device arrays
  // (d_mass_diff, d_momX_diff, ...) for plotting.
  friend class Visualizer;
  friend class RESULTS;
public:
  // Lifecycle of the PHYSICSCHECKER object: setup and cleanup.
  // The constructor takes a pointer to the KernelsManager (used to launch
  // all GPU kernels) and other pointer to the LATTICEBOLTZMANN. The
  // destructor frees host and device memory
  PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua);
  ~PHYSICSCHECKER(void);

  // collisionConservationRestrictionDifferences(t)
  //     Launches the kernel that computes, per site, the difference between the
  //     current macroscopic fiels and their equilibrium values (mass, x-momentum,
  //     y-momentum, energy), plus the entropy restriction check.
  //
  // countViolations()
  //     Returns how many sites have the entropy-violation flag set
  //
  // conservationRestrictionViolations(t,off)
  //     Checks entropy after the collision and marks all the ubication
  //     where there is an entropy violation
  //
  // applyELBM(offset)
  //     Finds omega_eff per site and applies the entropic collision
  void collisionConservationRestrictionDifferences(int t);
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
