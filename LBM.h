// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  LBM_H is not defined -> enter, read the class,
//                       and #define LBM_H to mark it as "already read".
//   - Any later inclusion: LBM_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef LBM_H
#define LBM_H

// The #include lines below pull in the headers needed by this class.
//
//   Constants.h      -> lattice sizes (Lx, Ly, Q) and physical constants
//   error.h          -> CUDA_CHECK / KERNEL_CHECK macros used by LBM.cu
//   KernelsManager.h -> the KernelsManager type used as a pointer member
//
// They are included here (not just in LBM.cu) so that any file that
// includes LBM.h gets everything the LBM implementation needs.
#include "Constants.h"
#include "error.h"
#include "KernelsManager.h"

class LATTICEBOLTZMANN{
private:
  // CPU-side arrays. Used to initialize the simulation and to copy results
  // back from the GPU for analysis or plotting.
  //
  //   h_f       distribution function
  //   h_rho     density
  //   h_jx/jy   momentum density (x and y components)
  //   h_rho_e   internal energy density
  //   h_h       H-function (entropy-related)
  //   h_feq     equilibrium distribution function
  //   h_Cx/Cy   discrete velocity components
  //   h_w       lattice weights
  // -------------------------------------------------------------------------
  double *h_f,*h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;
  int *h_Cx,*h_Cy;
  double *h_w;
  
  // GPU-side arrays. Used during the simulation.
  //
  //   d_f       distribution function
  //   d_rho     density
  //   d_jx/jy   momentum density (x and y components)
  //   d_rho_e   internal energy density
  //   d_h       H-function (entropy-related)
  //   d_feq     equilibrium distribution function
  double *d_f,*d_rho,*d_jx,*d_jy,*d_rho_e,*d_h,*d_feq;
  
  // Pointer to the KernelsManager, used to call its kernel-launching methods
  // (computeMacros, computeFeq, Collision, Stream) via km->.
  KernelsManager *km;

  // Friend classes can access the private members of LATTICEBOLTZMANN.
  // PHYSICSCHECKER and RESULTS need direct access to the device arrays
  // (d_f, d_rho, ...) for diagnostics and plotting.
  friend class PHYSICSCHECKER;
  friend class RESULTS;
public:
  // Lifecycle of the LBM object: setup and cleanup.
  // The constructor takes a pointer to the KernelsManager (used to launch
  // all GPU kernels). The destructor frees host and device memory
  LATTICEBOLTZMANN(KernelsManager *Gargantua);
  ~LATTICEBOLTZMANN(void);

  // GPU-side simulation steps: compute macroscopic moments, compute the
  // equilibrium distribution, perform the collision, and stream.
  void computeMacros(int offset);
  void computeFeq(void);
  void Collision(void);
  void Stream(void);
  
  // Public accessors for the device arrays, so main() can pass them as
  // arguments to the KernelsManager and PhysicsChecker methods
  double *get_d_f(void){return d_f;}
  double *get_d_rho(void){return d_rho;}
  double *get_d_jx(void){return d_jx;}
  double *get_d_jy(void){return d_jy;}
  double *get_d_rho_e(void){return d_rho_e;}
  double *get_d_h(void){return d_h;}
  double *get_d_feq(void){return d_feq;}
};

#endif
