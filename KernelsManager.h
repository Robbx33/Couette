// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  KERNELSMANAGER_H is not defined -> enter, read the class,
//                       and #define KERNELSMANAGER_H to mark it as "already read".
//   - Any later inclusion: KERNELSMANAGER_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef KERNELSMANAGER_H
#define KERNELSMANAGER_H

// Constants.h -> Lx,Ly,Q, THREADS_PER_BLOCK_*, and physical constants
// error.h     -> CUDA_CHECK / KERNEL_CHECK / SYNC_CHECK macros
// Kernels.cuh -> __constant__ arrays (d_Cx,d_Cy,d_w) and the declarations
//                of all the kernels this file defines
#include "Constants.h"
#include "error.h"
#include "Kernels.cuh"

class KernelsManager{
private:
  // Launch configurations for the kernels. Computed once in the
  // constructor and reused by every lauch, since the grid and block
  // sizes never change during a run.
  //
  //   blockSize2D,gridSize2D  used by kernels with one thread per
  //                           lattice site (i,j)
  //   blockSize3D,gridSize3D  used by kernels with one thread per
  //                           lattice site (i,j,k)
  dim3 blockSize2D,gridSize2D;
  dim3 blockSize3D,gridSize3D;
public:
  // Lifecycle of the KernelsManager object: setup and cleanup.
  KernelsManager(void);
  ~KernelsManager(void);
  
  // Launchers for the two kernels that compute derived fields from the
  // distribution function.
  //
  //   launchComputeMacros
  //       Computes the macroscopic moments rho, jx, jy, rho_e, h at
  //       every site, reading from the given offset (0=pre-collision,
  //       1 = post-collision).
  //
  //   launchComputeFeq
  //       Computes the equilibrium distribution feq at every site from
  //       rho, jx, jy (the current macros).
  void launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset);
  void launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq);
  
  // Launchers for the diagnostics and rendering kerneles.
  //
  //   launchComputeCollisionDifferences
  //       Computes, per site, the difference between the current macros
  //       (rho, jx, jy, rho_e, h) and their equilibrium values derived
  //       from feq. Writes into the five d_*_diff arrays.
  //
  //   launchFindMinMax
  //       Reduces d_data over the whole grid. Each block writes its own
  //       partial min/max into d_min and d_max; the caller finishes the
  //       reduction on the host.
  //
  //   launchRenderAndFill
  //       Fills the color buffer and the mapped VBOs (positions and UVs)
  //       for the 3D surface, normalizing d_data with center and scale.
  void launchComputeCollisionDifferences(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff);
  void launchFindMinMax(double *d_data,double *d_min,double *d_max,int offset);
  void launchRenderAndFill(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale);
  
  // Launcher for the BGK collision kernel. Reads f and feq at every
  // site and writes the post_collision f into buffer 'offset'
  // (0 = pre-collision, 1 = post-collision).
  void launchCollision(double *d_f,double *d_feq,int offset);
  
  // Launchers for the Entropic-LBM (ELBM) correction path
  //
  //   launchComputeLocalDifferences
  //       Computes, per site, the differences between the macros at the
  //       pre-collision (offset=0) and those at post-collision (offset=1)
  //       (mass, x/y-momentum, energy and entropy).
  //
  //   launchMarkEntropyViolations
  //       Scans the entropy difference with respect to time and sets
  //       d_violation_mask[id] = 1 where the entropy restriction is
  //       violated (d_entropy_diff < 0), 0 elsewhere.
  //
  //   launchFindOmegaEff
  //       For each site marked in d_violation_mask, solves the entropy
  //       condition H(omega_eff) = 0 with a Newton-Bisection hybrid and
  //       stores the result in d_omega_eff. Non_violating sites are
  //       untouched.
  //
  //   launchEntropicCollision
  //       Redo the collision at the violating sites using omega_eff
  //       instead of the BGK omega, so the entropy restriction is
  //       respected there. Non_violating sites are left as they are.
  void launchComputeLocalDifferences(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double* d_h,double *d_feq,double *d_mass_dif,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset);
  void launchMarkEntropyViolations(double *d_energy_diff,int *d_violation_mask,int offset);
  void launchFindOmegaEff(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  void launchEntropicCollision(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
  
  // Launcher for the streaming kernel. Each thread moves one f value
  // from its source site to the neighbor in direction k, reading from buffer
  // offset = 1 and writing to the buffer 1 - offset = 0. 
  void launchStream(double *d_f,int offset);
};

#endif
