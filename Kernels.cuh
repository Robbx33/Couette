// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read these declarations twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  KERNELS_CUH is not defined -> enter, read the declarations,
//                       and #define KERNELS_CUH to mark it as "already read".
//   - Any later inclusion: KERNELS_CUH is already defined -> skip everything
//                          until #endif. These declarations are not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef KERNELS_CUH
#define KERNELS_CUH

// The #include lines below pull in the headers needed by this class.
//
//   Constants.h      -> lattice sizes (Lx, Ly, Q) and physical constants
#include "Constants.h"

// D2Q9 lattice constants store in CUDA constant memory, so every
// thread reads them from the same chached location instead of the
// global memory.
//
//   d_Cx[k] x component of the k-th discrete velocity
//   d_Cy[k] y component of the k-th discrete velocity
//   d_w[k]  equilibrium weight of the k-th direction
// The nine directions, in order:
//   k = 0 : ( 0,  0)  rest
//   k = 1 : ( 1,  0)  east
//   k = 2 : ( 0,  1)  north
//   k = 3 : (-1,  0)  west
//   k = 4 : ( 0, -1)  south
//   k = 5 : ( 1,  1)  north-east
//   k = 6 : (-1,  1)  north-west
//   k = 7 : (-1, -1)  south-west
//   k = 8 : ( 1, -1)  south-east
__constant__ int d_Cx[Q] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
__constant__ int d_Cy[Q] = {0, 0, 1, 0,-1, 1, 1,-1,-1};
__constant__ double d_w[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

// Forward declarations of the two kernels that compute the macros and
// the equilibrium distribution.
//
//   computeMacros Kernel  fills the macroscopic moments rho jx, jy,
//                         rho_e, h from f at the give offset
//   computeFeqKernel      fills the equilibrium distribution feq from
//                         rho, jx, jy
__global__ void computeMacrosKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e, double *d_h,int offset);
__global__ void computeFeqKernel(double *d_rho,double *d_jx,double *d_jy,double *d_feq);

// Forward declarations of the dagnostics and rendering kernels.
//
//   computeCollisionDifferencesKernel  compare the current macros with
//                                      the local moments of feq and writes
//                                      the differences into the five d_*_diff arrays
//   findMinMaxKernel                   per_block partial min/max of
//                                      d_data at the given offset
//   renderAndfillKernel                fills d_color and the mapped VBOs for
//                                      the 3D surface.
__global__ void computeCollisionDifferencesKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff);
__global__ void findMinMaxKernel(double *d_data,double *d_min,double *d_max,int offset);
__global__ void renderAndfillKernel(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale);

// Forward declaration of the BGK collision kernel: reads f and feq
// and writes the post-collision f into buffer offset = 1. 
__global__ void collisionKernel(double *d_f,double *d_feq,int offset);

// Forward declarations of the Entropic-LBM (ELBM) correction kernel
//
//   computeLocalDifferencesKernel  difference between the macros at pre-collision
//                                  (offset=0) and those at post-collision (offset=1)
//   markEntropyViolationsKernel    flags sites where the entropy restriction is
//                                  violated
//   findOmegaEffKernel             solves H(omega_eff) = 0 at each flagged site
//   entropicCollisionKernel        redoes the collision at the flagged sites with
//                                  omega_eff instead of the BGK omega.
__global__ void computeLocalDifferencesKernel(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset);
__global__ void markEntropyViolationsKernel(double *d_entropy_diff,int *d_violation_mask,int offset);
__global__ void findOmegaEffKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);
__global__ void entropicCollisionKernel(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset);

// Forward declaration of the streaming kernel: moves each f value to
// the neighbor in direction k. REads from buffer post-collision (offset=1),
// writes into buffer 1-offset.
__global__ void streamKernel(double *d_f,int offset);

#endif
