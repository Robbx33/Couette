// KernelsManager.h     -> class declaration
// iostream  -> std::cout for status messages
// using namespace std lets us write cout instead of std::cout
#include "KernelsManager.h"
#include <iostream>
using namespace std;

// ===========================================================
// CONSTRUCTOR-KernelsManager
// ===========================================================
KernelsManager::KernelsManager(void){
  // Launch configuration for the 2D kernels (one thread per (i,j) lattice site)
  //
  //   blockSize2D  threads per block: THREADS_PER_BLOCK_X by
  //                THREADS_PER_BLOCK_Y by 1
  //   gridSize2D   blocks needed to cover the domain:
  //                (Lx+blockSize2D.x-1)/blockSize2D.x by
  //                (Ly+blockSize2D.y-1)/blockSize2D.y by 1
  blockSize2D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,1);
  gridSize2D  = dim3((Lx+blockSize2D.x-1)/blockSize2D.x,(Ly+blockSize2D.y-1)/blockSize2D.y,1);
  // Launch configuration for the 3D kernels (one thread per (i,j,k) lattice site)
  //
  //   blockSize3D  threads per block: THREADS_PER_BLOCK_X by
  //                THREADS_PER_BLOCK_Y by Q
  //   gridSize3D   blocks needed to cover the domain:
  //                (Lx+blockSize3D.x-1)/blockSize3D.x by
  //                (Ly+blockSize3D.y-1)/blockSize3D.y by
  //                (Q+blockSize3D.z-1)/blockSize3D.z
  blockSize3D = dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_Y,Q);
  gridSize3D  = dim3((Lx+blockSize3D.x-1)/blockSize3D.x,(Ly+blockSize3D.y-1)/blockSize3D.y,(Q+blockSize3D.z-1)/blockSize3D.z);
}

// ===========================================================
// DESTRUCTOR-KernelsManager
// ===========================================================
KernelsManager::~KernelsManager(void){
  // KernelsManager checkout
  cout<<"Memory freed KernelsManager(CPU+GPU)."<<endl;
}

// ===========================================================
// launchComputeMacros-KernelsManager
// ===========================================================
void KernelsManager::launchComputeMacros(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,int offset){
  // Launch computeMacrosKernel on a 2D grid (one thread per site) to
  // fill the macros fields from the distribution function f at
  // the given offset (0 = pre-collision, 1 = post-collision).
  // KERNEL_CHECK catches launch failers; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  computeMacrosKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchComputeFeq-KernelsManager
// ===========================================================
void KernelsManager::launchComputeFeq(double *d_rho,double *d_jx,double *d_jy,double *d_feq){
  // Launch computeFeqKernel on a 3D grid (one thread per site (i,j,k)) to
  // compute the equilibrium distribution feq at every site_direction
  // pair form the current macros rho, jx, jy.
  // KERNEL_CHECK catches launch failers; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  computeFeqKernel<<<gridSize3D,blockSize3D>>>((double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_feq);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchComputeCollisionDifferences-KernelsManager
// ===========================================================  
void KernelsManager::launchComputeCollisionDifferences(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff){
  // Launch computeCollisionDifferencesKernel on a 2D grid (one thread
  // per site (i,j)). For each site, it computes the local moments of feq and
  // compares them with the current macros rho, jx, jy, rho_e, h, writing
  // the absolute differences into the five d_*_diff arrays (for mass,
  // x/y-momentum, energy and entropy).
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  computeCollisionDifferencesKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(double*)d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchFindMinMax-KernelsManager
// ===========================================================  
void KernelsManager::launchFindMinMax(double *d_data,double *d_min,double *d_max,int offset){
  // Launch findMinMaxKernel on a 2D grid (one thread per site (i,j)). Each
  // block reduces its own portion of d_data and writes one partial
  // min/max into d_min/d_max, indexed by block id. The caller finishes
  // the reduction on the host. `offset` selects which buffer to read
  // (0 = pre-collision, 1 = post-collision).
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  findMinMaxKernel<<<gridSize2D,blockSize2D>>>((double*)d_data,(double*)d_min,(double*)d_max,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchRenderAndFill-KernelsManager
// ===========================================================  
void KernelsManager::launchRenderAndFill(uchar4 *d_color,double *vbo_positions_ptr,double *vbo_uv_ptr,double *d_data,double center,double scale){
  // Launch renderAndFillKernel on a 2D grid (one thread per site). Each
  // thread writes the color of its site into d_color, and the (x,y,z)
  // position and (u,v) texture coordinates into the mapped VBO buffers.
  // d_data is normalized with center and scale so the colors and the
  // surface height map to a fixed range.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  renderAndfillKernel<<<gridSize2D,blockSize2D>>>((uchar4*)d_color,(double*)vbo_positions_ptr,(double*)vbo_uv_ptr,(double*)d_data,(double)center,(double)scale);
  KERNEL_CHECK();
  SYNC_CHECK();
}

// ===========================================================
// launchCollision-KernelsManager
// ===========================================================  
void KernelsManager::launchCollision(double *d_f,double *d_feq,int offset){
  // Launch collisionKernel on a 3D grid (one thread per (i,j,k)). Each
  // thread applies the BGK collision to one direction of one site:
  // reads f and feq form buffer 0 (offset=0, pre-collision), writes the
  // post-collision f into buffer 1 (offset=1).
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  collisionKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(double*)d_feq,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}

// ===========================================================
// launchComputeLocalDifferences-KernelsManager
// ===========================================================  
void KernelsManager::launchComputeLocalDifferences(double *d_f,double *d_rho,double *d_jx,double *d_jy,double *d_rho_e,double *d_h,double *d_feq,double *d_mass_diff,double *d_momX_diff,double *d_momY_diff,double *d_energy_diff,double *d_entropy_diff,int offset){
  // Launch computeLocalDifferencesKernel on a 2D grid (one thread per
  // site (i,j)). For each site, it computes the difference between the
  // macros at pre-collision (offset=0) and those at post-collision (offset=1)
  // storing the difference values into d_mass_diff, d_momX_diff, d_momY_diff,
  // d_energy_diff, d_entropy_diff.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  computeLocalDifferencesKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_rho,(double*)d_jx,(double*)d_jy,(double*)d_rho_e,(double*)d_h,(double*)d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchMarkEntropyViolations-KernelsManager
// ===========================================================  
void KernelsManager::launchMarkEntropyViolations(double *d_entropy_diff,int *d_violation_mask,int offset){
  // Launch markEntropyViolationsKernel on a 2D grid (one thread per
  // site (i,j)). Each thread reads the entropy difference at the
  // offset=1, and writes 1 into d_violation_mask[id] if the entropy
  // restriction is violated (d_entropy_diff < 0), 0 otherwise.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  markEntropyViolationsKernel<<<gridSize2D,blockSize2D>>>((double*)d_entropy_diff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchFindOmegaEff-KernelsManager
// ===========================================================  
void KernelsManager::launchFindOmegaEff(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  // Launch findOmegaEffKernel on a 2D grid (one thread per site (i,j)). For
  // every site marked in d_violation_mask, the thread solves the entropy
  // condition H(omega_eff) = 0 using a Newton-Bisection hybrid and writes
  // the result into d_omega_eff. Non-violating sites are left untouched.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  findOmegaEffKernel<<<gridSize2D,blockSize2D>>>((double*)d_f,(double*)d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
// ===========================================================
// launchEntropicCollision-KernelsManager
// ===========================================================  
void KernelsManager::launchEntropicCollision(double *d_f,double *d_feq,double *d_omega_eff,int *d_violation_mask,int offset){
  // Launch entropicCollisionKernel on a 3D grid (one thread per
  // site (i,j,k)). Each thread checks its site against d_violation_mask
  // and, if the site is flagged, recomputes its f at the offset=1
  // (post-collision) using omega_eff instead of the BGK omega.
  // Non-violating sites are left as they are.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  entropicCollisionKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(double*)d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}

// ===========================================================
// launchStream-KernelsManager
// ===========================================================  
void KernelsManager::launchStream(double *d_f,int offset){
  // Launch streamKernel on a 3D grid (one thread per (i,j,k)).
  // Each thread move one f value from its source site to the
  // neighbor in direction k. Reads form buffer 'offset=1',
  // writes into buffer 1-offset.
  // KERNEL_CHECK catches launch failures; SYNC_CHECK waits for the
  // kernel to finish and catches runtime errors.
  streamKernel<<<gridSize3D,blockSize3D>>>((double*)d_f,(int)offset);
  KERNEL_CHECK();
  SYNC_CHECK();
}
