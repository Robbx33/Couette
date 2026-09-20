// PhysicsChecker.h     -> class declaration
// iostream  -> std::cout for status messages
// using namespace std lets us write cout instead of std::cout
#include "PhysicsChecker.h"
#include <iostream>
using namespace std;

// ===========================================================
// CONSTRUCTOR-PHYSICSCHECKER
// ===========================================================
PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Gauss,KernelsManager *Gargantua){  
  // Point lbm & km at the begining of their respective classes making
  // them equal to their respective objects passed in.
  lbm = Gauss;
  km = Gargantua;
  
  // On the GPU
  // Checks for mass, momentum, internal energy conservation and entropy restriction
  // Σ Ωᵢ(f(x,t)) = 0                 
  // Σ C_iα.Ωᵢ(f(x,t)) = 0       
  // Σ C_iα.C_iα.Ωᵢ(f(x,t)) = 0 
  // Σ Ln(f_i(x,t)).Ωᵢ(f(x,t)) ≤  0
  // i.e.
  // Σ f_i(x,t) - Σ f_i^eq(x,t) = 0
  // Σ C_iα.f_i(x,t) - Σ C_iα.f_i^eq(x,t) = 0
  // Σ C_iα.C_iα.f_i(x,t) - Σ C_iα.C_iα.f_i^eq(x,t) = 
  // Σ f_i(x,t).Ln(f_i(x,t)/w_i) - Σ f_i^eq(x,t).Ln(f_i^eq(x,t)/w_i) ≤  0
  // which means
  // ρ(x,t) - ρ^eq(x,t) = 0
  // ρ(x,t).u_α(x,t) - ρ^eq(x,t).u_α^eq(x,t) = 0
  // ρ(x,t)E(x,t) - ρ^eq(x,t)E^eq(x,t) = 0
  // h(x,t) - h^eq(x,t) ≤ 0
  //
  // On the GPU buffer area
  // Checks for mass, momentum, internal energy conservation and entropy restriction all at the same point of space x
  // (ρ(x,t+dt)-ρ(x,t))/dt + 0 = 0
  // (ρ(x,t+dt).u_α(x,t+dt)-ρ(x,t).u_α(x,t))/dt + 0 = F_α
  // (ρ(x,t+dt).e(x,t+dt)-ρ(x,t).e(x,t))/dt + 0 = 0
  // (h(x,t+dt)-h(x,t))/dt + 0  ≤  0
  // i.e.
  // ρ(x,t+dt) - ρ(x,t) = 0
  // ρ(x,t+dt).u_α(x,t+dt) - ρ(x,t).u_α(x,t) = F_α 
  // ρ(x,t+dt)e(x,t+dt) - ρ(x,t)e(x,t) = 0
  // h(x,t+dt) - h(x,t) ≤ 0
  // which means
  // Σ f_i(x,t+dt) - Σ f_i(x,t) = 0
  // Σ C_iα.f_i(x,t+dt) - Σ C_iα.f_i(x,t) = F_α
  // Σ C_iα.C_iα.f_i(x,t+dt) - Σ C_iα.C_iα.f_i(x,t) = 0 
  // Σ f_i(x,t+dt).Ln(f_i(x,t+dt)/w_i) - Σ f_i(x,t).Ln(f_i(x,t)/w_i) ≤  0
  CUDA_CHECK(cudaMalloc((void**)&d_mass_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momX_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_momY_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_energy_diff,Lx*Ly*2*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_entropy_diff,Lx*Ly*2*sizeof(double)));

  // On the CPU and GPU. Not all the position on space will be use, just the
  // ones where there are entropy restriction violations
  h_violation_mask = (int*)malloc(Lx*Ly*sizeof(int));  
  CUDA_CHECK(cudaMalloc((void**)&d_violation_mask,Lx*Ly*sizeof(int)));

  // On the GPU. There will be a correction of ω by ω_eff(x,t) on those
  // places where there were entropy restriction violations
  CUDA_CHECK(cudaMalloc((void**)&d_omega_eff,Lx*Ly*sizeof(double)));
}

// ===========================================================
// DESTRUCTOR-PHYSICSCHECKER
// ===========================================================
PHYSICSCHECKER::~PHYSICSCHECKER(void){
  // Liberates GPU memory used by to store Conservative & Restrictive
  // quantities differences
  CUDA_CHECK(cudaFree((double*)d_mass_diff));
  CUDA_CHECK(cudaFree((double*)d_momX_diff));
  CUDA_CHECK(cudaFree((double*)d_momY_diff));
  CUDA_CHECK(cudaFree((double*)d_energy_diff));
  CUDA_CHECK(cudaFree((double*)d_entropy_diff));
  
  // Liberates CPU and GPU memory used to mark entropy violations
  free((int*)h_violation_mask);
  CUDA_CHECK(cudaFree((int*)d_violation_mask));

  // Liberates GPU memory used to store roots of F(omega_eff)=0
  CUDA_CHECK(cudaFree((double*)d_omega_eff));

  // PHYSICSCHECKER checkout
  cout<<"Memory freed PHYSICSCHECKER(CPU+GPU)."<<endl;
}

// ===========================================================
// doubleCheckPhysics-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::collisionConservationRestrictionDifferences(int t){
  // Compute the per-site differences between the current macroscopic
  // fields and their equilibrium values, and write them into the
  // d_*_diff arrays:
  //   d_mass_diff    |rho - rho_eq|
  //   d_momX_diff    |jx  - jx_eq |
  //   d_momY_diff    |jy  - jy_eq |
  //   d_energy_diff  |E   - E_eq  |
  //   d_entropy_diff  h   - h_eq   (signed; < 0 is a violation)
  km->launchComputeCollisionDifferences((double*)lbm->d_f,(double*)lbm->d_rho,(double*)lbm->d_jx,(double*)lbm->d_jy,(double*)lbm->d_rho_e,(double*)lbm->d_h,(double*)lbm->d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff);
 }

// ===========================================================
// countViolations-PHYSICSCHECKER
// ===========================================================
int PHYSICSCHECKER::countViolations(void){
  // Count how many lattice sites have a non-zero violation flag.
  // Copies the violation mask from GPU to host, then scans all Lx*Ly
  // entries and tallies the ones set to 1. Used to report how many
  // cells violate the entropy condition before and after applying ELBM.
  CUDA_CHECK(cudaMemcpy((void*)(h_violation_mask+0+0*Lx),(const void*)(d_violation_mask+0+0*Lx),(size_t)Lx*Ly*sizeof(int),cudaMemcpyDeviceToHost));
  int count = 0;
  for(int iy=0;iy<Ly;iy++){
    for(int ix=0;ix<Lx;ix++){
      int idx = ix + iy*Lx;
      if(*(h_violation_mask+idx)>0){
	count++;
      }
    }
  }
  return count;
}

// ===========================================================
// conservationRestrictionViolations-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::conservationRestrictionViolations(int t,int offset){
  // Compute the per-site differences between the post-collision state
  // (offset) and the pre-collision state (offset 0), and store them:
  //   d_mass_diff     |rho(offset)   - rho(0)|
  //   d_momX_diff     |jx(offset)    - jx(0) |
  //   d_momY_diff     |jy(offset)    - jy(0) |
  //   d_energy_diff   |rho_e(offset) - rho_e(0)|
  //   d_entropy_diff  h(0) - h(offset) (signed; < 0 means h increased,i.e. Violation)
  km->launchComputeLocalDifferences((double*)lbm->d_f,(double*)lbm->d_rho,(double*)lbm->d_jx,(double*)lbm->d_jy,(double*)lbm->d_rho_e,(double*)lbm->d_h,(double*)lbm->d_feq,(double*)d_mass_diff,(double*)d_momX_diff,(double*)d_momY_diff,(double*)d_energy_diff,(double*)d_entropy_diff,(int)offset);
  
  // Scan the entropy-difference array and flag each site where the
  // entropy restriction is violated. Because d_entropy_diff is stored
  // as h_pre-h_post, a violation (h_post>h_pre) shows up as a negative
  // value, so any site with d_entropy_diff < 0 gets its d_violation_mask
  // entry set to 1. Sites that pass are set to 0.
  km->launchMarkEntropyViolations((double*)d_entropy_diff,(int*)d_violation_mask,(int)offset);
}

// ===========================================================
// applyELBM-PHYSICSCHECKER
// ===========================================================
void PHYSICSCHECKER::applyELBM(int offset){
  // For every site marked in d_violation_mask, solve the entropy
  // condition H(omega_eff) = 0 with the Newton-Bisection hybrid method
  // and store the resulting omega_eff in d_omega_eff. Sites without a
  // violation are left untouched, since the entropic collision kernel
  // only reads d_omega_eff where d_violation_mask == 1.
  km->launchFindOmegaEff((double*)lbm->d_f,(double*)lbm->d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);

  // Redo the collision at the violating sites using the corrected
  // omega_eff instead of the BGK omega, so the entropy restriction
  // is respected there. Non_violating sites are left as they are.
  km->launchEntropicCollision((double*)lbm->d_f,(double*)lbm->d_feq,(double*)d_omega_eff,(int*)d_violation_mask,(int)offset);

  // Recompute the macroscopic fields at the offset, since the entropic
  // collision just changed the distribution function
  km->launchComputeMacros((double*)lbm->d_f,(double*)lbm->d_rho,(double*)lbm->d_jx,(double*)lbm->d_jy,(double*)lbm->d_rho_e,(double*)lbm->d_h,(int)offset);
}
