// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  CONSTANTS_H is not defined -> enter, read the class,
//                       and #define CONSTANTS_H to mark it as "already read".
//   - Any later inclusion: CONSTANTS_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef CONSTANTS_H
#define CONSTANTS_H

// Lattice and physical constants.
//
//   Lx, Ly     grid size in the x and y directions
//   Q          number of discrete velocities (D2Q9)
//   dt, dx     time step and grid spacing (both 1.0->lattice units)
//   c_s        speed of sound, dx/(sqrt(3)*dt) from isothermal ideal gas
//   c_s2       c_s squared
//   nu         kinematic viscosity nu=mu/rho
//   Tau        relaxation time, nu/c_s2 + 0.5*dt
//   Omega      collision frequency, dt/Tau
//   OmegaPrima 1-Omega, the leftover factor in the BGK update
#define Lx 128
#define Ly 128
#define Q 9
#define dt 1.0
#define dx 1.0
#define c_s (dx/(sqrt(3.0)*dt))
#define c_s2 (c_s*c_s)
#define nu 0.1
#define Tau (nu/c_s2 + 0.5*dt)
#define Omega (dt/Tau)
#define OmegaPrima (1.0 - Omega)

// CUDA kernel launch configuration.
//
//   THREADS_PER_BLOCK_X  threads per block along x
//   THREADS_PER_BLOCK_Y  threads per block along y
//   THREADS_PER_BLOCK    total threads per 2D block (X*Y)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 2
#define THREADS_PER_BLOCK THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y  

// Initial Condition (base) of the fluid.
//
//   RHO0  base density
//   UX0   initial velocity in x
//   UY0   initial velocity in y
#define RHO0 1.0
#define UX0 0.0
#define UY0 0.0

// Initial Condition (Gaussian perturbation) parameters for the fluid
//
//   amplitude  peak height of the perturbation (mass)
//   Sigma      standard deviation (width of the Gaussian)
#define amplitude 0.001
#define sigma 5.0

#endif
