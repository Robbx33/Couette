// Constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>

// ALL MACROS for device/host
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

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 2
#define THREADS_PER_BLOCK THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y  

// Physical constants
#define RHO0 1.0
#define UX0 0.0
#define UY0 0.0

// Gaussian perturbation constants
#define amplitude 0.001
#define sigma 5.0

#endif
