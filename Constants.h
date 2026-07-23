// Constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>

// Compile-time constants
static constexpr int Lx = 128;
static constexpr int Ly = 128;
static constexpr int Q = 9;
static constexpr double dt = 1.0;
static constexpr double dx = 1.0;

// Macros for device and host
#define dt 1.0
#define dx 1.0
#define c_s (dx/(sqrt(3.0)*dt))
#define c_s2 c_s*c_s

// Physical constants
static const double RHO0=1.0,UX0=0.0,UY0=0.0;
static const double nu  = 0.15;
static const double Tau = nu/c_s2 + 0.5*dt;

#endif
