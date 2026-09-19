// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  RESULTS_H is not defined -> enter, read the class,
//                       and #define RESULTS_H to mark it as "already read".
//   - Any later inclusion: RESULTS_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef RESULTS_H
#define RESULTS_H

// LBM.h            -> LATTICEBOLTZMANN class (method implementations)
// unistd.h         -> usleep (pauses after killing old gnuplot processes)
#include "LBM.h"
#include "PhysicsChecker.h"

class RESULTS{
 private:
  // Scratch buffers for the min/max reduction in findMinMax
  //
  //   d_min_temp, d_max_temp  per-block outcome on the GPU, written by
  //                           findMinMaxKernel (one entry per block)
  //   h_min_temp, h_max_temp  host copies of the above, for the final
  //                           reduction loop on the CPU
  //   grid_size_reduce        number of blocks in the 2D launch
  //                           (= gridSize2D.x*gridSize2D.y), used as
  //                           the length of all four arrays above
  double *h_min_temp,*h_max_temp;
  double *d_min_temp,*d_max_temp;
  int grid_size_reduce;

  // data_file  plain text file holding the numeric values (min/max of
  //            each conserved quantity per time step); gnuplot reads it
  FILE *data_file;

  // Pointers to the other two objects this class works with.
  //
  // lbm points to the Gauss object that points to the start
  //     of the LATTICEBOLTZMANN start of the class, pck points
  // to the begining of the PHYSICSCHECKER class and km points
  // to the begining of the KernelsManager class
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;
  PHYSICSCHECKER *pck;
 public:
  // Lifecycle of the Results object: setup and cleanup.
  // The constructor takes a pointer to LATTICEBOLTZMANN, other to
  // PHYSICSCHECKER and other to KernelsManager
  // destructor frees host and device memory
  RESULTS(LATTICEBOLTZMANN *Gauss,PHYSICSCHECKER *Noah,KernelsManager *Gargantua);
  ~RESULTS();

  // findMinMax(d_data,min_val,max_val,offset)
  //   Reduces d_data over teh whole grid and returns the minimum
  //   and maximum values. Used by writeConservationsRestriction
  //   below and by Visualizer for the color/height scaling of the
  //   surface. Launches findMinMaxKernel via KernelsManager, copies
  //   the per-block results back, and finishes the reduction on CPU.
  void findMinMax(double *d_data,double *min_val,double *max_val,int offset);

  // writeConservationRestriction(t,offset)
  //   For each of the five d_*_diff arrays (mass, x-momentum, y-momentum,
  //   energy, entropy), computes min and max with findMinMax and appends
  //   a row to  conservation&restriction_quantities.dat:
  //      t mass_min mass_max momX_min momX_max momY_min momY_max energy_min
  //      energy_max entropy_min entropy_max
  void writeConservationRestriction(int t, int offset);
  
  // writeMacroData(filename,data)
  //    Writes one horizontal line of the domain (at iy=Ly/2) to
  //    a two-column .dat file: column 1 is ix, column 2 is *(data+ix+(Ly/2)*Lx].
  //
  // writeMacros(t)
  //    Copies rho, jx,jy,rho_e, and h from the GPU to the host
  //    arrays in LATTICEBOLTZMANN, then writes each of them to its
  //    own .dat file. Visualizer reads these files and drives gnuplot.
  void writeMacroData(const char *filename,double *data);
  void writeMacros(int t);
  //double analytical_rho(int ix,int iy,int t);
};

#endif
