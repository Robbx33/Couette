// LBM.h
#ifndef LBM_H
#define LBM_H

#include "Constants.h"
#include "error.h"

class LATTICEBOLTZMANN{
private:
  // Host data
  double *h_f,*h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;
  int *h_Cx,*h_Cy;
  double *h_w;
  
  // Device data
  double *d_f,*d_rho,*d_jx,*d_jy,*d_rho_e,*d_h,*d_feq;

  friend class PHYSICSCHECKER;
  friend class RESULTS;
public:
  LATTICEBOLTZMANN(void);
  ~LATTICEBOLTZMANN(void);
  
  // Getters for main
  double *get_d_f(){return d_f;}
  double *get_d_rho(){return d_rho;}
  double *get_d_jx(){return d_jx;}
  double *get_d_jy(){return d_jy;}
  double *get_d_rho_e(){return d_rho_e;}
  double *get_d_h(){return d_h;}
  double *get_d_feq(){return d_feq;}
};

#endif
