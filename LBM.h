// LBM.h
#ifndef LBM_H
#define LBM_H

#include <cstdio>

#define Lx 128
#define Ly 128
#define Q 9
#define dt 1
#define dx 1
#define c_s dx/(sqrt(3.0)*dt)
#define c_s2 c_s*c_s

class PHYSICSCHECKER;

class LATTICEBOLTZMANN {
 private:
  double *h_f,*d_f;
  int *h_Cx,*h_Cy;
  double *h_w;
  double h_Omega,h_OmegaPrima;
  FILE *gp_pipe,*gp_pipe_error;
  double *h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;

  friend class PHYSICSCHECKER;

public:
    LATTICEBOLTZMANN();
    ~LATTICEBOLTZMANN();
    void Choque(int t);
    void Adveccion();
    void copyBack();
    void calcularMacros();
    void dibuje3D(int t);
    void calcularFeq();
};

#endif
