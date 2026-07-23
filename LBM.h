// LBM.h
#ifndef LBM_H
#define LBM_H

/*#include <cmath>

static constexpr int Lx = 128;
static constexpr int Ly = 128;
static constexpr int Q = 9;
static constexpr double dt = 1.0;
static constexpr double dx = 1.0;

static const double c_s = dx/(sqrt(3.0)*dt);
static const double c_s2 = c_s*c_s;

static const double RHO0=1.0,UX0=0.0,UY0=0.0;
static const double nu  = 0.15;
static const double Tau = nu/c_s2 + 0.5*dt;*/
// Extern constants for main.cu
//extern const double nu;
//extern const double Tau;

// Forward declarations
/*class PHYSICSCHECKER;
class RESULTS;
class KernelsManager;*/

class LATTICEBOLTZMANN{
private:
  // Host data
  double *h_f,*h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;
  int *h_Cx,*h_Cy;
  double *h_w;
  double h_Omega,h_OmegaPrima;
    /*FILE *gp_pipe,*gp_pipe_error;*/
  
  // Device data
  double *d_f,*d_rho,*d_jx,*d_jy,*d_rho_e,*d_h,*d_feq;
  int *d_Cx,*d_Cy;
  double *d_w;
  double d_Omega,d_OmegaPrima;
  //double *d_mass_err,*d_momX_err,*d_momY_err,*d_energy_err,*d_entropy_diff;*/

  friend class PHYSICSCHECKER;
    /*friend class RESULTS;
  friend class KernelsManager;*/
  
public:
  LATTICEBOLTZMANN();
  ~LATTICEBOLTZMANN();
  void copyBack();
  void calcularMacros();
  void calcularFeq();
  
  // Getters for friends
  double *get_d_f(){ return d_f;}
  int *get_d_Cx(){ return d_Cx;}
  int *get_d_Cy(){ return d_Cy;}
  double *get_d_w(){ return d_w;}
  double get_d_Omega(){ return d_Omega;}
  double get_d_OmegaPrima(){ return d_OmegaPrima;}
    /*double *get_rho(){ return h_rho;}
  double *get_jx(){ return h_jx;}
  double *get_jy(){ return h_jy;}
  double *get_rho_e(){ return h_rho_e;}*/
  double *get_h_f(){ return h_f;};
  double *get_h_rho(){ return h_rho;}
};

#endif
