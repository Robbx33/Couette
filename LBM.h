// LBM.h
#ifndef LBM_H
#define LBM_H

class LATTICEBOLTZMANN{
private:
  // Host data
  double *h_f,*h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;
  int *h_Cx,*h_Cy;
  double *h_w;
    /*FILE *gp_pipe,*gp_pipe_error;*/
  
  // Device data
  double *d_f,*d_rho,*d_jx,*d_jy,*d_rho_e,*d_h,*d_feq;
  int *d_Cx,*d_Cy;
  double *d_w;

  friend class PHYSICSCHECKER;
  friend class RESULTS;
      /*friend class KernelsManager;*/
  
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
  double *get_d_rho(){ return d_rho;}
  double *get_d_jx(){ return d_jx;}
  double *get_d_jy(){ return d_jy;}
  double *get_d_rho_e(){ return d_rho_e;}
  double *get_d_h(){ return d_h;}
  double *get_d_feq(){ return d_feq;}
  
  
  double *get_rho(){ return h_rho;}
  double *get_jx(){ return h_jx;}
  double *get_jy(){ return h_jy;}
  double *get_rho_e(){ return h_rho_e;}
  double *get_h_f(){ return h_f;};
  double *get_h_rho(){ return h_rho;}
};

#endif
