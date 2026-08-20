// PhysicsChecker.h
#ifndef PHYSICSCHECKER_H
#define PHYSICSCHECKER_H

#include "Constants.h"
#include "error.h"
#include <unistd.h>

class PHYSICSCHECKER{
private:  
  double *h_mass_err,*h_momX_err,*h_momY_err,*h_energy_err,*h_hfhfeq_diff;
  double *d_mass_err,*d_momX_err,*d_momY_err,*d_energy_err,*d_hfhfeq_diff;
  
  FILE *gp_pipe_mass_error;
  FILE *gp_pipe_momX_error;
  FILE *gp_pipe_momY_error;
  FILE *gp_pipe_energy_error;
  FILE *gp_pipe_hfhfeq_diff;  
public:
  PHYSICSCHECKER(void);
  ~PHYSICSCHECKER(void);
  
  double *get_d_mass_err(void){return d_mass_err;}
  double *get_d_momX_err(void){return d_momX_err;}
  double *get_d_momY_err(void){return d_momY_err;}
  double *get_d_energy_err(void){return d_energy_err;}
  double *get_d_hfhfeq_diff(void){return d_hfhfeq_diff;}
};

#endif
