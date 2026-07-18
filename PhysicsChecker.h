// PhysicsChecker.h
#ifndef PHYSICSCHECKER_H
#define PHYSICSCHECKER_H

#include <cstdio>

class LATTICEBOLTZMANN;

class PHYSICSCHECKER{
private:
  LATTICEBOLTZMANN *lbm;
  FILE *gp_pipe_mass,*gp_pipe_momX,*gp_pipe_momY,*gp_pipe_energy,*gp_pipe_entropy;
  
public:
  PHYSICSCHECKER(LATTICEBOLTZMANN *Noah);
  ~PHYSICSCHECKER();
  void plotCollisionConservation_RestrictionErrors(int t);
};

#endif
