// main.cu
#include "LBM.h"
#include "Constants.h"
#include "PhysicsChecker.h"
#include "Results.h"
#include "KernelsManager.h"
#include <iostream>
using namespace std;

int main(){
  // Create LBM simulation
  LATTICEBOLTZMANN Noah;
  KernelsManager kernels;
  PHYSICSCHECKER physics((LATTICEBOLTZMANN*)&Noah,(KernelsManager*)&kernels);
  RESULTS Results((LATTICEBOLTZMANN*)&Noah);

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    // Compute
    
    kernels.launchCollision(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_w());
    kernels.launchStream(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy());
  
    // Every 30 steps, copy back and visualize
    if(t%30==0 || t==1999){
      kernels.launchComputeMacros(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_rho(),Noah.get_d_jx(),Noah.get_d_jy(),Noah.get_d_rho_e(),Noah.get_d_h());
      kernels.launchComputeFeq(Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_w(),Noah.get_d_rho(),Noah.get_d_jx(),Noah.get_d_jy(),Noah.get_d_feq());

      
      /*Noah.copyBack();
      Noah.calcularMacros();
      Noah.calcularFeq();*/
      
      // Physics diagnostics      
      //physics.plotCollisionConservation_RestrictionErrorsCPU((int)t);
      physics.plotCollisionConservation_RestrictionErrorsGPU((int)t);
      // Flow visualization
      //Results.plotAll(t);

      //cout<<"t="<<t<<" | ";
      /*physics.printSummary();*/
      }
  }



  
  //Final report

  //cout<<"\n=== Final Report ==="<<endl;
  //physics.printReport();
  //cout<<"=== Program Ended ==="<<endl;

  return 0;
}
