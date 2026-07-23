// main.cu
#include "LBM.h"
#include "Constants.h"
#include "PhysicsChecker.h"
//#include "Results.h"
#include "KernelsManager.h"
//#include "error.h"
#include <iostream>
using namespace std;

int main(){
  // Create LBM simulation
  LATTICEBOLTZMANN Noah;
  KernelsManager kernels((int)Lx,(int)Ly);
  PHYSICSCHECKER physics((LATTICEBOLTZMANN*)&Noah);
  
  //Create managers
  //PHYSICSCHECKER physics((LATTICEBOLTZMANN*)&Noah);
  //RESULTS results((LATTICEBOLTZMANN*)&Noah);
  

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    // Compute
    kernels.launchCollision(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_w(),Noah.get_d_Omega(),Noah.get_d_OmegaPrima());
    kernels.launchStream(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy());

    // Every 30 steps, copy back and visualize
    if(t%30==0 || t==1999){
      Noah.copyBack();
      Noah.calcularMacros();
      Noah.calcularFeq();
	
      // Physics diagnostics      
      physics.plotCollisionConservation_RestrictionErrors((int)t);
	
      // Flow visualization
      //results.plotAll(t);

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
