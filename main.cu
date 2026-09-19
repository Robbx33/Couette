// main.cu
#include "LBM.h"
#include "Visualizer.h"
#include "KernelsManager.h"
#include "PhysicsChecker.h"
#include "Results.h"

#include <iostream>
using namespace std;

int main(int argc,char **argv){
  KernelsManager Gargantua;
  LATTICEBOLTZMANN Gauss((KernelsManager*)&Gargantua);
  PHYSICSCHECKER Noah((LATTICEBOLTZMANN*)&Gauss,(KernelsManager*)&Gargantua);
  RESULTS Jeremias((LATTICEBOLTZMANN*)&Gauss,(PHYSICSCHECKER*)&Noah,(KernelsManager*)&Gargantua);
  //Visualizer Roberto((int)argc,(char**)argv,(PHYSICSCHECKER*)&Noah,(RESULTS*)&Jeremias,(KernelsManager*)&Gargantua);


  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    Gauss.computeMacros(0);
    Gauss.computeFeq();
    
    //if(t%1==0){
      //Noah.collisionConservationRestrictionDifferences((int)t);
      //Roberto.plotConservationRestriction((int)t,0);
      //Roberto.renderAndDisplay((double*)Gauss.get_d_h(),(int)t);
      //Roberto.plotMacros((int)t);
    //}
    
    Gauss.Collision(1);
    Noah.conservationRestrictionViolations((int)t,1);
    //Roberto.plotConservationRestriction((int)t,1);
    Noah.applyELBM(1);
    //Noah.conservationRestrictionViolations((int)t,1);
    //Roberto.plotConservationRestriction((int)t,1);
    
    Gauss.Stream(1);
  }
  
  return 0;
}
