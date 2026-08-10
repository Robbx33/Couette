// main.cu
#include "LBM.h"
#include "PhysicsChecker.h"
#include "Results.h"
#include "Visualizer.h"
#include <iostream>
using namespace std;

//OpenGL is an Application Programming Interface that can handle things in the CPU and the GPU
//argc, argv are empty arguments that (OpenGL-GLUT) is expecting
//GLUT is the toolkit of OpenGL
int main(int argc,char **argv){
  // Create LBM simulation
  LATTICEBOLTZMANN Noah;
  KernelsManager kernels;
  PHYSICSCHECKER physics((LATTICEBOLTZMANN*)&Noah,(KernelsManager*)&kernels);
  RESULTS Results((LATTICEBOLTZMANN*)&Noah);
  //Visualizer viz((LATTICEBOLTZMANN*)&Noah,(KernelsManager*)&kernels,(int)argc,(char**)argv);

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    // Compute
    kernels.launchCollision(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_w());
    kernels.launchStream(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy());
    
    // Visualize every 5 steps
    /*if(t%5==0){
      kernels.launchComputeMacros(Noah.get_d_f(),Noah.get_d_Cx(),Noah.get_d_Cy(),Noah.get_d_rho(),Noah.get_d_jx(),Noah.get_d_jy(),Noah.get_d_rho_e(),Noah.get_d_h());
      viz.display((int)t);
      }*/
    
    // Diagnostics every 30 steps
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
