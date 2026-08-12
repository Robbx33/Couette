// main.cu
#include "LBM.h"
#include "PhysicsChecker.h"
#include "Results.h"
#include "Visualizer.h"
#include <iostream>
using namespace std;

int main(int argc,char **argv){
  // Create LBM simulation
  LATTICEBOLTZMANN Noah;
  Visualizer viz((LATTICEBOLTZMANN*)&Noah,(int)argc,(char**)argv);
  KernelsManager kernels((Visualizer*)&viz);
  PHYSICSCHECKER physics((LATTICEBOLTZMANN*)&Noah,(KernelsManager*)&kernels);
  //RESULTS Results((LATTICEBOLTZMANN*)&Noah);

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    // Compute
    kernels.launchCollision((double*)Noah.get_d_f(),(int*)Noah.get_d_Cx(),(int*)Noah.get_d_Cy(),(double*)Noah.get_d_w());
    kernels.launchStream((double*)Noah.get_d_f(),(int*)Noah.get_d_Cx(),(int*)Noah.get_d_Cy());
    
    // Visualize every 5 steps
    if(t%5==0){
      kernels.launchComputeMacros((double*)Noah.get_d_f(),(int*)Noah.get_d_Cx(),(int*)Noah.get_d_Cy(),(double*)Noah.get_d_rho(),(double*)Noah.get_d_jx(),(double*)Noah.get_d_jy(),(double*)Noah.get_d_rho_e(),(double*)Noah.get_d_h());
      kernels.launchRenderAndFill((uchar4*)viz.get_d_color(),(double*)viz.get_d_positions(),(double*)viz.get_d_uv(),(double*)Noah.get_d_rho(),(int)t);
      }
    
    // Diagnostics every 30 steps
    if(t%30==0 || t==1999){
      kernels.launchComputeMacros((double*)Noah.get_d_f(),(int*)Noah.get_d_Cx(),(int*)Noah.get_d_Cy(),(double*)Noah.get_d_rho(),(double*)Noah.get_d_jx(),(double*)Noah.get_d_jy(),(double*)Noah.get_d_rho_e(),(double*)Noah.get_d_h());
      kernels.launchComputeFeq((int*)Noah.get_d_Cx(),(int*)Noah.get_d_Cy(),(double*)Noah.get_d_w(),(double*)Noah.get_d_rho(),(double*)Noah.get_d_jx(),(double*)Noah.get_d_jy(),(double*)Noah.get_d_feq());
      
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
