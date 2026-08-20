// main.cu
#include "LBM.h"
#include "Visualizer.h"
#include "KernelsManager.h"
#include "PhysicsChecker.h"
#include "Results.h"

#include <iostream>
using namespace std;

int main(int argc,char **argv){
  LATTICEBOLTZMANN Gauss;
  Visualizer Roberto((int)argc,(char**)argv);
  KernelsManager Gargantua((Visualizer*)&Roberto);
  PHYSICSCHECKER Noah;
  RESULTS Jeremias((LATTICEBOLTZMANN*)&Gauss);

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    // Compute
    Gargantua.launchComputeMacros((double*)Gauss.get_d_f(),(double*)Gauss.get_d_rho(),(double*)Gauss.get_d_jx(),(double*)Gauss.get_d_jy(),(double*)Gauss.get_d_rho_e(),(double*)Gauss.get_d_h());
    Gargantua.launchComputeFeq((double*)Gauss.get_d_rho(),(double*)Gauss.get_d_jx(),(double*)Gauss.get_d_jy(),(double*)Gauss.get_d_feq());
    
    if(t%1==0){
      Gargantua.launchComputeErrors((double*)Gauss.get_d_f(),(double*)Gauss.get_d_rho(),(double*)Gauss.get_d_jx(),(double*)Gauss.get_d_jy(),(double*)Gauss.get_d_rho_e(),(double*)Gauss.get_d_h(),(double*)Gauss.get_d_feq(),(double*)Noah.get_d_mass_err(),(double*)Noah.get_d_momX_err(),(double*)Noah.get_d_momY_err(),(double*)Noah.get_d_energy_err(),(double*)Noah.get_d_hfhfeq_diff());
      Gargantua.launchRenderAndFill((uchar4*)Roberto.get_d_color(),(double*)Roberto.get_d_positions(),(double*)Roberto.get_d_uv(),(double*)Noah.get_d_hfhfeq_diff(),(int)t);
      Jeremias.plotMacros((int)t);
    }
    
    Gargantua.launchCollision((double*)Gauss.get_d_f(),(double*)Gauss.get_d_feq());
    Gargantua.launchStream((double*)Gauss.get_d_f());
  }
  return 0;
}
