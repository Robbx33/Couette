// main.cu
#include "LBM.h"
#include "Visualizer.h"
#include "KernelsManager.h"
#include "PhysicsChecker.h"
#include "Results.h"

#include <iostream>
using namespace std;

int main(int argc,char **argv){

  Visualizer Roberto((int)argc,(char**)argv);
  KernelsManager Gargantua((Visualizer*)&Roberto);
  LATTICEBOLTZMANN Gauss((KernelsManager*)&Gargantua);
  PHYSICSCHECKER Noah((LATTICEBOLTZMANN*)&Gauss,(KernelsManager*)&Gargantua);
  RESULTS Jeremias((LATTICEBOLTZMANN*)&Gauss);

  cout<<"=== LBM Simulation Started ==="<<endl;
  cout<<"Grid: "<<Lx<<"x"<<Ly<<", Q="<<Q<<endl;
  cout<<"nu="<<nu<<", Tau="<<Tau<<endl;
  cout<<"=============================="<<endl;
  
  for(int t=0;t<2000;t++){
    Gauss.computeMacros(0);
    Gauss.computeFeq();

    if(t%1==0){
      Gargantua.launchComputeCollisionErrors((double*)Gauss.get_d_f(),(double*)Gauss.get_d_rho(),(double*)Gauss.get_d_jx(),(double*)Gauss.get_d_jy(),(double*)Gauss.get_d_rho_e(),(double*)Gauss.get_d_h(),(double*)Gauss.get_d_feq(),(double*)Noah.get_d_mass_diff(),(double*)Noah.get_d_momX_diff(),(double*)Noah.get_d_momY_diff(),(double*)Noah.get_d_energy_diff(),(double*)Noah.get_d_entropy_diff());
      //Noah.doubleCheckPhysics((int)t);
      Gargantua.launchRenderAndFill((uchar4*)Roberto.get_d_color(),(double*)Roberto.get_d_positions(),(double*)Roberto.get_d_uv(),(double*)Noah.get_d_entropy_diff(),(int)t);
      //Jeremias.plotMacros((int)t);
    }
    
    Gauss.Collision();
    Noah.entropyLocalConservation(t,1);
    Gauss.Stream();
  }

  return 0;
}
