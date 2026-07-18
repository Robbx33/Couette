// main.cu
#include "LBM.h"
#include "PhysicsChecker.h"
#include <iostream>
using namespace std;

int main() {
  LATTICEBOLTZMANN Noah;
  PHYSICSCHECKER Gauss((LATTICEBOLTZMANN*)&Noah);
  
  for(int t=0;t<2000;t++){
    Noah.Choque((int)t);
    Noah.Adveccion();
    
    if(t%30==0 || t==1999){
      Noah.copyBack();
      Noah.calcularMacros();
      
      //Noah.dibuje3D((int)t);

      Noah.calcularFeq();
      Gauss.plotCollisionConservation_RestrictionErrors((int)t);
    }
    
  }
  //Noah.calcularFeq(); 
  
  cout<<"Program ended."<<endl;
  return 0;
}
