// PhysicsChecker.cu
#include "PhysicsChecker.h"
#include "LBM.h"
#include "Constants.h"
#include <unistd.h>
#include <iostream>
using namespace std;

PHYSICSCHECKER::PHYSICSCHECKER(LATTICEBOLTZMANN *Noah){
  lbm = Noah;
  
  system("pkill gnuplot 2>/dev/null");
  usleep(300000);
  
  gp_pipe_mass    = popen("gnuplot -persist","w");
  gp_pipe_momX    = popen("gnuplot -persist","w");
  gp_pipe_momY    = popen("gnuplot -persist","w");
  gp_pipe_energy  = popen("gnuplot -persist","w");
  gp_pipe_entropy = popen("gnuplot -persist","w");
  // Configure gp_pipe_mass
  fprintf(gp_pipe_mass,"set xlabel 'ix'\n");
  fprintf(gp_pipe_mass,"set ylabel 'iy'\n");
  fprintf(gp_pipe_mass,"set zlabel 'Error'\n");
  fprintf(gp_pipe_mass,"set grid\n");
  fprintf(gp_pipe_mass,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_mass,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_mass,"set hidden3d\n");
  fprintf(gp_pipe_mass,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_mass);
  // Configure gp_pipe_momX
  fprintf(gp_pipe_momX,"set xlabel 'ix'\n");
  fprintf(gp_pipe_momX,"set ylabel 'iy'\n");
  fprintf(gp_pipe_momX,"set zlabel 'Error'\n");
  fprintf(gp_pipe_momX,"set grid\n");
  fprintf(gp_pipe_momX,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_momX,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_momX,"set hidden3d\n");
  fprintf(gp_pipe_momX,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_momX);
  // Configure gp_pipe_momY
  fprintf(gp_pipe_momY,"set xlabel 'ix'\n");
  fprintf(gp_pipe_momY,"set ylabel 'iy'\n");
  fprintf(gp_pipe_momY,"set zlabel 'Error'\n");
  fprintf(gp_pipe_momY,"set grid\n");
  fprintf(gp_pipe_momY,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_momY,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_momY,"set hidden3d\n");
  fprintf(gp_pipe_momY,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_momY);
  // Configure gp_pipe_energy (internal energy error)
  fprintf(gp_pipe_energy,"set xlabel 'ix'\n");
  fprintf(gp_pipe_energy,"set ylabel 'iy'\n");
  fprintf(gp_pipe_energy,"set zlabel 'Error'\n");
  fprintf(gp_pipe_energy,"set grid\n");
  fprintf(gp_pipe_energy,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_energy,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_energy,"set hidden3d\n");
  fprintf(gp_pipe_energy,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_energy);
  // Configure gp_pipe_entropy (internal energy error)
  fprintf(gp_pipe_entropy,"set xlabel 'ix'\n");
  fprintf(gp_pipe_entropy,"set ylabel 'iy'\n");
  fprintf(gp_pipe_entropy,"set zlabel 'H(f)-H(feq)'\n");
  fprintf(gp_pipe_entropy,"set grid\n");
  fprintf(gp_pipe_entropy,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe_entropy,"set yrange [0:%d]\n",Ly);
  fprintf(gp_pipe_entropy,"set hidden3d\n");
  fprintf(gp_pipe_entropy,"set palette defined (0'#0000FF',0.5'#00FF00',1'#FF0000')\n");
  fflush(gp_pipe_entropy);
}

PHYSICSCHECKER::~PHYSICSCHECKER(){
  pclose(gp_pipe_mass);
  pclose(gp_pipe_momX);
  pclose(gp_pipe_momY);
  pclose(gp_pipe_energy);
  pclose(gp_pipe_entropy);
}

void PHYSICSCHECKER::plotCollisionConservation_RestrictionErrors(int t){
  double max_mass    = 0.0;
  double max_momX    = 0.0;
  double max_momY    = 0.0;
  double max_energy  = 0.0;
  double min_entropy = 0.0;
  double max_entropy = 0.0;
  
  // Write mass, momentum, energy errors (sampled every 2 points)
  FILE *tmp = fopen("mass_error_3D.dat","w");
  FILE *tmpX = fopen("momX_error_3D.dat","w");
  FILE *tmpY = fopen("momY_error_3D.dat","w");
  FILE *tmpE = fopen("energy_error_3D.dat","w");
  FILE *tmpH = fopen("entropy_diff_3D.dat","w");
  
  for(int iy=0;iy<Ly;iy+=2){
    for(int ix=0;ix<Lx;ix+=2){
      int idx = ix + iy*Lx;
      
      double rho_f     = *(lbm->h_rho+idx);
      double rho_feq   = 0.0;
      double jx_f      = *(lbm->h_jx+idx);
      double jx_feq    = 0.0;
      double jy_f      = *(lbm->h_jy+idx);
      double jy_feq    = 0.0;
      double rhoE_f    = *(lbm->h_rho_e+idx);
      double rhoE_feq  = 0.0;
      
      //Entropy: sum f*log(f/feq) >= sum feq*log(f/feq)
      double entropy_f   = 0.0;
      double entropy_feq = 0.0;
      
      for(int iz=0;iz<Q;iz++){
	rho_feq  += *(lbm->h_feq+idx+iz*Lx*Ly);
	jx_feq   += *(lbm->h_Cx+iz)*(*(lbm->h_feq+idx+iz*Lx*Ly));
	jy_feq   += *(lbm->h_Cy+iz)*(*(lbm->h_feq+idx+iz*Lx*Ly));
	rhoE_feq += 0.5 * (*(lbm->h_Cx+iz)*(*(lbm->h_Cx+iz)) + *(lbm->h_Cy+iz)*(*(lbm->h_Cy+iz))) * (*(lbm->h_feq+idx+iz*Lx*Ly));
	
	// Entropy terms
	if(*(lbm->h_f+idx+iz*Lx*Ly) > 1e-15 && *(lbm->h_feq+idx+iz*Lx*Ly) > 1e-15){
	  entropy_f   += *(lbm->h_f+idx+iz*Lx*Ly)*(log(*(lbm->h_f+idx+iz*Lx*Ly)/(*(lbm->h_feq+idx+iz*Lx*Ly))));
	  entropy_feq += *(lbm->h_feq+idx+iz*Lx*Ly)*(log(*(lbm->h_f+idx+iz*Lx*Ly)/(*(lbm->h_feq+idx+iz*Lx*Ly)))); 
	}
      }
      double energy_feq = rhoE_feq-0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq));
      
      double mass_err     = fabs(rho_f-rho_feq);
      double momX_err     = fabs(jx_f-jx_feq);
      double momY_err     = fabs(jy_f-jy_feq);
      double energy_err   = fabs(rhoE_f-energy_feq);
      double entropy_diff = entropy_f-entropy_feq;// Should be >=0
      if(mass_err > max_mass){
	max_mass = mass_err;
      }
      if(momX_err > max_momX){
	max_momX = momX_err;
      }
      if(momY_err > max_momY){
	max_momY = momY_err;
      }
      if(energy_err > max_energy){
	max_energy = energy_err;
      }
      if(entropy_diff > max_entropy){
	max_entropy = entropy_diff;
      }
      if(entropy_diff < min_entropy){
	min_entropy = entropy_diff;
      }
      fprintf(tmp,"%d %d %e\n",ix,iy,mass_err);
      fprintf(tmpX,"%d %d %e\n",ix,iy,momX_err);
      fprintf(tmpY,"%d %d %e\n",ix,iy,momY_err);
      fprintf(tmpE,"%d %d %e\n",ix,iy,energy_err);
      fprintf(tmpH,"%d %d %e\n",ix,iy,entropy_diff);
    }
    fprintf(tmp,"\n");
    fprintf(tmpX,"\n");
    fprintf(tmpY,"\n");
    fprintf(tmpE,"\n");
    fprintf(tmpH,"\n");
  }
  fclose(tmp);
  fclose(tmpX);
  fclose(tmpY);
  fclose(tmpE);
  fclose(tmpH);
 
  // Plot mass error in its own window
  fprintf(gp_pipe_mass,"set title 'Collision Mass Conservation Error - t=%d (max=%e)'\n",t,max_mass);
  fprintf(gp_pipe_mass,"splot 'mass_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_mass); 
  
  // Plot momentum X error in its own window
  fprintf(gp_pipe_momX,"set title '|Collision Momentum X Error| - t=%d (max=%e)'\n",t,max_momX);
  fprintf(gp_pipe_momX,"splot 'momX_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momX);
  
  // Plot momentum Y error in its own window
  fprintf(gp_pipe_momY,"set title '|Collision Momentum Y Error| - t=%d (max=%e)'\n",t,max_momY);
  fprintf(gp_pipe_momY,"splot 'momY_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_momY);
  
  // Internal energy error
  fprintf(gp_pipe_energy,"set title 'Collision Internal Energy Error - t=%d (max=%e)'\n", t, max_energy);
  fprintf(gp_pipe_energy,"splot 'energy_error_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_energy);
  
  // Entropy Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)]
  fprintf(gp_pipe_entropy,"set title 'Collision Entropy Restriction: sum[f*log(f/feq)] >= sum[feq*Log(f/feq)] - t=%d (min=%e, max=%e)'\n", t, min_entropy, max_entropy);
  fprintf(gp_pipe_entropy,"splot 'entropy_diff_3D.dat' with points pt 5 ps 0.5 palette\n");
  fflush(gp_pipe_entropy);
  
  usleep(2000000);
}

/*void PHYSICSCHECKER::printSummary(){
  // Print one-line summary of current diagnostics
  // You can implement this to show max values
  cout<<"Diagnostics OK"<<endl;
}

void PHYSICSCHECKER::printReport(){
  cout<<"Collision Mass Conservation: OK"<<endl;
  cout<<"Collision Momentum X Conservation: OK"<<endl;
  cout<<"Collision Momentum Y Conservation: OK"<<endl;
  cout<<"Collision Energy Conservation: OK"<<endl;
  cout<<"Collision Entropy Restriction: OK"<<endl;
  }*/
