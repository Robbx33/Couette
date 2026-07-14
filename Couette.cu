// ===========================================================
// HEADER FILES
// ===========================================================
#include <iostream> // g++ compiles (standard C++)
#include "book.h"   // nvcc compiles (CUDA helper macros)
#include <unistd.h>
using namespace std;// g++ compiles (standard C++)
// ===========================================================
// CONSTANTS (grid size) - Preprocessor (runs before both compilers)
// ===========================================================
#define Lx 128 
#define Ly 128   
#define Q 9
#define dt 1
#define dx 1
#define c_s dx/(sqrt(3.0)*dt)
#define c_s2 c_s*c_s
// ===========================================================
// CHAPMAN-ENSKOG union between LBM and BE -> Macroscopic equations
// ===========================================================
const double nu = 0.15;
const double Tau = nu/(c_s*c_s) + 0.5*dt;
// ===========================================================
// LATTICE D2Q9 constants
// ===========================================================
__constant__ int d_Cx[Q];
__constant__ int d_Cy[Q];
__constant__ double d_w[Q];
__constant__ double d_Omega;
__constant__ double d_OmegaPrima;
// ===========================================================
// BOUNDARY & INITIAL CONDITIONS
// ===========================================================
const double RHO0=1.0,UX0=0.0,UY0=0.0;
const double Rho0=1.0,Ux0=0.0,Uy0=0.0;
// ===========================================================
// COLLISION KERNEL
// ===========================================================
__global__ void collisionKernel(double* d_f){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  if(i<Lx && j<Ly && k<Q){
    int id = i + j*Lx;
    double rho=0.0,jx=0.0,jy=0.0;
    for(int kaux=0;kaux<Q;kaux++){
      double f = *(d_f+id+kaux*Lx*Ly);
      rho += f;
      jx  += *(d_Cx+kaux)*f;
      jy  += *(d_Cy+kaux)*f;
    }
    double feq;
    feq = d_w[k]*rho*(1+(d_Cx[k]*jx/rho+d_Cy[k]*jy/rho)/c_s2+0.5*((d_Cx[k]*jx/rho)*(d_Cx[k]*jx/rho)+2*(d_Cx[k]*jx/rho)*(d_Cy[k]*jy/rho)+(d_Cy[k]*jy/rho)*(d_Cy[k]*jy/rho))/(c_s2*c_s2)-0.5*(jx*jx/(rho*rho)+jy*jy/(rho*rho))/c_s2);
    // BGK collision store in second half
    *(d_f+id+(k+Q)*Lx*Ly) = *(d_f+id+k*Lx*Ly)*d_OmegaPrima + d_Omega*feq;
  }
}
// ===========================================================
// STREAMING KERNEL
// ===========================================================
__global__ void streamKernel(double* d_f){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z;
  if(i<Lx && j<Ly && k<Q){
    *(d_f+(i+*(d_Cx+k)+Lx)%Lx+((j+*(d_Cy+k)+Ly)%Ly)*Lx+k*Lx*Ly)=*(d_f+i+j*Lx+(k+Q)*Lx*Ly);
  }
}
class LATTICEBOLTZMANN{
private:
  double *h_f;
  double *d_f;
  int *h_Cx;
  int *h_Cy;
  double *h_w;
  double h_Omega;
  double h_OmegaPrima;
  FILE *gp_pipe;
  
  // Macroscopic quantities
  double *h_rho,*h_jx,*h_jy,*h_rho_e,*h_h,*h_feq;
public:
  LATTICEBOLTZMANN(){
    // 1. Allocate memory
    h_f = (double*)malloc(Lx*Ly*Q*sizeof(double));
    HANDLE_ERROR(cudaMalloc((void**)&d_f,Lx*Ly*Q*2*sizeof(double)));

    h_Cx = (int*)malloc(Q*sizeof(int));
    h_Cy = (int*)malloc(Q*sizeof(int));
    h_w  = (double*)malloc(Q*sizeof(double));

    h_rho   = (double*)malloc(Lx*Ly*sizeof(double));
    h_jx    = (double*)malloc(Lx*Ly*sizeof(double));
    h_jy    = (double*)malloc(Lx*Ly*sizeof(double));
    h_rho_e = (double*)malloc(Lx*Ly*sizeof(double));
    h_h     = (double*)malloc(Lx*Ly*sizeof(double));
    h_feq   = (double*)malloc(Lx*Ly*Q*sizeof(double));
    
    // 2. Set constants
    *h_Cx     = 0;  *h_Cy     = 0;
    *(h_Cx+1) = 1;  *(h_Cy+1) = 0;
    *(h_Cx+2) = 0;  *(h_Cy+2) = 1;
    *(h_Cx+3) = -1; *(h_Cy+3) = 0;
    *(h_Cx+4) = 0;  *(h_Cy+4) = -1;
    *(h_Cx+5) = 1;  *(h_Cy+5) = 1;
    *(h_Cx+6) = -1; *(h_Cy+6) = 1;
    *(h_Cx+7) = -1; *(h_Cy+7) = -1;
    *(h_Cx+8) = 1;  *(h_Cy+8) = -1;

    *h_w = 4.0/9;
    *(h_w+1) = *(h_w+2) = *(h_w+3) = *(h_w+4) =1.0/9;
    *(h_w+5) = *(h_w+6) = *(h_w+7) = *(h_w+8) =1.0/36;

    h_Omega = dt/Tau;
    h_OmegaPrima = 1.0 - h_Omega;

    // 3. Initialize h_f with equilibrium (rho=1.0, ux=0, uy=0)
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	for(int iz=0;iz<Q;iz++){
	  if(iy==Ly-1){
	    *(h_f+ix+iy*Lx+iz*Lx*Ly) = h_w[iz]*RHO0*(1+(h_Cx[iz]*UX0+h_Cy[iz]*UY0)/c_s2+0.5*((h_Cx[iz]*UX0)*(h_Cx[iz]*UX0)+2*(h_Cx[iz]*UX0)*(h_Cy[iz]*UY0)+(h_Cy[iz]*UY0)*(h_Cy[iz]*UY0))/(c_s2*c_s2)-0.5*(UX0*UX0+UY0*UY0)/c_s2);
	  }
	  else{
	    *(h_f+ix+iy*Lx+iz*Lx*Ly) = h_w[iz]*Rho0*(1+(h_Cx[iz]*Ux0+h_Cy[iz]*Uy0)/c_s2+0.5*((h_Cx[iz]*Ux0)*(h_Cx[iz]*Ux0)+2*(h_Cx[iz]*Ux0)*(h_Cy[iz]*Uy0)+(h_Cy[iz]*Uy0)*(h_Cy[iz]*Uy0))/(c_s2*c_s2)-0.5*(Ux0*Ux0+Uy0*Uy0)/c_s2);
	  }
	}
      }
    }
    // -------- Test: Add Gaussian pertubation for periodic test -----------------
    double amplitude = 0.01;
    double sigma = 10.0;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double r2 = (ix-0.5*Lx)*(ix-0.5*Lx)+(iy-0.5*Ly)*(iy-0.5*Ly);
	double perturbation = amplitude*exp(-r2/(2.0*sigma*sigma));
	for(int iz=0;iz<Q;iz++){
	  *(h_f+idx+iz*Lx*Ly) += *(h_w+iz)*perturbation;
	}
      }
    }
    
    // 4. Copy h_f to GPU
    HANDLE_ERROR(cudaMemcpy((void*)(d_f+0+0*Lx+0*Lx*Ly),(const void*)(h_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyHostToDevice));
    
    // 5. Copy constants to GPU
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_Cx+0),(const void*)(h_Cx+0),(size_t)Q*sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_Cy+0),(const void*)(h_Cy+0),(size_t)Q*sizeof(int)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)(d_w+0),(const void*)(h_w+0),(size_t)Q*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)&d_Omega,(const void*)&h_Omega,(size_t)sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol((const void*)&d_OmegaPrima,(const void*)&h_OmegaPrima,(size_t)sizeof(double)));

    // 6. Initialize gnuplot for density
    //-- Kill any existing gnuplot processes--
    system("pkill gnuplot 2>/dev/null");
    usleep(300000);
  
    gp_pipe = popen("gnuplot -persist", "w");
    fprintf(gp_pipe,"set xlabel 'ix'\n");
    fprintf(gp_pipe,"set ylabel 'iy'\n");
    fprintf(gp_pipe,"set zlabel 'Density'\n");
    fprintf(gp_pipe,"set grid\n");
    fprintf(gp_pipe,"set xrange [0:%d]\n", Lx);
    fprintf(gp_pipe,"set yrange [0:%d]\n", Ly);
    fprintf(gp_pipe,"set zrange [0.98:1.015]\n");
    fprintf(gp_pipe,"set cbrange [0.98:1.015]\n");
    fprintf(gp_pipe,"set palette defined (0'#0000FF',0.33'#00FFFF',0.66'#FFFF00',1'#FF0000')\n");
    fflush(gp_pipe);

    cout<<"Memory allocated(CPU+GPU). "<<endl;
  }
  ~LATTICEBOLTZMANN(){
    free(h_f);
    free(h_Cx);
    free(h_Cy);
    free(h_w);
    free(h_rho);
    free(h_jx);
    free(h_jy);
    free(h_rho_e);
    free(h_h);
    free(h_feq);
    pclose(gp_pipe);
    HANDLE_ERROR(cudaFree(d_f));
    HANDLE_ERROR(cudaDeviceReset());
    cout<<"Memory freed (CPU+GPU) and Device Reset. "<<endl;
  }
  void Choque(int t){
    // Invoke kernel 2D grid 2D block
    int dimx=4,dimy=2,dimz=Q;
    dim3 block(dimx,dimy,dimz);
    dim3 grid((Lx+block.x-1)/block.x,(Ly+block.y-1)/block.y,(Q+block.z-1)/block.z);
    collisionKernel<<<grid,block>>>((double*)d_f);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }
  void Adveccion(){
    // Invoke kernel 2D grid 2D block
    int dimx=4,dimy=2,dimz=Q;
    dim3 block(dimx,dimy,dimz);
    dim3 grid((Lx+block.x-1)/block.x,(Ly+block.y-1)/block.y,(Q+block.z-1)/block.z);
    streamKernel<<<grid,block>>>((double*)d_f);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }
  void copyBack(){
    HANDLE_ERROR(cudaMemcpy((void*)(h_f+0+0*Lx+0*Lx*Ly),(const void*)(d_f+0+0*Lx+0*Lx*Ly),(size_t)Lx*Ly*Q*sizeof(double),cudaMemcpyDeviceToHost));
  }
  void calcularMacros(){
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double rho=0.0,jx=0.0,jy=0.0,rhoE=0.0,h=0.0;
	for(int iz=0;iz<Q;iz++){
	  double f = *(h_f+idx+iz*Lx*Ly);
	  rho  += f;
	  jx   += *(h_Cx+iz)*f;
	  jy   += *(h_Cy+iz)*f;
	  rhoE += 0.5*(*(h_Cx+iz)*(*(h_Cx+iz))+*(h_Cy+iz)*(*(h_Cy+iz)))*f;
	  if(f>1e-12){
	    h += f*log(f);
	  }
	}
	*(h_rho+idx)   = rho;
	*(h_jx+idx)    = jx;
	*(h_jy+idx)    = jy;
	*(h_rho_e+idx) = rhoE - 0.5*rho*(jx*jx/(rho*rho)+jy*jy/(rho*rho));
	*(h_h+idx)     = h;
      }
    }
  }
  void dibuje3D(int t){
    copyBack();
    calcularMacros();
    
    // Write current data (full resolution for pm3d)
    FILE* tmp = fopen("density.dat", "w");
    for(int iy=0; iy<Ly; iy++){
      for(int ix=0; ix<Lx; ix++){
	int idx = ix + iy*Lx;
	fprintf(tmp, "%d %d %f\n", ix, iy, h_rho[idx]);
      }
      fprintf(tmp, "\n");
    }
    fclose(tmp);
    
    // COLORED SURFACE (pm3d) - much easier to see!
    fprintf(gp_pipe, "set title 'Gaussian Pulse - t=%d'\n", t);
    fprintf(gp_pipe, "splot 'density.dat' with pm3d\n"); //with lines
    fflush(gp_pipe);
    
    usleep(2000000);
  }
  void calcularFeq(){
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double rho = *(h_rho+idx);
	double jx  = *(h_jx+idx);
	double jy  = *(h_jy+idx);
	for(int iz=0;iz<Q;iz++){
	  *(h_feq+idx+iz*Lx*Ly) = h_w[iz]*rho*(1+(h_Cx[iz]*jx/rho+h_Cy[iz]*jy/rho)/c_s2+0.5*((h_Cx[iz]*jx/rho)*(h_Cx[iz]*jx/rho)+2*(h_Cx[iz]*jx/rho)*(h_Cy[iz]*jy/rho)+(h_Cy[iz]*jy/rho)*(h_Cy[iz]*jy/rho))/(c_s2*c_s2)-0.5*(jx*jx/(rho*rho)+jy*jy/(rho*rho))/c_s2);
	}
      }
    }
  }
  void printMaxDifferences(){
    double max_rho_diff = 0.0;
    double max_jx_diff = 0.0;
    double max_jy_diff = 0.0;
    double max_energy_diff = 0.0;
    double max_entropy_diff = 0.0;
    
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	// mass
	double rho_f = *(h_rho+idx);
	double rho_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  rho_feq += *(h_feq+idx+iz*Lx*Ly);
	}
	max_rho_diff = max(max_rho_diff,fabs(rho_f-rho_feq));
	// momentum x
	double jx_f = *(h_jx+idx);
	double jx_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  jx_feq += *(h_feq+idx+iz*Lx*Ly)*h_Cx[iz];
	}
	max_jx_diff = max(max_jx_diff,fabs(jx_f-jx_feq));
	// momentum y
	double jy_f = *(h_jx+idx);
	double jy_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  jy_feq += *(h_feq+idx+iz*Lx*Ly)*h_Cy[iz];
	}
	max_jy_diff = max(max_jy_diff,fabs(jy_f-jy_feq));
	// energy
	double rho_e_f = *(h_rho_e+idx);
	double rho_E_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  rho_E_feq += *(h_feq+idx+iz*Lx*Ly)*0.5*(h_Cx[iz]*h_Cx[iz]+h_Cy[iz]*h_Cy[iz]);
	}
	double energy_feq = rho_E_feq-0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq));
	max_energy_diff = max(max_energy_diff,fabs(rho_e_f-energy_feq));
	// entropy
	double h_h_f = *(h_h+idx);
	double h_h_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  double f = *(h_f+idx+iz*Lx*Ly);
	  if(f > 1e-12){
	    h_h_feq += *(h_feq+idx+iz*Lx*Ly)*log(f);
	  }
	}
	max_entropy_diff = max(max_entropy_diff,fabs(h_h_f-h_h_feq));
      }
    }
    cout<<"Max rho diff: "<<"\t"<<max_rho_diff<<endl;
    cout<<"Max jx diff: "<<"\t"<<max_jx_diff<<endl;
    cout<<"Max jy diff: "<<"\t"<<max_jy_diff<<endl;
    cout<<"Max energy diff: "<<"\t"<<max_energy_diff<<endl;
    cout<<"Max entropy diff: "<<"\t"<<max_entropy_diff<<endl;
  }
  // ===========================================================
  // PHYSICS CHECKS
  // ===========================================================
  bool chequeoMasa(){
    double epsilon = 1.0e-8;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double rho_f = *(h_rho+idx);
	double rho_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  rho_feq += *(h_feq+idx+iz*Lx*Ly);
	}
	if(abs(rho_f - rho_feq) > epsilon){
	  return false;
	}
      }
    }
    return true;
  }
  bool chequeoMomentumX(){
    double epsilon = 1.0e-4;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double jx_f = *(h_jx+idx);
	double jx_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  jx_feq += *(h_Cx+iz)*(*(h_feq+idx+iz*Lx*Ly));
	}
	if(fabs(jx_f - jx_feq) > epsilon){
	  return false;
	}
      }
    }
    return true;
  }
  bool chequeoMomentumY(){
    double epsilon = 1.0e-4;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double jy_f = *(h_jx+idx);
	double jy_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  jy_feq += *(h_Cy+iz)*(*(h_feq+idx+iz*Lx*Ly));
	}
	if(fabs(jy_f - jy_feq) > epsilon){
	  return false;
	}
      }
    }
    return true;
  }
  bool chequeoEnergia(){
    double epsilon = 1.0e-5;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double rho_e_f = *(h_rho_e+idx);
	double rho_E_feq = 0.0;
	double rho_feq = 0.0;
	double jx_feq = 0.0;
	double jy_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  rho_E_feq += 0.5*(*(h_Cx+iz)*(*(h_Cx+iz))+*(h_Cy+iz)*(*(h_Cy+iz)))*(*(h_feq+idx+iz*Lx*Ly));
	  rho_feq += *(h_feq+idx+iz*Lx*Ly);
	  jx_feq += *(h_Cx+iz)*(*(h_feq+idx+iz*Lx*Ly));
	  jy_feq += *(h_Cy+iz)*(*(h_feq+idx+iz*Lx*Ly));
	}
	if(fabs(rho_e_f - (rho_E_feq-0.5*rho_feq*(jx_feq*jx_feq/(rho_feq*rho_feq)+jy_feq*jy_feq/(rho_feq*rho_feq)))) > epsilon){
	  return false;
	}
      }
    }
    return true;
  }
  bool chequeoEntropia(){
    double epsilon  = 1.0e-6;
    for(int ix=0;ix<Lx;ix++){
      for(int iy=0;iy<Ly;iy++){
	int idx = ix + iy*Lx;
	double h_h_f = *(h_h+idx);
	double h_h_feq = 0.0;
	for(int iz=0;iz<Q;iz++){
	  double f = *(h_f+idx+iz*Lx*Ly);
	  if(f>1.0e-12){
	    h_h_feq += *(h_feq+idx+iz*Lx*Ly)*log(f);
	  }
	}
	if(h_h_f + epsilon < h_h_feq ){
	  return false;
	}
      }
    }
    return true;
  }
  void reporte(){
    if(chequeoMasa()){
      cout<<"Mass conservation: OK"<<endl;
    }
    else{
      cout<<"Mass conservation: FAIL"<<endl;
    }
    if(chequeoMomentumX()){
      cout<<"x Momentum conservation: OK"<<endl;
    }
    else{
      cout<<"x Momentum conservation: FAIL"<<endl;
    }
    if(chequeoMomentumY()){
      cout<<"y Momentum conservation: OK"<<endl;
    }
    else{
      cout<<"y Momentum conservation: FAIL"<<endl;
    }
    if(chequeoEnergia()){
      cout<<"Energy conservation: OK"<<endl;
    }
    else{
      cout<<"Energy conservation: FAIL"<<endl;
    }
    if(chequeoEntropia()){
      cout<<"Entropia restriction: OK"<<endl;
    }
    else{
      cout<<"Entropia restriction: FAIL"<<endl;
    }
  }
};	  
int main(){
  LATTICEBOLTZMANN Noah;
  
  // Time loop
  for(int t=0;t<500;t++){
    Noah.Choque((int)t);
    Noah.Adveccion();
    
    if(t%30==0 || t==499){
      Noah.dibuje3D((int)t);
    }
  }
  
  //Noah.copyBack();

  // Post-processing
  //Noah.calcularMacros();
  Noah.calcularFeq();
  //Noah.printMaxDifferences();
  Noah.reporte();
  
  cout<<"Program ended."<<endl;
  return 0;
}
