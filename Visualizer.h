// Visualizer.h
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "LBM.h"
#include "KernelsManager.h"
#include "GL/glew.h" // (OpenGL-GLUT) extensions packet
#include <GL/glut.h> // (OpenGL-GLUT) functions (texture) 

class Visualizer{
 private:
  LATTICEBOLTZMANN *lbm;
  KernelsManager *km;
  
  //GLuint is a type of variable in the CPU 
  GLuint texture_id;//(GLuint-CPU) variable
  
  //uchar4 is a vector type that holds uncharacter data (4 bytes each - RGBA)
  uchar4 *d_buffer;//GPU memory used to hold RGBA colors 
  uchar4 *h_buffer;//CPU memory used to hold RGBA colors
 public:
  Visualizer(LATTICEBOLTZMANN *Noah,KernelsManager *kernels,int arc,char **argv);
  ~Visualizer();
  void display(int t);
  
  uchar4 *get_d_buffer(){return d_buffer;}
};

#endif
