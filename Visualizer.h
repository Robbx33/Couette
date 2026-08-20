// Visualizer.h
#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Constants.h"
#include "error.h"
#include "GL/glew.h" // (OpenGL-GLUT) extensions packet
#include <GL/glut.h> // (OpenGL-GLUT) functions (texture) 
#include <cuda_gl_interop.h>

class Visualizer{
 private:
  // GLuint is a type of variable in the CPU 
  GLuint texture_id;
  GLuint vbo_positions;
  GLuint vbo_uv;

  // CUDA-OpenGL interop resources
  cudaGraphicsResource_t cuda_texture_resource;
  cudaGraphicsResource_t cuda_vbo_positions_resource;
  cudaGraphicsResource_t cuda_vbo_uv_resource;
  
  // CUDA color buffer 
  uchar4 *d_color;
  double *d_positions;
  double *d_uv;
  
  // Camera rotation angles for 3D view
  double angle_x = 0.0;
  double angle_y = 22.0;
 public:
  Visualizer(int arc,char **argv);
  ~Visualizer(void);
  void copyColorToTexture(uchar4 *d_color);
  void mapVBOs(double **d_positions,double **d_uv);
  void unmapVBOs(void);
  void display(int t,double min_val,double max_val);
  
  uchar4 *get_d_color(void){return d_color;}
  double *get_d_positions(void){return d_positions;}
  double *get_d_uv(void){return d_uv;}
};

#endif
