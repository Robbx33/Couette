// Visualizer.cu
#include "Visualizer.h"

Visualizer::Visualizer(int argc,char **argv){
  // Allocate (GPU-Cuda) buffer for render kernel
  CUDA_CHECK(cudaMalloc((void**)&d_color,Lx*Ly*sizeof(uchar4)));
  CUDA_CHECK(cudaMalloc((void**)&d_positions,Lx*Ly*3*sizeof(double)));
  CUDA_CHECK(cudaMalloc((void**)&d_uv,Lx*Ly*2*sizeof(double)));
  
  // Initialize GLUT
  glutInit((int*)&argc,(char**)argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  // Auto-size window based on domain
  int window_size = 800;
  if(Lx>256 || Ly>256){
    window_size = 1000;
  }
  if(Lx>512 || Ly>512){
    window_size = 1200;
  }
  glutInitWindowSize(window_size,window_size);
  glutCreateWindow("LBM 3D Surface");
  glewInit();

  // 3D PROJECTION
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0,1.0,0.1,100.0);  

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0,-7.0,6.5, 0.0,0.0,0.0, 0.0,0.0,1.0);

  glEnable(GL_DEPTH_TEST);

  // ===== LIGHTING SETUP =====
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
  
  GLfloat light_pos[] = {2.0,2.0,5.0,1.0};
  glLightfv(GL_LIGHT0,GL_POSITION,light_pos);
  
  GLfloat light_ambient[] = {0.3,0.3,0.3,1.0};
  glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient);
  
  GLfloat light_diffuse[] = {0.8,0.8,0.8,1.0};
  glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
  
  glShadeModel(GL_SMOOTH);

  // ==== CREATE TEXTURE ====
  glGenTextures(1,(GLuint*)&texture_id);
  glBindTexture(GL_TEXTURE_2D,(GLuint)texture_id);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,Lx,Ly,0,GL_RGBA,GL_UNSIGNED_BYTE, NULL);
  // Register texture with CUDA 
  cudaGraphicsGLRegisterImage(&cuda_texture_resource,texture_id,GL_TEXTURE_2D,cudaGraphicsMapFlagsWriteDiscard);

  // Vertex positions VBO (x,y,z)
  glGenBuffers(1,&vbo_positions);
  glBindBuffer(GL_ARRAY_BUFFER,vbo_positions);
  glBufferData(GL_ARRAY_BUFFER,Lx*Ly*3*sizeof(double),NULL,GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  // Register Vertex position VBO with CUDA
  cudaGraphicsGLRegisterBuffer(&cuda_vbo_positions_resource,vbo_positions,cudaGraphicsMapFlagsWriteDiscard);

  // Texture coordinates VBO (u,v)
  glGenBuffers(1,&vbo_uv);
  glBindBuffer(GL_ARRAY_BUFFER,vbo_uv);
  glBufferData(GL_ARRAY_BUFFER,Lx*Ly*2*sizeof(double),NULL,GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  // Register Texture coordinates VBO with CUDA
  cudaGraphicsGLRegisterBuffer(&cuda_vbo_uv_resource,vbo_uv,cudaGraphicsMapFlagsWriteDiscard);
}

Visualizer::~Visualizer(void){  
  glDeleteTextures(1,&texture_id);
  glDeleteBuffers(1,&vbo_positions);
  glDeleteBuffers(1,&vbo_uv);
  cudaFree(d_color);
  cudaFree(d_positions);
  cudaFree(d_uv);
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_texture_resource);
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_vbo_positions_resource);
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_vbo_uv_resource);
}

void Visualizer::copyColorToTexture(uchar4 *d_color){
  // Map texture for CUDA access
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_texture_resource,0);
  cudaArray *array;
  cudaGraphicsSubResourceGetMappedArray((cudaArray**)&array,(cudaGraphicsResource_t)cuda_texture_resource,0,0);
  // ==== COPY COLORS TO TEXTURE ====
  cudaMemcpy2DToArray((cudaArray*)array,0,0,(uchar4*)d_color,Lx*sizeof(uchar4),Lx*sizeof(uchar4),Ly,cudaMemcpyDeviceToDevice);
  // ==== UNMAP RESOURCES ====
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_texture_resource,0);
}

void Visualizer::mapVBOs(double **d_positions,double **d_uv){
  // Map positions & UV VBOs
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_positions_resource,0);
  cudaGraphicsResourceGetMappedPointer((void**)d_positions,0,(cudaGraphicsResource_t)cuda_vbo_positions_resource);
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_uv_resource,0);
  cudaGraphicsResourceGetMappedPointer((void**)d_uv,0,(cudaGraphicsResource_t)cuda_vbo_uv_resource);  
}

void Visualizer::unmapVBOs(){
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_positions_resource,0);
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_uv_resource,0);
}

void Visualizer::display(int t,double min_val,double max_val){
  // ==== RENDER ====
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  //glLoadIdentity();
  glPushMatrix();
  glTranslated(0.0,0.0,0.0);
  glRotated((double)angle_x,0.0,1.0,0.0);
  glRotated((double)angle_y,1.0,0.0,0.0);
  
  // Disable lighting to see colors directly
  glEnable(GL_LIGHTING);
  
  //Enable texture
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D,(GLuint)texture_id);
  
  //Position VBO
  glBindBuffer(GL_ARRAY_BUFFER,(GLuint)vbo_positions);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3,GL_DOUBLE,0,0);

  //UV VBO
  glBindBuffer(GL_ARRAY_BUFFER,(GLuint)vbo_uv);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2,GL_DOUBLE,0,0);
  
  // Draw surface
  for(int iy=0;iy<Ly-1;iy++){   
    glDrawArrays(GL_LINE_STRIP,iy*Lx,Lx);
  }
  
  // Cleanup
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER,0);

  glPopMatrix();
  
  // Update window title with current step
  char title[256];
  snprintf(title,sizeof(title),"LBM 3D Surface - Step: %d Min:%.6e Max:%.6e",t,min_val,max_val);
  glutSetWindowTitle(title);
  
  // flip the texture to use it on the back
  glutSwapBuffers();
  glutPostRedisplay();
}
