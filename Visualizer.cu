// Visualizer.cu
#include "Visualizer.h"

Visualizer::Visualizer(LATTICEBOLTZMANN *Noah,KernelsManager *kernels,int argc,char **argv){
  lbm = Noah;
  km = kernels;
  
  // Allocate (GPU-Cuda) buffer for render kernel
  CUDA_CHECK(cudaMalloc((void**)&d_color,Lx*Ly*sizeof(uchar4)));
  
  // Initialize GLUT
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(800,800);
  glutCreateWindow("LBM 3D Surface");
  glewInit();

  // 3D PROJECTION
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0,1.0,0.1,100.0);  // Instead of glOrtho
  //glOrtho(-2.0,-2.0,-2.0,2.0,-10.0,10.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0,-1.0,1.5, 0.0,0.0,0.0, 0.0,0.0,1.0);

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
  glGenTextures(1,&texture_id);
  glBindTexture(GL_TEXTURE_2D,texture_id);
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
  
  printf("Visualizer initialized: 3D surface with interop\n");
}

Visualizer::~Visualizer(){
  if(texture_id){
    glDeleteTextures(1,&texture_id);
  }
  if(vbo_positions){
    glDeleteBuffers(1,&vbo_positions);
  }
  if(vbo_uv){
    glDeleteBuffers(1,&vbo_uv);
  }
  if(d_color){
    cudaFree(d_color);
  }
  if(cuda_texture_resource){
    cudaGraphicsUnregisterResource(cuda_texture_resource);
  }
  if(cuda_vbo_positions_resource){
    cudaGraphicsUnregisterResource(cuda_vbo_positions_resource);
  }
  if(cuda_vbo_uv_resource){
    cudaGraphicsUnregisterResource(cuda_vbo_uv_resource);
  }
}

void Visualizer::display(int t){
  // ==== MAP RESOURCES FOR CUDA ====
  // Map texture for CUDA access
  cudaGraphicsMapResources(1,&cuda_texture_resource,0);
  cudaArray *array;
  cudaGraphicsSubResourceGetMappedArray(&array,cuda_texture_resource,0,0);

  // Map positions VBO
  cudaGraphicsMapResources(1,&cuda_vbo_positions_resource,0);
  double *d_positions;
  cudaGraphicsResourceGetMappedPointer((void**)&d_positions,0,cuda_vbo_positions_resource);

  // Map UV VBO
  cudaGraphicsMapResources(1,&cuda_vbo_uv_resource,0);
  double *d_uv;
  cudaGraphicsResourceGetMappedPointer((void**)&d_uv,0,cuda_vbo_uv_resource);
  
  // ==== LAUNCH RENDER & FILL KERNEL ====
  // Does: colors + vertex positions + texture coordinates in one pass
  km->launchRenderAndFill(d_color,d_positions,d_uv,lbm->get_d_rho());

  // ==== COPY COLORS TO TEXTURE ====
  cudaMemcpy2DToArray(array,0,0,d_color,Lx*sizeof(uchar4),Lx*sizeof(uchar4),Ly,cudaMemcpyDeviceToDevice);

  // ==== UNMAP RESOURCES ====
  cudaGraphicsUnmapResources(1,&cuda_texture_resource,0);
  cudaGraphicsUnmapResources(1,&cuda_vbo_positions_resource,0);
  cudaGraphicsUnmapResources(1,&cuda_vbo_uv_resource,0);

  // ==== RENDER ====
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glLoadIdentity();
  glTranslated(0.0,-3.5,-6.0);
  glRotated(angle_y,0.0,1.0,0.0);
  glRotated(angle_x,1.0,0.0,0.0);
  
  // Disable lighting to see colors directly
  glEnable(GL_LIGHTING);
  
  //Enable texture
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D,texture_id);
  
  //Position VBO
  glBindBuffer(GL_ARRAY_BUFFER,vbo_positions);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3,GL_DOUBLE,0,0);

  //UV VBO
  glBindBuffer(GL_ARRAY_BUFFER,vbo_uv);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2,GL_DOUBLE,0,0);
  
  // Draw surface
  for(int iy=0;iy<Ly-1;iy++){   // step=2 for speed
    glDrawArrays(GL_LINE_STRIP,iy*Lx,Lx);
  }
  // Draw as triangles using GL_TRIANGLES
  //glDrawArrays(GL_LINE, 0, Lx * Ly);
  
  // Cleanup
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  
  // Update window title with current step
  char title[256];
  snprintf(title,sizeof(title),"LBM 3D Surface - Step %d",t);
  glutSetWindowTitle(title);
  
  // flip the texture to use it on the back
  glutSwapBuffers();
  glutPostRedisplay();
}
