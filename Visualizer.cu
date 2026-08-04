// Visualizer.cu
#include "Visualizer.h"

Visualizer::Visualizer(LATTICEBOLTZMANN *Noah,KernelsManager *kernels,int argc,char **argv){
  lbm = Noah;
  km = kernels;
  
  // Allocate (GPU-Cuda) buffer for render kernel
  CUDA_CHECK(cudaMalloc((void**)&d_buffer,Lx*Ly*sizeof(uchar4)));
  
  // Initialize CPU-OpenGL-GLUT and create window
  glutInit(&argc, argv);//this function initializes CPU-OpenGL-GLUT toolkit, though argc and argv are empty
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);//GLUT_DOUBLE tells CPU-OpenGL-GLUT that exists 2 drawing areas, like a notebook page, so you can write front and back (GLTEXTURE_2D). GLUT_RGB tells CPU-OpenGL-GLUT to use RGB
  glutInitWindowSize(800,800); //this functions tells CPU-OpenGL-GLUT that a window that will be created have 800pixels per side
  glutCreateWindow("LBM Visualization");//this function tells CPU-OpenGL-GLUT to create that window 
  glewInit();//this function checks if GPU features are compatible for the GPU-OpenGL side
  
  // Create texture
  glGenTextures(1, &texture_id);//this function creates a GPU-OpenGL 1(texture_id=1) textures 
  glBindTexture(GL_TEXTURE_2D, texture_id);//this function activates 1(texture_id=1) GPU-OpenGL texture in 2 dimmensions
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//this function will linearize the texture in the GPU-OpenGL if it is too width
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);//this function will linearize the texture in the GPU-OpenGL if it is too thin
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);//this function will colorthe off extra texture in the GPU-OpenGL if it is too big
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);//this function will restrict the coloring on the edge of the texture in the GPU-OpenGL
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Lx, Ly, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);//this function allocates GPU-OpenGL memory for the texture and sets its format

  // Register texture with CUDA
  cudaGraphicsGLRegisterImage(&cuda_resource,texture_id,GL_TEXTURE_2D,cudaGraphicsMapFlagsWriteDiscard);//register cuda_resource as the part where all the three arguments are the same
  
  printf("Visualizer initialized: %dx%d texture\n", Lx, Ly);
}

Visualizer::~Visualizer(){
  if(d_buffer){
    cudaFree(d_buffer);
  }
  if(cuda_resource){
    cudaGraphicsUnregisterResource(cuda_resource);
  }
}

void Visualizer::display(int t){
  // GPU -> GPU transfer (no CPU copy)
  cudaGraphicsMapResources(1,&cuda_resource,0);//Map texture for CUDA access

  // Get array from texture
  cudaArray *array;
  cudaGraphicsSubResourceGetMappedArray(&array,cuda_resource,0,0);
  
  // Copy GPU buffer to texture (GPU->GPU)
  cudaMemcpy2DToArray(array,0,0,d_buffer,Lx*sizeof(uchar4),Lx*sizeof(uchar4),Ly,cudaMemcpyDeviceToDevice);// this function copies from d_buffer (GPU-Cuda) to array (GPU)

  // Unmap
  cudaGraphicsUnmapResources(1,&cuda_resource,0);
  
  // Draw
  glClear(GL_COLOR_BUFFER_BIT);//this function sets a communication channel in OpenGL between CPU-OpenGL and GPU-OpenGL that flushes the screen
  glEnable(GL_TEXTURE_2D);//this function tells GPU-OpenGL to have the 1(1=texture_id) textures i.e. GL_TEXTURE_2D ready
  // Update texture from host buffer
  glBindTexture(GL_TEXTURE_2D, texture_id);//this function tells again to GPU-OpenGL to remember that 1(1=texture_id) textures was created
  //These lines 55-60, tells OpenGL to draw using the texture (GPU-OpenGL) pining the coordinates in the window (CPU-OpenGL)  
  glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(-1,-1);
    glTexCoord2f(1,0); glVertex2f(1,-1);
    glTexCoord2f(1,1); glVertex2f(1,1);
    glTexCoord2f(0,1); glVertex2f(-1,1);
  glEnd();
  glDisable(GL_TEXTURE_2D);//sets free the 1(1=texture_id) textures in the GPU-OpenGL

  // Update window title with current step
  char title[256];
  snprintf(title,sizeof(title),"LBM Simulation - Step %d", t);
  glutSetWindowTitle(title);
  
  glutSwapBuffers();//flip the texture to be use as again but in the back
}
