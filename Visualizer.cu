// Visualizer.h     -> class declaration
// iostream  -> std::cout for status messages
// using namespace std lets us write cout instead of std::cout
#include "Visualizer.h"
#include <iostream>
using namespace std;

// ===========================================================
// CONSTRUCTOR-Visualizer
// ===========================================================
Visualizer::Visualizer(int argc,char **argv,PHYSICSCHECKER *Noah,RESULTS *Jeremias,KernelsManager *Gargantua){
  // Point lbm at the begining of LATTICEBOLTZMANN class making it equal
  // to its respective object passed in. pck points to the begining of the
  // class PHYSICSCHECKER and jm points to the RESULTS class
  km = Gargantua;
  pck = Noah;
  jm = Jeremias;
  
  // Kill any leftover gnuplot process from a previous run, then wait
  // 200 ms to make sure it's fully gone before continuing.
  system("pkill -f gnuplot 2>/dev/null");
  usleep(200000);
  
  // Open a pipe to gnuplot and a file to log the physics errors.
  // The pipe lets us send plot commands directly from C++.
  // The data file holds the numeric values that gnuplot reads.
  // first_call flags the very first plot so gnuplot runs 'plot' once,
  // then 'replot' on every later timestep.
  gp_pipe = popen("gnuplot -persist", "w");
  first_call = 1;
  
  // Send initial plot configuration to gnuplot through the pipe:
  // title, axis labels, grid, and a logarithmic y-axis (errors span
  // many orders of magnitude, so log scale makes them readable).
  // fflush pushes the commands to gnuplot immediately instead of
  // waiting for the buffer to fill up
  fprintf(gp_pipe,"set title 'conservation and restriction quantities vs time'\n");
  fprintf(gp_pipe,"set xlabel 'time step'\n");
  fprintf(gp_pipe,"set ylabel 'Error'\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set logscale y\n");
  fflush(gp_pipe);

  // Open one gnuplot process per macroscopic field, each in its own
  // window. Commands are sent later via fprintf through these pipes,
  // and the .dat files they plot are written by RESULTS::writeMacros.
  // Closed in the destructor with pclose.
  gp_pipe_rho = popen("gnuplot -persist","w");
  gp_pipe_jx = popen("gnuplot -persist","w");
  gp_pipe_jy = popen("gnuplot -persist","w");
  gp_pipe_rho_e = popen("gnuplot -persist","w");
  gp_pipe_h = popen("gnuplot -persist","w");

  // Send the initial "set ..." configuration to each gnuplot pepe:
  // azis labels, grid, xrange, and wondow title. Called once here
  // so the windows are ready before the first plotMacros() call.
  configureGnuplotPipe(gp_pipe_rho,"rho");
  configureGnuplotPipe(gp_pipe_jx,"jx");
  configureGnuplotPipe(gp_pipe_jy,"jy");
  configureGnuplotPipe(gp_pipe_rho_e,"rho_e");
  configureGnuplotPipe(gp_pipe_h,"h");
  
  // Allocate (GPU-Cuda) device buffer that the render kernel writes to.
  // Its content get copied into the OpenGL texture at
  // render time (copyColorToTexture in renderAndDisplay).
  //
  //    d_color      one RGBA color per lattice site
  CUDA_CHECK(cudaMalloc((void**)&d_color,Lx*Ly*sizeof(uchar4)));
  
  // Initialize GLUT and pick the display mode.
  //    GLUT_DOUBLE  use double buffering (draw off-screen, then swap)
  //    GLUT_RGB     standard RGB color buffer
  //    GLUT_DEPTH   enable the depth buffer for 3D hidden-surface removal
  glutInit((int*)&argc,(char**)argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  
  // Create the GLUT window and initialize GLEW
  // The window size scales with the domain so larger grids still get a
  // usable window:
  //   Lx,Ly <= 256 ->  800x800
  //   Lx,Ly <= 512 -> 1000x1000
  //   Lx,Ly > 256  -> 1200x1200
  // The title names the window. glewInit() must come after
  // glutCreatWindow so the OpenGL context exists when GLEW probes it.
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

  // Install the perspective projection.
  //   glMatrixMode(GL_PROJECTION)  select the projection matrix, so the
  //                                next calls affect how the 3D world is
  //                                flattened onto the 2D screen
  //   glLoadIdentity()             reset it to the identity (no transform)
  //   gluPerspective(45, 1, 0.1, 100)
  //       fovy   = 45 degrees vertical field of view
  //       aspect = 1.0 (square, matches the square window)
  //       zNear  = 0.1 (anything closer to the camera is clipped)
  //       zFar   = 100 (anything farther is clipped)
  // Objects farther from the camera appear smaller, giving a sense of depth.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0,1.0,0.1,100.0);  

  // Set the camera for the modelview matrix.
  //   eye    = (0,-7,6.5) camera sits behind (negative y) and above
  //                       (positive z) the domain
  //   center = (0,0,0)    looks toward the domain center
  //   up     = (0,0,1)    z is "up" in this scene
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0,-7.0,6.5, 0.0,0.0,0.0, 0.0,0.0,1.0);

  //Enable depth testing. When OpenGL draws two fragments at the same
  //screen pixel, the one closer to the camera wins. Without this, later
  //draws would always paint over earlier ones regardless of depth, and the 3D surface would look wring where  it folds over itself.
  glEnable(GL_DEPTH_TEST);

  // Basic lighting setup.
  //   glEnable(GL_LIGHTING)       turn lighting on (otherwise colors
  //                               are used directly, with no shading)
  //   glEnable(GL_LIGHT0)         enable the first light source
  //   glEnable(GL_COLOR_MATERIAL) let per-vertex colors act as material
  //                               colors, so the colors texture affects
  //                               how the surface responds to light
  //   glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
  //                               apply these colors to both ambient
  //                               and diffuse components, on both the
  //                               front and back faces of the surface 
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);

  // Position of the first light source, in world coordinates.
  // The four values are (x,y,z,w): w=1.0 means a positional light
  // at (2,2,5), not a directional light. So the light sits above and
  // to one side of the domain, and the surface is lit from that corner
  GLfloat light_pos[] = {2.0,2.0,5.0,1.0};
  glLightfv(GL_LIGHT0,GL_POSITION,light_pos);

  // Ambient color of the light: the base lighting that hits every
  // surface from all directions. RGB = (0.3, 0.3, 0.3), a dim gray, so
  // even surfaces facing away from the light are not pitch black.
  // The fourth value is alpha (1.0), unused for lights
  GLfloat light_ambient[] = {0.3,0.3,0.3,1.0};
  glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient);

  // Diffusive color of the light: the light that reflects off the surface
  // according to how directly it faces the light. RGB = (0.8,0.8,0.8), a
  // bright gray, giving stron shading acdross sthe surface. The fourth
  // value is alpha (1.0), unused for lights.
  GLfloat light_diffuse[] = {0.8,0.8,0.8,1.0};
  glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);

  // Use smooth (Gouraud) shading instead of flat shading. Colors and
  // lighting are interpolated across each polygon, so the surface looks
  // continuous rather than blocky.
  glShadeModel(GL_SMOOTH);

  // Create the 2D texture that will hold the per-site colors
  //   glGenTextures   reserve one texture object, store its id
  //   glBindTexture   make it the current texture 2D texture
  //   GL_TEXTURE_MIN_FILTER/_MAG_FILTER = GL_LINEAR
  //                   when the texture is scaled down or up, interpolate
  //                   between neghboring texels so colors look smooth
  //   GL_TEXTURE_WRAP_S/_T = GL_CLAMP
  //                   don't repeat the texture at the edges; clamp to the
  //                   border texel instead
  //   glTexImage2D    allocate the texture storage, Lx by Ly texels,
  //                   RGBA, one undigned byte per channel. The NULL
  //                   means we don't provide initial pixel data here;
  //                   the colors are written later via CUDA interop
  glGenTextures(1,(GLuint*)&texture_id);
  glBindTexture(GL_TEXTURE_2D,(GLuint)texture_id);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,Lx,Ly,0,GL_RGBA,GL_UNSIGNED_BYTE, NULL);
  // Register texture with CUDA so the render kernel can write to it
  // directly. After this, cuda_texture_resource is teh CUDA handle used
  // to map the texture and get a device pointer to its contents.
  // cudaGraphicsMapFlagsWriteDiscard tells CUDA we only write to it,
  // which lets the driver skip loading the old contents.
  cudaGraphicsGLRegisterImage(&cuda_texture_resource,texture_id,GL_TEXTURE_2D,cudaGraphicsMapFlagsWriteDiscard);

  // Create the vertex buffer object (VBO) for the surface positions.
  //   glGenBuffers        reserve one buffer, store its id
  //   glBindBuffer        make it the current GL_ARRAY_BUFFER
  //   glBufferData        allocate storage: Lx*Ly vertices, 3 doubles
  //                       each (x,y,z). NULL means no initial data;
  //                       the render kernel fills it later.
  //                       GL_DYNAMIC_DRAW  hints that it will be rewritten
  //                       often so the criver places it where updates are
  //                       cheap.
  //   glBindBuffer(...,0) unbind, so later code doesn't accidentally
  //                       touch this buffer
  glGenBuffers(1,&vbo_positions);
  glBindBuffer(GL_ARRAY_BUFFER,vbo_positions);
  glBufferData(GL_ARRAY_BUFFER,Lx*Ly*3*sizeof(double),NULL,GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  // Register the position VBO with CUDA so the render kernel can write
  // to it directly. After this, cuda_vbo_positions_resource is the CUDA
  // handle used to map the VBO and get a device pointer to its storage.
  // cudaGraphicsMapFlagsWriteDiscard: we only write to it, so the driver
  // skips preserving whatever was there.
  cudaGraphicsGLRegisterBuffer(&cuda_vbo_positions_resource,vbo_positions,cudaGraphicsMapFlagsWriteDiscard);

  // Create the vertex buffer object (VBO) (u,v) for the texture coordinates.
  //   glGenBuffers         reserve one buffer, store its id
  //   glBindBuffer         make it the current GL_ARRAY_BUFFER
  //   glBufferData         allocate storage: Lx*Ly vertices, 2 doubles
  //                        each (u, v). NULL means no initial data;
  //                        the render kernel fills it later.
  //                        GL_DYNAMIC_DRAW hints that it will be
  //                        rewritten often, so the driver places it
  //                        where updates are cheap.
  //   glBindBuffer(...,0)  unbind, so later code doesn't accidentally
  //                        touch this buffer
  glGenBuffers(1,&vbo_uv);
  glBindBuffer(GL_ARRAY_BUFFER,vbo_uv);
  glBufferData(GL_ARRAY_BUFFER,Lx*Ly*2*sizeof(double),NULL,GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  // Register the UV VBO with CUDA so the render kernel can write to it
  // directly. After this, cuda_vbo_uv_resource is the CUDA handle used
  // to map the VBO and get a device pointer to its storage.
  // cudaGraphicsMapFlagsWriteDiscard: we only write to it, so the driver
  // skips preserving whatever was there.
  cudaGraphicsGLRegisterBuffer(&cuda_vbo_uv_resource,vbo_uv,cudaGraphicsMapFlagsWriteDiscard);
}

// ===========================================================
// DESTRUCTOR-Visualizer
// ===========================================================
Visualizer::~Visualizer(void){
  // Close the gnuplot pipe and the data file cleanly.
  // 'exit' tells gnuplot to quit; fflush makes sure the command is
  // sent; pclose waits for gnuplot to terminate and closes the pipe.
  // fclose flushes and closes the data file.
  fprintf(gp_pipe, "exit\n");
  fflush(gp_pipe);
  pclose(gp_pipe);

  // Close the five gnuplot pipes opened in the constructor. pclose
  // waits for each gnuplot process to terminate and then releases the
  // pipe. After this, the associated windows close.
  pclose(gp_pipe_rho);
  pclose(gp_pipe_jx);
  pclose(gp_pipe_jy);
  pclose(gp_pipe_rho_e);
  pclose(gp_pipe_h);

  // Free the OpenGL objects created in the constructor
  //   glDeleteTextures  release texture_id
  //   glDeleteBuffers   release vbo_positions and vbo_uv
  // After this, the ids are invalid and must not be used
  glDeleteTextures(1,&texture_id);
  glDeleteBuffers(1,&vbo_positions);
  glDeleteBuffers(1,&vbo_uv);

  // Free the CUDA-side texture buffer allocated in the constructor for the
  // render kernel (per-site color). This is separated
  // from the OpenGL objects; only the CUDA allocation is realeased here.
  CUDA_CHECK(cudaFree((uchar4*)d_color));

  // Unregister the CUDA-OpenGL interop handles created in the constructor.
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_texture_resource);
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_vbo_positions_resource);
  cudaGraphicsUnregisterResource((cudaGraphicsResource_t)cuda_vbo_uv_resource);

  // Visualizer checkout
  cout<<"Memory freed Visualizer(CPU+GPU)."<<endl;
}

void Visualizer::copyColorToTexture(uchar4 *d_color){
  // Copy the per-site colors from the CUDA buffer d_color into the
  // OpenGL texture, so the surface can be drawn with those colors.
  //
  // The three steps are the standard CUDA-OpenGL interop dance:
  //   1. Map the texture so CUDA can access it, and grab the underlying
  //      cudaArray.
  //   2. Copy the color data into the cudaArray, device-to-device, one
  //      row of Lx uchar4 values at a time for Ly rows.
  //   3. Unmap the texture so OpenGL can use it again.
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_texture_resource,0);
  cudaArray *array;
  cudaGraphicsSubResourceGetMappedArray((cudaArray**)&array,(cudaGraphicsResource_t)cuda_texture_resource,0,0);
  cudaMemcpy2DToArray((cudaArray*)array,0,0,(uchar4*)d_color,Lx*sizeof(uchar4),Lx*sizeof(uchar4),Ly,cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_texture_resource,0);
}

void Visualizer::mapVBOs(double **vbo_positions_ptr,double **vbo_uv_ptr){
  // Map positions & UV VBOs so CUDA can write to them, and return
  // the device pointers via vbo_positions_ptr and vbo_uv_ptr. The render kernel uses
  // these pointers to fill in the (x,y,z) position and (u,v) texture
  // coordinates of every lattice site.
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_positions_resource,0);
  cudaGraphicsResourceGetMappedPointer((void**)vbo_positions_ptr,0,(cudaGraphicsResource_t)cuda_vbo_positions_resource);
  cudaGraphicsMapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_uv_resource,0);
  cudaGraphicsResourceGetMappedPointer((void**)vbo_uv_ptr,0,(cudaGraphicsResource_t)cuda_vbo_uv_resource);  
}

void Visualizer::unmapVBOs(){
  // Unmap the position and UV VBOs after the render kernel has written
  // to them
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_positions_resource,0);
  cudaGraphicsUnmapResources(1,(cudaGraphicsResource_t*)&cuda_vbo_uv_resource,0);
}

void Visualizer::display(int t,double min_val,double max_val){
  // Draw one frame of the 3D surface.
  //   1. Clear the color and depth buffers.
  //   2. Save the current modelview matrix with glPushMatrix, then apply
  //      the camera rotation: angle_x around y, angle_y around x, and a
  //      no-op translation. The perspective and camera set up in the
  //      constructor are preserved below this push.
  //   3. Enable lighting (kept on for depth shading), enable 2D texturing,
  //      and bind the color texture produced by copyColorToTexture.
  //   4. Bind the position VBO and tell OpenGL to read it as a 3-double
  //      vertex array. Then bind the UV VBO and enable the texture
  //      coordinate array from it. Each vertex now carries a position
  //      and a (u,v) coordinate into the texture.
  //   5. Draw the surface as one GL_LINE_STRIP per grid row: Ly-1 strips,
  //      each starting at iy*Lx and containing Lx vertices.
  //   6. Disable the two client-side arrays, unbind the buffer, and pop
  //      the modelview matrix so the push from step 2 is undone.
  //   7. Update the window title with the timestep and the min/max of
  //      the data being shown.
  //   8. Swap the double buffers (show the newly drawn frame) and request
  //      a redraw.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  glPushMatrix();
  glTranslated(0.0,0.0,0.0);
  glRotated((double)angle_x,0.0,1.0,0.0);
  glRotated((double)angle_y,1.0,0.0,0.0);
  
  glEnable(GL_LIGHTING);
  
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D,(GLuint)texture_id); 
  glBindBuffer(GL_ARRAY_BUFFER,(GLuint)vbo_positions);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3,GL_DOUBLE,0,0);
  glBindBuffer(GL_ARRAY_BUFFER,(GLuint)vbo_uv);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2,GL_DOUBLE,0,0);
  
  for(int iy=0;iy<Ly-1;iy++){   
    glDrawArrays(GL_LINE_STRIP,iy*Lx,Lx);
  }
  
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER,0);
  glPopMatrix();
  
  char title[256];
  snprintf(title,sizeof(title),"LBM 3D Surface - Step: %d Min:%.6e Max:%.6e",t,min_val,max_val);
  glutSetWindowTitle(title);
  
  glutSwapBuffers();
  glutPostRedisplay();
}

void Visualizer::renderAndDisplay(double *d_data,int t){
  // Render one frame of the 3D surface from d_data.
  //
  //   1. findMinMax gets the smallest and largest values of d_data so
  //      the colors and height can be normalized.
  //   2. center and scale turn that range into a midpoint and half-width
  //      for the kernel. scale is clamped so a constant field does not
  //      divide by zero.
  //        center = the center of your data range
  //        scale  = the radius of your data range (half the width)
  //        (min_val) → 5.0 (center) → 7.5 (scale) → 2.5 (max_val) → 10.0
  //   3. copyColorToTexture copies the per-site colors into the GL texture.
  //   4. mapVBOs gives device pointers to the position and UV buffers.
  //   5. launchRenderAndFill runs the render kernel.
  //   6. unmapVBOs hands the buffers back to OpenGL.
  //   7. display draws the frame.
  double min_val,max_val;
  jm->findMinMax((double*)d_data,(double*)&min_val,(double*)&max_val,0);
  
  double center = (min_val+max_val)/2.0;
  double scale = (max_val-min_val)/2.0;
  if(scale<1e-30){
    scale = 1.0;
  }
  
  copyColorToTexture((uchar4*)d_color);

  double *vbo_positions_ptr,*vbo_uv_ptr;
  mapVBOs((double**)&vbo_positions_ptr,(double**)&vbo_uv_ptr);
  
  km->launchRenderAndFill((uchar4*)d_color,(double*)vbo_positions_ptr,(double*)vbo_uv_ptr,(double*)d_data,(double)center,(double)scale);
  
  unmapVBOs();
  
  display((int)t,(double)min_val,(double)max_val);
}

void Visualizer::plotConservationRestriction(int t,int offset){
  // Update the conservation and restriction plot for this time step.
  //
  // jm->writeConservationRestriction(t, offset) computes, for each of the
  // five diagnostic arrays (mass, x-momentum, y-momentum, energy,
  // entropy), the min and max over the grid and appends a row to
  // conservation&restriction_quantities.dat:
  //
  //     t  mass_min mass_max  momX_min momX_max  momY_min momY_max
  //        energy_min energy_max  entropy_min entropy_max
  //
  // Then gnuplot is told to redraw. On the first call it receives the
  // full 'plot' command, which defines what each column pair means and
  // labels them in the legend. first_call is set to 0. On every later
  // call gnuplot just gets 'replot', which redraws the same plot with
  // the updated file
  jm->writeConservationRestriction((int)t,(int)offset);
 
  if(first_call==1){
    fprintf((FILE*)gp_pipe,"plot 'conservation&restriction_quantities.dat' using 1:2 w lp title 'mass_min', '' using 1:3 w lp title 'mass_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:4 w lp title 'momX_min', '' using 1:5 w lp title 'momX_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:6 w lp title 'momY_min', '' using 1:7 w lp title 'momY_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:8 w lp title 'energy_min', '' using 1:9 w lp title 'energy_max', ");
    fprintf((FILE*)gp_pipe,"'' using 1:10 w lp title 'entropy_min', '' using 1:11 w lp title 'entropy_max'\n");
    fflush((FILE*)gp_pipe);
    first_call = 0;
  }
  else{
    fprintf((FILE*)gp_pipe,"replot\n");
    fflush((FILE*)gp_pipe);
  } 
}

void Visualizer::configureGnuplotPipe(FILE *gp_pipe,const char *title){
  // Send the initial "set ..." commands to one gnuplot pipe, so the
  // window is configured before the first plot arrives:
  //   xlabel       the horizontal axis is the grid index ix
  //   ylabel       left empty
  //   grid         draw grid lines
  //   xrange       limit x to [0, Lx]
  //   title        use the given title for the window
  // Called once per pipe from the constructor.
  fprintf(gp_pipe,"set xlabel 'ix'\n");
  fprintf(gp_pipe,"set ylabel ' '\n");
  fprintf(gp_pipe,"set grid\n");
  fprintf(gp_pipe,"set xrange [0:%d]\n",Lx);
  fprintf(gp_pipe,"set title '%s'\n",title);
  fflush(gp_pipe);
}

void Visualizer::plotMacros(int t){
  // Update the five macroscopic-field plots for this time step.
//
//   1. jm->writeMacros(t) copies rho, jx, jy, rho_e, and h from GPU
//      to host and writes each to its own .dat file.
//   2. For each field, send its gnuplot pipe a title and a 'plot'
//      command for the corresponding .dat file.
//   3. fflush after each pipe so the command reaches gnuplot
//      immediately.
//   4. usleep(1s) throttles the update rate so gnuplot has time to
//      redraw before the next iteration overwrites the files.
  jm->writeMacros((int)t);
  
  fprintf(gp_pipe_rho,"set title 'rho - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_rho,"plot 'rho_2D.dat' with lines lw 0.2\n");
  fflush(gp_pipe_rho); 
  fprintf(gp_pipe_jx,"set title 'jx - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_jx,"plot 'jx_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_jx);
  fprintf(gp_pipe_jy,"set title 'jy - t=%d (max=) (min=)'\n",t);
  fprintf(gp_pipe_jy,"plot 'jy_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_jy);
  fprintf(gp_pipe_rho_e,"set title 'rho_e - t=%d (max=) (min=)'\n", t);
  fprintf(gp_pipe_rho_e,"plot 'rho_e_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_rho_e);
  fprintf(gp_pipe_h,"set title 'h - t=%d (max=, min=)'\n", t);
  fprintf(gp_pipe_h,"plot 'h_2D.dat' with lines lw 0.1\n");
  fflush(gp_pipe_h);
  
  usleep(1000000);
}
