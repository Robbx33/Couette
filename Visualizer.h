// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  VISUALIZER_H is not defined -> enter, read the class,
//                       and #define VISUALIZER_H to mark it as "already read".
//   - Any later inclusion: VISUALIZER_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef VISUALIZER_H
#define VISUALIZER_H

// Results.h        -> call writeMacros(t) before plotting
// unistd.h         -> usleep (pauses after killing old gnuplot processes,
//                     and between plot updates)
// GL/glew.h        -> OpenGL-GLUT extension packet loader (just come before glut.h)
// GL/glut.h        -> OpenGL-GLUT window, rendering, OpenGL entry points and
//                     functions (texture)
// cuda_gl_interop.h-> CUDA-OpenGL interop (register texture and VBOs,
//                     map/unmap them for render kernel)
#include "Results.h"
#include <unistd.h>
#include "GL/glew.h"
#include <GL/glut.h> 
#include <cuda_gl_interop.h>

class Visualizer{
 private:  
  // GLuint is a type of variable in the CPU
  //
  //   texture_id     2D texture that holds the per-site color map.
  //                  Written to on the GPU (via CUDA interop) and read
  //                  by OpenGL when drawing the surface.
  //
  //   vbo_positions  Vertex Buffer Object holding the (x,y,z) position
  //                  of each lattice site. Written by renderAndfillKernel.
  //
  //   vbo_uv         Vertex Buffer Object holding the (u,v) texture
  //                  coordinates for each site, so the color texture
  //                  maps correctly onto the surface.
  GLuint texture_id;
  GLuint vbo_positions;
  GLuint vbo_uv;

  // CUDA-OpenGL interop handles. Each one links an OpenGL object to CUDA
  // so the render kernel can write to it directly, without going through
  // the host.
  //
  //   cuda_texture_resource         links texture_id
  //   cuda_vbo_positions_resource   links vbo_positions
  //   cuda_vbo_uv_resource          links vbo_uv
  //
  // Mapped before launching the render kernel (so the kernel gets device
  // pointers into these GL objects) and unmapped after.
  cudaGraphicsResource_t cuda_texture_resource;
  cudaGraphicsResource_t cuda_vbo_positions_resource;
  cudaGraphicsResource_t cuda_vbo_uv_resource;
  
  // Device buffers that mirrors the OpenGL resources, so the render
  // kernel has something to write to. It is allocated here but
  // its content is copied into the GL texture at render
  // time (copColorToTexture in renderAndDisplay).
  //
  //   d_color    per-site RGBA color, one entry per lattce site
  uchar4 *d_color;
  
  // Camera rotation angles (in degrees) for 3D view
  // Applied in display() before drawing the surface
  // angle_x rotates around the vertical axis (left/right view)
  // angle_y tilts the surface toward the viewer (up/down view)
  double angle_x = 0.0;
  double angle_y = 22.0;

  // Gnuplot output for the conservation diagnostics
  //
  // gp_pipe    pipe to a gnuplot process; commands are written to it
  //            via fprintf and gnuplot draws the plots
  // first_call flag: 1 on the first diagnose() call so gnuplot receives
  //            'plot' once, then 0 so later calls send 'replot'
  FILE *gp_pipe;
  int first_call;

  // Gnuplot pipes for the 2D line plots of the macroscopic fields
  // Each pipe is a separate gnuplot process with its own window
  //
  //   gp_pipe_rho     plots rho     vs ix
  //   gp_pipe_jx      plots jx      vs ix
  //   gp_pipe_jy      plots jy      vs ix
  //   gp_pipe_rho_e   plots rho_e   vs ix
  //   gp_pipe_h       plots h       vs ix
  //
  // The .dat files those plots read are written by RESULTS::writeMacros,
  // which Visualizer::plotMacros calls before sending the plot commands
  FILE *gp_pipe_rho;
  FILE *gp_pipe_jx;
  FILE *gp_pipe_jy;
  FILE *gp_pipe_rho_e;
  FILE *gp_pipe_h;

  // Pointers to the other two objects this class works with.
  //
  // lbm points to the Gauss object that points to the start
  //     of the LATTICEBOLTZMANN start of the class, pck points
  // to the begining of the PHYSICSCHECKER class and km points
  // to the begining of the KernelsManager class
  KernelsManager *km;
  PHYSICSCHECKER *pck;
  RESULTS *jm;
public:
  // Lifecycle of the Results object: setup and cleanup.
  // The constructor takes a pointer to LATTICEBOLTZMANN, other to
  // PHYSICSCHECKER and other to KernelsManager
  // destructor frees host and device memory
  Visualizer(int arc,char **argv,PHYSICSCHECKER *Noah,RESULTS *Jeremias,KernelsManager *Gargantua);
  ~Visualizer(void);

  // Updates the conservation & restriction plot
  //
  // Calls RESULTS::writeConservationRestriction(t,offset) to append
  // the current min/max row to onservation&restriction_quantities.dat,
  // then sends gnuplot a 'plot' the first time or 'replot' on every
  // later call, so the wondow show the updated data.
  void plotConservationRestriction(int t,int offset);

  // Macroscopic-field plots, one gnuplot window per field (rho,jx,jy,rho_e,h)
  //
  //   configureGnuplotPipe(gp_pipe,title)
  //     Sends the intial "set ..." commands to one gnupolt pipe:
  //     xlabel, ylabel,grid, xrange, and window title. Called
  //     once per pipe from the constructor.
  //
  //   plotMacros(t)
  //     Asks RESULTS to refresh the .dat files with the current
  //     macroscopic fields (jm->writeMacros), the sends each
  //     gnuplot pipe a 'plot' command for its file. The 1-second
  //     sleep at the end throttles the update rate.
  void configureGnuplotPipe(FILE *gp_pipe,const char *title);
  void plotMacros(int t);

  // OpenGL rendering path.
  //   copyColorToTexture(d_color)
  //     Maps the CUDA-GL texture resource and copies the per-site
  //     RGBA colors from d_color into the OpenGL texture. Then unmaps.
  //
  //   mapVBOs(d_positions,d_uv)
  //     Maps the CUDA-GL VBO resources and returns the device pointers
  //     to the positions and UV buffers, so the render kernel can write
  //     to them.
  //
  //   unmapVBOs()
  //     Unmaps the position and UV VBO resources after the render
  //     kernel has finished.
  //
  //   display(t,min_val,max_val)
  //     Draws the surface: sets the camera, binds the texture and the
  //     two VBOs, draws one GL_LINE_STRIP per grid row, updates the
  //     window title with the timestep and range, and swaps buffers.
  //
  //   renderAndDisplay(d_data,t)
  //     Full render of one frame. Finds min/max of d_data (via
  //     RESULTS::findMinMax) to normalize colors and height, copies
  //     the colors to the texture, maps the VBOs, launches the render
  //     kernel through KernelsManager, unmaps the VBOs, and calls
  //     display().
  void copyColorToTexture(uchar4 *d_color);
  void mapVBOs(double **vbo_positions_ptr,double **vbo_uv_ptr);
  void unmapVBOs(void);
  void display(int t,double min_val,double max_val);
  void renderAndDisplay(double *d_data,int t);
};

#endif
