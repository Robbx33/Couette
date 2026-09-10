// The #ifndef / #define / #endif block below is an "include guard".
//
// Without it, if this header is included more than once in the same
// compilation (e.g., directly by main.cu and also indirectly through
// another header), the compiler would read the class definition twice
// and fail with a "redefinition" error.
//
// How it works:
//   - First inclusion:  ERROR_H is not defined -> enter, read the class,
//                       and #define ERROR_H to mark it as "already read".
//   - Any later inclusion: ERROR_H is already defined -> skip everything
//                          until #endif. The class is not read again.
//
// Purely a compile-time mechanism. Nothing to do with objects or runtime.
#ifndef ERROR_H
#define ERROR_H

// stdio.h  -> fprintf, stderr (error messages)
// stdlib.h -> exit, EXIT_FAILURE (terminate on error)
#include <stdio.h>
#include <stdlib.h>

// =========================================================================
// cudaCheck
// =========================================================================
inline void cudaCheck(cudaError_t err,const char* file,int line){
  // Called by the CUDA_CHECK macro after every CUDA call.
  //
  //   err   error code returned by the CUDA call
  //   file  source file where the call was made (passed as __FILE__)
  //   line  line number where the call was made (passed as __LINE__)
  //
  // If err == cudaSuccess, does nothing.
  // If err != cudaSuccess, prints the file, line, and CUDA's human-readable
  // error message to stderr, then terminates the program.
  if(err!=cudaSuccess){
    fprintf(stderr,"CUDA Error at %s:%d: %s\n",file,line,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// =========================================================================
// cudaCheckLast
// =========================================================================
inline void cudaCheckLast(const char* file,int line){
  // Called by the KERNEL_CHECK macro after launching a kernel.
  //
  //   file  source file where the launch was made (passed as __FILE__)
  //   line  line number where the launch was made (passed as __LINE__)
  //
  // Kernel launches don't return an error code directly. Errors from a
  // launch show up on the next CUDA call, so this function retrieves the
  // last error with cudaGetLastError() and forwards it to cudaCheck,
  // which does the actual printing and exiting.
  cudaCheck(cudaGetLastError(),file,line);
}

// =========================================================================
// cudaCheckSync
// =========================================================================
inline void cudaCheckSync(const char* file,int line){
  // Called by the SYNC_CHECK macro after cudaDeviceSynchronize().
  //
  //   file  source file where the sync was made (passed as __FILE__)
  //   line  line number where the sync was made (passed as __LINE__)
  //
  // cudaDeviceSynchronize() blocks until all previously queued GPU work
  // finishes and returns any error that occurred during execution.
  // Its return value is forwarded to cudaCheck, which does the actual
  // printing and exiting.
  cudaCheck(cudaDeviceSynchronize(),file,line);
}

// Check the result of a CUDA call
#define CUDA_CHECK(call)  cudaCheck((call), __FILE__, __LINE__)

// Check for errors after launching a kernel
#define KERNEL_CHECK()    cudaCheckLast(__FILE__, __LINE__)

// Check for errors after a cudaDeviceSynchronize
#define SYNC_CHECK()      cudaCheckSync(__FILE__, __LINE__)

#endif
