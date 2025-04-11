#pragma once

/**
 * This header file provides a mapping from CUDA-specific function names and types to their equivalent HIP
 * counterparts, allowing for cross-platform development between CUDA and HIP. By including this header, code
 * originally written for CUDA can be compiled with the HIP compiler (hipcc) by automatically replacing CUDA API
 * calls with their HIP equivalents.
 */
#ifdef __HIPCC__
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaGetLastError       hipGetLastError
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDefault      hipMemcpyDefault
#define cudaMemset             hipMemset
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize
#endif