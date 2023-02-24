#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __CUDACC__
#define KERNEL_FLOAT_CUDA (1)

#ifdef __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE      __forceinline__ __device__
#define KERNEL_FLOAT_CUDA_DEVICE (1)
#define KERNEL_FLOAT_CUDA_HOST   (0)
#define KERNEL_FLOAT_CUDA_ARCH   (__CUDA_ARCH__)
#else
#define KERNEL_FLOAT_INLINE      __forceinline__ __host__
#define KERNEL_FLOAT_CUDA_DEVICE (0)
#define KERNEL_FLOAT_CUDA_HOST   (1)
#define KERNEL_FLOAT_CUDA_ARCH   (0)
#endif
#else
#define KERNEL_FLOAT_INLINE      inline
#define KERNEL_FLOAT_CUDA        (0)
#define KERNEL_FLOAT_CUDA_HOST   (1)
#define KERNEL_FLOAT_CUDA_DEVICE (0)
#define KERNEL_FLOAT_CUDA_ARCH   (0)
#endif

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
#define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
#define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
#define KERNEL_FLOAT_FP8_AVAILABLE (0)
#endif

#endif  //KERNEL_FLOAT_MACROS_H
