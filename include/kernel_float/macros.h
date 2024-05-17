#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __CUDACC__
#define KERNEL_FLOAT_CUDA (1)

#ifdef __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE    __forceinline__ __device__
#define KERNEL_FLOAT_IS_DEVICE (1)
#define KERNEL_FLOAT_IS_HOST   (0)
#define KERNEL_FLOAT_CUDA_ARCH (__CUDA_ARCH__)
#else  // __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE    __forceinline__ __host__
#define KERNEL_FLOAT_IS_DEVICE (0)
#define KERNEL_FLOAT_IS_HOST   (1)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif  // __CUDA_ARCH__
#else  // __CUDACC__
#define KERNEL_FLOAT_INLINE    inline
#define KERNEL_FLOAT_CUDA      (0)
#define KERNEL_FLOAT_IS_HOST   (1)
#define KERNEL_FLOAT_IS_DEVICE (0)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif  // __CUDACC__

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
#define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
#define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
#ifdef __CUDACC_VER_MAJOR__
#define KERNEL_FLOAT_FP8_AVAILABLE (__CUDACC_VER_MAJOR__ >= 12)
#else  // __CUDACC_VER_MAJOR__
#define KERNEL_FLOAT_FP8_AVAILABLE (0)
#endif  // __CUDACC_VER_MAJOR__
#endif  // KERNEL_FLOAT_FP8_AVAILABLE

#define KERNEL_FLOAT_ASSERT(expr) \
    do {                          \
    } while (0)
#define KERNEL_FLOAT_UNREACHABLE __builtin_unreachable()

// Somet utility macros
#define KERNEL_FLOAT_CONCAT_IMPL(A, B) A##B
#define KERNEL_FLOAT_CONCAT(A, B)      KERNEL_FLOAT_CONCAT_IMPL(A, B)
#define KERNEL_FLOAT_CALL(F, ...)      F(__VA_ARGS__)

// TOOD: check if this way is support across all compilers
#if defined(__has_builtin) && 0  // Seems that `__builtin_assume_aligned` leads to segfaults
#if __has_builtin(__builtin_assume_aligned)
#define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) static_cast <TYPE*>(
        __builtin_assume_aligned(static_cast <TYPE*>(PTR), (ALIGNMENT)))
#else
#define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) (PTR)
#endif
#else
#define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) (PTR)
#endif

#define KERNEL_FLOAT_MAX_ALIGNMENT (32)

#ifndef KERNEL_FLOAT_FAST_MATH
#define KERNEL_FLOAT_FAST_MATH (0)
#endif

#endif  //KERNEL_FLOAT_MACROS_H
