#ifndef KERNEL_FLOAT_TYPES_H
#define KERNEL_FLOAT_TYPES_H

#include "all.h"

namespace kernel_float {

template<typename T>
using vec2 = vec<T, 2>;
template<typename T>
using vec3 = vec<T, 3>;
template<typename T>
using vec4 = vec<T, 4>;
template<typename T>
using vec5 = vec<T, 5>;
template<typename T>
using vec6 = vec<T, 6>;
template<typename T>
using vec7 = vec<T, 7>;
template<typename T>
using vec8 = vec<T, 8>;

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T) \
    template<size_t N>                   \
    using NAME##X = vec<T, N>;           \
    using NAME##1 = vec<T, 1>;           \
    using NAME##2 = vec<T, 2>;           \
    using NAME##3 = vec<T, 3>;           \
    using NAME##4 = vec<T, 4>;           \
    using NAME##5 = vec<T, 5>;           \
    using NAME##6 = vec<T, 6>;           \
    using NAME##7 = vec<T, 7>;           \
    using NAME##8 = vec<T, 8>;

KERNEL_FLOAT_TYPE_ALIAS(char, char)
KERNEL_FLOAT_TYPE_ALIAS(short, short)
KERNEL_FLOAT_TYPE_ALIAS(int, int)
KERNEL_FLOAT_TYPE_ALIAS(long, long)
KERNEL_FLOAT_TYPE_ALIAS(longlong, long long)

KERNEL_FLOAT_TYPE_ALIAS(uchar, unsigned char)
KERNEL_FLOAT_TYPE_ALIAS(ushort, unsigned short)
KERNEL_FLOAT_TYPE_ALIAS(uint, unsigned int)
KERNEL_FLOAT_TYPE_ALIAS(ulong, unsigned long)
KERNEL_FLOAT_TYPE_ALIAS(ulonglong, unsigned long long)

KERNEL_FLOAT_TYPE_ALIAS(float, float)
KERNEL_FLOAT_TYPE_ALIAS(f32x, float)
KERNEL_FLOAT_TYPE_ALIAS(float32x, float)

KERNEL_FLOAT_TYPE_ALIAS(double, double)
KERNEL_FLOAT_TYPE_ALIAS(f64x, double)
KERNEL_FLOAT_TYPE_ALIAS(float64x, double)

#if KERNEL_FLOAT_FP16_AVAILABLE
using float16 = half;
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
using bfloat16 = __nv_bfloat16;
KERNEL_FLOAT_TYPE_ALIAS(bfloat16x, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bf16x, __nv_bfloat16)
#endif
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TYPES_H
