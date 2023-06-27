#ifndef KERNEL_FLOAT_PRELUDE_H
#define KERNEL_FLOAT_PRELUDE_H

#include "tensor.h"

namespace kernel_float {
namespace prelude {

template<typename T>
using kscalar = tensor<T, extents<>>;

template<typename T, size_t N>
using kvec = tensor<T, extents<N>>;

template<typename T, size_t N, size_t M>
using kmat = tensor<T, extents<N, M>>;

template<typename T, size_t... Ns>
using ktensor = tensor<T, extents<Ns...>>;

// clang-format off
template<typename T> using kvec1 = kvec<T, 1>;
template<typename T> using kvec2 = kvec<T, 2>;
template<typename T> using kvec3 = kvec<T, 3>;
template<typename T> using kvec4 = kvec<T, 4>;
template<typename T> using kvec5 = kvec<T, 5>;
template<typename T> using kvec6 = kvec<T, 6>;
template<typename T> using kvec7 = kvec<T, 7>;
template<typename T> using kvec8 = kvec<T, 8>;
// clang-format on

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T) \
    using k##NAME = scalar<T>;           \
    template<size_t N>                   \
    using v##NAME = vec<T, N>;           \
    using v##NAME##1 = vec<T, 1>;        \
    using v##NAME##2 = vec<T, 2>;        \
    using v##NAME##3 = vec<T, 3>;        \
    using v##NAME##4 = vec<T, 4>;        \
    using v##NAME##5 = vec<T, 5>;        \
    using v##NAME##6 = vec<T, 6>;        \
    using v##NAME##7 = vec<T, 7>;        \
    using v##NAME##8 = vec<T, 8>;

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
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(bfloat16, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bf16, __nv_bfloat16)
#endif

template<size_t... Ns>
static constexpr extents<Ns...> kshape = {};

template<typename... Args>
KERNEL_FLOAT_INLINE kvec<promote_t<Args...>, sizeof...(Args)> make_kvec(Args&&... args) {
    return make_vec(std::forward<Args>(args)...);
};
}  // namespace prelude
}  // namespace kernel_float

#endif