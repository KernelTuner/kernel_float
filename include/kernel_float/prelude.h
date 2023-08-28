#ifndef KERNEL_FLOAT_PRELUDE_H
#define KERNEL_FLOAT_PRELUDE_H

#include "bf16.h"
#include "constant.h"
#include "fp16.h"
#include "vector.h"

namespace kernel_float {
namespace prelude {
namespace kf = ::kernel_float;

template<typename T>
using kscalar = vector<T, extent<1>>;

template<typename T, size_t N>
using kvec = vector<T, extent<N>>;

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

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)  \
    template<size_t N>                    \
    using k##NAME = vector<T, extent<N>>; \
    using k##NAME##1 = vec<T, 1>;         \
    using k##NAME##2 = vec<T, 2>;         \
    using k##NAME##3 = vec<T, 3>;         \
    using k##NAME##4 = vec<T, 4>;         \
    using k##NAME##5 = vec<T, 5>;         \
    using k##NAME##6 = vec<T, 6>;         \
    using k##NAME##7 = vec<T, 7>;         \
    using k##NAME##8 = vec<T, 8>;

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

template<size_t N>
static constexpr extent<N> kextent = {};

template<typename... Args>
KERNEL_FLOAT_INLINE kvec<promote_t<Args...>, sizeof...(Args)> make_kvec(Args&&... args) {
    return make_vec(std::forward<Args>(args)...);
};

template<typename T = double>
using kconstant = constant<T>;

template<typename T = double>
KERNEL_FLOAT_INLINE constexpr kconstant<T> kconst(T value) {
    return value;
}

KERNEL_FLOAT_INLINE
static constexpr kconstant<double> operator""_c(long double v) {
    return static_cast<double>(v);
}

KERNEL_FLOAT_INLINE
static constexpr kconstant<long long int> operator""_c(unsigned long long int v) {
    return static_cast<long long int>(v);
}

// Deduction guides for aliases are only supported from C++20
#if defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201907L
template<typename T>
kscalar(T&&) -> kscalar<decay_t<T>>;

template<typename... Args>
kvec(Args&&...) -> kvec<promote_t<Args...>, sizeof...(Args)>;

template<typename T>
kconstant(T&&) -> kconstant<decay_t<T>>;
#endif

}  // namespace prelude
}  // namespace kernel_float

#endif
