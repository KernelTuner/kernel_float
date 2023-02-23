#ifndef KERNEL_FLOAT_UNARY_H
#define KERNEL_FLOAT_UNARY_H

#include "macros.h"
#include "storage.h"

namespace kernel_float {

template<size_t N, typename Is = make_index_sequence<N>>
struct map_apply_helper;

template<typename F, typename T, size_t N>
struct map_helper {
    using return_type = result_t<F, T>;
    KERNEL_FLOAT_INLINE static vec<return_type, N> call(F fun, const vec<T, N>& input) noexcept {
        return map_apply_helper<N>::call(fun, input);
    }
};

template<size_t N, size_t... Is>
struct map_apply_helper<N, index_sequence<Is...>> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, N> call(F fun, const vec<T, N>& input) noexcept {
        return detail::vec_storage<R, N> {fun(input.get(Is))...};
    }
};

template<>
struct map_apply_helper<2> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 2> call(F fun, const vec<T, 2>& input) noexcept {
        return {fun(input.get(constant_index<0> {})), fun(input.get(constant_index<1> {}))};
    }
};

template<>
struct map_apply_helper<4> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 4> call(F fun, const vec<T, 4>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1> {})),
            map(fun, input.get(index_sequence<1, 2> {}))};
    }
};

template<>
struct map_apply_helper<6> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 6> call(F fun, const vec<T, 6>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map(fun, input.get(index_sequence<4, 5> {}))};
    }
};

template<>
struct map_apply_helper<8> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 8> call(F fun, const vec<T, 8>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map(fun, input.get(index_sequence<4, 5, 6, 7> {}))};
    }
};

template<typename T, size_t N, typename F, typename R = result_t<F, T>>
KERNEL_FLOAT_INLINE vec<R, N> map(const vec<T, N>& input, F fun) noexcept {
    return map_helper<F, T, N>::call(fun, input);
}

namespace ops {
template<typename T, typename R>
struct cast {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

template<typename T>
struct cast<T, T> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};
}  // namespace ops

template<typename T, size_t N>
struct map_helper<ops::cast<T, T>, T, N> {
    KERNEL_FLOAT_INLINE static vec<T, N> call(ops::cast<T, T>, const vec<T, N>& input) noexcept {
        return input;
    }
};

template<typename R, typename T, size_t N>
KERNEL_FLOAT_INLINE vec<R, N> cast(const vec<T, N>& input) noexcept {
    return map(input, ops::cast<T, R> {});
}

#define KERNEL_FLOAT_DEFINE_FUN1_OP(NAME, EXPR)                  \
    namespace ops {                                              \
    template<typename T>                                         \
    struct NAME {                                                \
        KERNEL_FLOAT_INLINE T operator()(T input) {              \
            return EXPR;                                         \
        }                                                        \
    };                                                           \
    }                                                            \
    template<typename T, size_t N>                               \
    KERNEL_FLOAT_INLINE vec<T, N> NAME(const vec<T, N>& input) { \
        return map(input, ops::NAME<T> {});                      \
    }

KERNEL_FLOAT_DEFINE_FUN1_OP(negate, -input)
KERNEL_FLOAT_DEFINE_FUN1_OP(bit_not, ~input)
KERNEL_FLOAT_DEFINE_FUN1_OP(logical_not, !input)

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator-(const vec<T, N>& input) {
    return map(input, ops::negate<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator~(const vec<T, N>& input) {
    return map(input, ops::bit_not<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator!(const vec<T, N>& input) {
    return map(input, ops::logical_not<T> {});
}

#define KERNEL_FLOAT_DEFINE_FUN1(NAME) KERNEL_FLOAT_DEFINE_FUN1_OP(NAME, ::NAME(input))

KERNEL_FLOAT_DEFINE_FUN1(acos)
KERNEL_FLOAT_DEFINE_FUN1(abs)
KERNEL_FLOAT_DEFINE_FUN1(acosh)
KERNEL_FLOAT_DEFINE_FUN1(asin)
KERNEL_FLOAT_DEFINE_FUN1(asinh)
KERNEL_FLOAT_DEFINE_FUN1(atan)
KERNEL_FLOAT_DEFINE_FUN1(atanh)
KERNEL_FLOAT_DEFINE_FUN1(cbrt)
KERNEL_FLOAT_DEFINE_FUN1(ceil)
KERNEL_FLOAT_DEFINE_FUN1(cos)
KERNEL_FLOAT_DEFINE_FUN1(cosh)
KERNEL_FLOAT_DEFINE_FUN1(cospi)
KERNEL_FLOAT_DEFINE_FUN1(erf)
KERNEL_FLOAT_DEFINE_FUN1(erfc)
KERNEL_FLOAT_DEFINE_FUN1(erfcinv)
KERNEL_FLOAT_DEFINE_FUN1(erfcx)
KERNEL_FLOAT_DEFINE_FUN1(erfinv)
KERNEL_FLOAT_DEFINE_FUN1(exp)
KERNEL_FLOAT_DEFINE_FUN1(exp10)
KERNEL_FLOAT_DEFINE_FUN1(exp2)
KERNEL_FLOAT_DEFINE_FUN1(expm1)
KERNEL_FLOAT_DEFINE_FUN1(fabs)
KERNEL_FLOAT_DEFINE_FUN1(floor)
KERNEL_FLOAT_DEFINE_FUN1(ilogb)
KERNEL_FLOAT_DEFINE_FUN1(lgamma)
KERNEL_FLOAT_DEFINE_FUN1(log)
KERNEL_FLOAT_DEFINE_FUN1(log10)
KERNEL_FLOAT_DEFINE_FUN1(logb)
KERNEL_FLOAT_DEFINE_FUN1(nearbyint)
KERNEL_FLOAT_DEFINE_FUN1(normcdf)
KERNEL_FLOAT_DEFINE_FUN1(rcbrt)
KERNEL_FLOAT_DEFINE_FUN1(sin)
KERNEL_FLOAT_DEFINE_FUN1(sinh)
KERNEL_FLOAT_DEFINE_FUN1(sqrt)
KERNEL_FLOAT_DEFINE_FUN1(tan)
KERNEL_FLOAT_DEFINE_FUN1(tanh)
KERNEL_FLOAT_DEFINE_FUN1(tgamma)
KERNEL_FLOAT_DEFINE_FUN1(trunc)
KERNEL_FLOAT_DEFINE_FUN1(y0)
KERNEL_FLOAT_DEFINE_FUN1(y1)
KERNEL_FLOAT_DEFINE_FUN1(yn)
KERNEL_FLOAT_DEFINE_FUN1(rint)
KERNEL_FLOAT_DEFINE_FUN1(rsqrt)
KERNEL_FLOAT_DEFINE_FUN1(round)
KERNEL_FLOAT_DEFINE_FUN1(signbit)
KERNEL_FLOAT_DEFINE_FUN1(isinf)
KERNEL_FLOAT_DEFINE_FUN1(isnan)

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNARY_H
