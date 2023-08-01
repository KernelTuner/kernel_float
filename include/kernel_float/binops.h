#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H

#include "broadcast.h"
#include "unops.h"

namespace kernel_float {

template<typename F, typename L, typename R>
using zip_type = vector<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    broadcast_vector_extent_type<L, R>>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements.
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_type<F, L, R> zip(F fun, const L& left, const R& right) {
    using A = vector_value_type<L>;
    using B = vector_value_type<R>;
    using O = result_t<F, A, B>;
    using E = broadcast_vector_extent_type<L, R>;

    return detail::apply_impl<F, E::value, O, A, B>::call(
        fun,
        broadcast<E>(left).storage(),
        broadcast<E>(right).storage());
}

template<typename F, typename L, typename R>
using zip_common_type = vector<
    result_t<F, promoted_vector_value_type<L, R>, promoted_vector_value_type<L, R>>,
    broadcast_vector_extent_type<L, R>>;

/**
 * Similar to `zip`, except `zip_common` promotes the element types of the inputs to a common type before applying the
 * binary function.
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_common_type<F, L, R> zip_common(F fun, const L& left, const R& right) {
    using T = promoted_vector_value_type<L, R>;
    using O = result_t<F, T, T>;
    using E = broadcast_vector_extent_type<L, R>;

    return detail::apply_impl<F, E::value, O, T, T>::call(
        fun,
        detail::convert_helper<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(left)),
        detail::convert_helper<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(right)));
}

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR)                                             \
    namespace ops {                                                                        \
    template<typename T>                                                                   \
    struct NAME {                                                                          \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                                \
            return T(EXPR);                                                                \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    template<typename L, typename R, typename C = promoted_vector_value_type<L, R>>        \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {    \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                                               \
    template<typename L, typename R, typename C = promote_t<L, R>, typename E1, typename E2>      \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E1>, vector<R, E2>> operator OP(  \
        const vector<L, E1>& left,                                                                \
        const vector<R, E2>& right) {                                                             \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<L, vector_value_type<R>>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E>, R> operator OP(               \
        const vector<L, E>& left,                                                                 \
        const R& right) {                                                                         \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<vector_value_type<L>, R>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, vector<R, E>> operator OP(               \
        const L& left,                                                                            \
        const vector<R, E>& right) {                                                              \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }

KERNEL_FLOAT_DEFINE_BINARY_OP(add, +)
KERNEL_FLOAT_DEFINE_BINARY_OP(subtract, -)
KERNEL_FLOAT_DEFINE_BINARY_OP(divide, /)
KERNEL_FLOAT_DEFINE_BINARY_OP(multiply, *)
KERNEL_FLOAT_DEFINE_BINARY_OP(modulo, %)

KERNEL_FLOAT_DEFINE_BINARY_OP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINARY_OP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINARY_OP(less, <)
KERNEL_FLOAT_DEFINE_BINARY_OP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater, >)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater_equal, >=)

KERNEL_FLOAT_DEFINE_BINARY_OP(bit_and, &)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_or, |)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_xor, ^)

// clang-format off
template<template<typename> typename F, typename T, typename E, typename R>
static constexpr bool is_vector_assign_allowed =
        is_vector_broadcastable<R, E> &&
        is_implicit_convertible<
            result_t<
                F<promote_t<T, vector_value_type<R>>>,
                    T,
                    vector_value_type<R>
                >,
            T
        >;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                               \
    template<                                                                        \
        typename T,                                                                  \
        typename E,                                                                  \
        typename R,                                                                  \
        typename = enabled_t<is_vector_assign_allowed<ops::NAME, T, E, R>>>          \
    KERNEL_FLOAT_INLINE vector<T, E>& operator OP(vector<T, E>& lhs, const R& rhs) { \
        using F = ops::NAME<T>;                                                      \
        lhs = zip_common(F {}, lhs, rhs);                                            \
        return lhs;                                                                  \
    }

KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(add, +=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(subtract, -=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(divide, /=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(multiply, *=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(modulo, %=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_and, &=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_or, |=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_xor, ^=)

#define KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME) KERNEL_FLOAT_DEFINE_BINARY(NAME, ::NAME(left, right))

KERNEL_FLOAT_DEFINE_BINARY_FUN(min)
KERNEL_FLOAT_DEFINE_BINARY_FUN(max)
KERNEL_FLOAT_DEFINE_BINARY_FUN(copysign)
KERNEL_FLOAT_DEFINE_BINARY_FUN(hypot)
KERNEL_FLOAT_DEFINE_BINARY_FUN(modf)
KERNEL_FLOAT_DEFINE_BINARY_FUN(nextafter)
KERNEL_FLOAT_DEFINE_BINARY_FUN(pow)
KERNEL_FLOAT_DEFINE_BINARY_FUN(remainder)

#if KERNEL_FLOAT_CUDA_DEVICE
KERNEL_FLOAT_DEFINE_BINARY_FUN(rhypot)
#endif

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_DEFINE_BINARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN)     \
    KERNEL_FLOAT_DEFINE_BINARY(FUN_NAME, ops::OP_NAME<T> {}(left, right)) \
    namespace ops {                                                       \
    template<>                                                            \
    struct OP_NAME<float> {                                               \
        KERNEL_FLOAT_INLINE float operator()(float left, float right) {   \
            return FLOAT_FUN(left, right);                                \
        }                                                                 \
    };                                                                    \
    }
#else
#define KERNEL_FLOAT_DEFINE_BINARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_BINARY(FUN_NAME, ops::OP_NAME<T> {}(left, right))
#endif

KERNEL_FLOAT_DEFINE_BINARY_FAST(fast_div, divide, __fdividef)
KERNEL_FLOAT_DEFINE_BINARY_FAST(fast_pow, pow, __powf)

namespace ops {
template<>
struct add<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left || right;
    }
};

template<>
struct multiply<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left && right;
    }
};

template<>
struct bit_and<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) && bool(right));
    }
};

template<>
struct bit_or<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) ^ bool(right));
    }
};

template<>
struct bit_and<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) && bool(right));
    }
};

template<>
struct bit_or<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) ^ bool(right));
    }
};
};  // namespace ops

}  // namespace kernel_float

#endif
