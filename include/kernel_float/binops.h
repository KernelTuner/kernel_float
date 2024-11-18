#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H

#include "conversion.h"
#include "unops.h"

namespace kernel_float {

template<typename F, typename L, typename R>
using zip_type = map_type<F, L, R>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements.
 *
 * Example
 * =======
 * ```
 * vec<bool, 3> make_negative = {true, false, true};
 * vec<int, 3> input = {1, 2, 3};
 * vec<int, 3> output = zip([](bool b, int n){ return b ? -n : +n; }, make_negative, input); // returns [-1, 2, -3]
 * ```
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_type<F, L, R> zip(F fun, const L& left, const R& right) {
    return ::kernel_float::map(fun, left, right);
}

template<typename F, typename L, typename R>
using zip_common_type = vector<
    result_t<F, promoted_vector_value_type<L, R>, promoted_vector_value_type<L, R>>,
    broadcast_vector_extent_type<L, R>>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements. The elements are promoted to a common type before applying
 * the binary function.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> a = {1.0f, 2.0f, 3.0f};
 * vec<int, 3> b = {4, 5, 6};
 * vec<float, 3> c = zip_common([](float x, float y){ return x + y; }, a, b); // returns [5.0f, 7.0f, 9.0f]
 * ```
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_common_type<F, L, R> zip_common(F fun, const L& left, const R& right) {
    using T = promoted_vector_value_type<L, R>;
    using O = result_t<F, T, T>;
    using E = broadcast_vector_extent_type<L, R>;

    vector_storage<O, extent_size<E>> result;

    detail::default_map_impl<F, extent_size<E>, O, T, T>::call(
        fun,
        result.data(),
        detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(left))
            .data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(right))
            .data());

    return result;
}

#define KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)                                                 \
    template<typename L, typename R, typename C = promoted_vector_value_type<L, R>>          \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {      \
        return zip_common(ops::NAME<C> {}, static_cast<L&&>(left), static_cast<R&&>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR, EXPR_F64, EXPR_F32)         \
    namespace ops {                                                        \
    template<typename T, typename = void>                                  \
    struct NAME {                                                          \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                \
            return ops::cast<decltype(EXPR), T> {}(EXPR);                  \
        }                                                                  \
    };                                                                     \
                                                                           \
    template<>                                                             \
    struct NAME<double> {                                                  \
        KERNEL_FLOAT_INLINE double operator()(double left, double right) { \
            return ops::cast<decltype(EXPR_F64), double> {}(EXPR_F64);     \
        }                                                                  \
    };                                                                     \
                                                                           \
    template<typename T>                                                   \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> {  \
        KERNEL_FLOAT_INLINE T operator()(T left_, T right_) {              \
            float left = ops::cast<T, float> {}(left_);                    \
            float right = ops::cast<T, float> {}(right_);                  \
            return ops::cast<decltype(EXPR_F32), T> {}(EXPR_F32);          \
        }                                                                  \
    };                                                                     \
    }                                                                      \
                                                                           \
    KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)

#define KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(NAME, OP, EXPR_F64, EXPR_F32)                      \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right, EXPR_F64, EXPR_F32)                           \
                                                                                                  \
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

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP) \
    KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(NAME, OP, left OP right, left OP right)

KERNEL_FLOAT_DEFINE_BINARY_OP(add, +)
KERNEL_FLOAT_DEFINE_BINARY_OP(subtract, -)
KERNEL_FLOAT_DEFINE_BINARY_OP(divide, /)
KERNEL_FLOAT_DEFINE_BINARY_OP(multiply, *)
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(modulo, %, ::fmod(left, right), ::fmodf(left, right))

KERNEL_FLOAT_DEFINE_BINARY_OP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINARY_OP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINARY_OP(less, <)
KERNEL_FLOAT_DEFINE_BINARY_OP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater, >)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater_equal, >=)

// clang-format off
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_and, &, bool(left) && bool(right), bool(left) && bool(right))
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_or, |, bool(left) | bool(right), bool(left) | bool(right))
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_xor, ^, bool(left) ^ bool(right), bool(left) ^ bool(right))
// clang-format on

// clang-format off
template<template<typename, typename=void> typename F, typename T, typename E, typename R>
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
        typename = enable_if_t<is_vector_assign_allowed<ops::NAME, T, E, R>>>        \
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

namespace ops {
template<typename T>
struct min {
    KERNEL_FLOAT_INLINE T operator()(T left, T right) {
        return left < right ? left : right;
    }
};

template<typename T>
struct max {
    KERNEL_FLOAT_INLINE T operator()(T left, T right) {
        return left > right ? left : right;
    }
};

template<>
struct min<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return ::fmin(left, right);
    }
};

template<>
struct max<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return ::fmax(left, right);
    }
};

template<>
struct min<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return ::fminf(left, right);
    }
};

template<>
struct max<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return ::fmaxf(left, right);
    }
};
}  // namespace ops

KERNEL_FLOAT_DEFINE_BINARY_FUN(min)
KERNEL_FLOAT_DEFINE_BINARY_FUN(max)

#define KERNEL_FLOAT_DEFINE_BINARY_MATH(NAME)                              \
    namespace ops {                                                        \
    template<typename T, typename = void>                                  \
    struct NAME;                                                           \
                                                                           \
    template<typename T>                                                   \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> {  \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                \
            return T(::NAME(left, right));                                 \
        }                                                                  \
    };                                                                     \
                                                                           \
    template<>                                                             \
    struct NAME<double> {                                                  \
        KERNEL_FLOAT_INLINE double operator()(double left, double right) { \
            return double(::NAME(left, right));                            \
        }                                                                  \
    };                                                                     \
    }                                                                      \
                                                                           \
    KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)

KERNEL_FLOAT_DEFINE_BINARY_MATH(copysign)
KERNEL_FLOAT_DEFINE_BINARY_MATH(fmod)
KERNEL_FLOAT_DEFINE_BINARY_MATH(nextafter)
KERNEL_FLOAT_DEFINE_BINARY_MATH(pow)
KERNEL_FLOAT_DEFINE_BINARY_MATH(remainder)

KERNEL_FLOAT_DEFINE_BINARY(
    hypot,
    ops::sqrt<T>()(left* left + right * right),
    ::hypot(left, right),
    ::hypotf(left, right))

#if KERNEL_FLOAT_IS_DEVICE
KERNEL_FLOAT_DEFINE_BINARY(
    rhypot,
    (ops::rcp<T>(ops::hypot<T>()(left, right))),
    ::rhypot(left, right),
    ::rhypotf(left, right))
#else
KERNEL_FLOAT_DEFINE_BINARY(
    rhypot,
    (ops::rcp<T>(ops::hypot<T>()(left, right))),
    (double(1) / ::hypot(left, right)),
    (float(1) / ::hypotf(left, right)))
#endif

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
};  // namespace ops

namespace detail {
template<typename Policy, typename T, size_t N>
struct apply_impl<Policy, ops::divide<T>, N, T, T, T> {
    KERNEL_FLOAT_INLINE static void
    call(ops::divide<T> fun, T* result, const T* lhs, const T* rhs) {
        T rhs_rcp[N];

        // Fast way to perform division is to multiply by the reciprocal
        apply_impl<Policy, ops::rcp<T>, N, T, T>::call({}, rhs_rcp, rhs);
        apply_impl<Policy, ops::multiply<T>, N, T, T, T>::call({}, result, lhs, rhs_rcp);
    }
};

template<typename T, size_t N>
struct apply_impl<accurate_policy, ops::divide<T>, N, T, T, T>:
    apply_base_impl<accurate_policy, ops::divide<T>, N, T, T, T> {};

#if KERNEL_FLOAT_IS_DEVICE
template<>
struct apply_impl<fast_policy, ops::divide<float>, 1, float, float, float> {
    KERNEL_FLOAT_INLINE static void
    call(ops::divide<float> fun, float* result, const float* lhs, const float* rhs) {
        *result = __fdividef(*lhs, *rhs);
    }
};
#endif
}  // namespace detail

template<typename L, typename R, typename T = promoted_vector_value_type<L, R>>
KERNEL_FLOAT_INLINE zip_common_type<ops::divide<T>, T, T>
fast_divide(const L& left, const R& right) {
    using E = broadcast_vector_extent_type<L, R>;
    vector_storage<T, extent_size<E>> result;

    detail::map_impl<fast_policy, ops::divide<T>, extent_size<E>, T, T, T>::call(
        ops::divide<T> {},
        result.data(),
        detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(left))
            .data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(right))
            .data());

    return result;
}

namespace detail {
template<typename T>
struct cross_impl {
    KERNEL_FLOAT_INLINE
    static vector<T, extent<3>>
    call(const vector_storage<T, 3>& av, const vector_storage<T, 3>& bv) {
        auto a = av.data();
        auto b = bv.data();
        vector<T, extent<6>> v0 = {a[1], a[2], a[0], a[2], a[0], a[1]};
        vector<T, extent<6>> v1 = {b[2], b[0], b[1], b[1], b[2], b[0]};
        vector<T, extent<6>> rv = v0 * v1;

        auto r = rv.data();
        vector<T, extent<3>> r0 = {r[0], r[1], r[2]};
        vector<T, extent<3>> r1 = {r[3], r[4], r[5]};
        return r0 - r1;
    }
};
};  // namespace detail

/**
 * Calculates the cross-product between two vectors of length 3.
 */
template<
    typename L,
    typename R,
    typename T = promoted_vector_value_type<L, R>,
    typename = enable_if_t<(vector_size<L> == 3 && vector_size<R> == 3)>>
KERNEL_FLOAT_INLINE vector<T, extent<3>> cross(const L& left, const R& right) {
    return detail::cross_impl<T>::call(convert_storage<T, 3>(left), convert_storage<T, 3>(right));
}

}  // namespace kernel_float

#endif
