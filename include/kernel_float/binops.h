#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H

#include "unops.h"

namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Left, typename Right, typename = void>
struct zip_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Left& left, const Right& right) {
        return call_with_indices(fun, left, right, make_index_sequence<vector_size<Output>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output
    call_with_indices(F fun, const Left& left, const Right& right, index_sequence<Is...> = {}) {
        return vector_traits<Output>::create(fun(vector_get<Is>(left), vector_get<Is>(right))...);
    }
};

template<typename F, typename V, size_t N>
struct zip_helper<F, nested_array<V, N>, nested_array<V, N>, nested_array<V, N>> {
    KERNEL_FLOAT_INLINE static nested_array<V, N>
    call(F fun, const nested_array<V, N>& left, const nested_array<V, N>& right) {
        return call(fun, left, right, make_index_sequence<nested_array<V, N>::num_packets> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static nested_array<V, N> call(
        F fun,
        const nested_array<V, N>& left,
        const nested_array<V, N>& right,
        index_sequence<Is...>) {
        return {zip_helper<F, V, V, V>::call(fun, left[Is], right[Is])...};
    }
};
};  // namespace detail

template<typename... Ts>
using common_vector_value_type = common_t<vector_value_type<Ts>...>;

template<typename... Ts>
static constexpr size_t common_vector_size = common_size<vector_size<Ts>...>;

template<typename F, typename L, typename R>
using zip_type = default_storage_type<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    common_vector_size<L, R>>;

/**
 * Applies ``fun`` to each pair of two elements from ``left`` and ``right`` and returns a new
 * vector with the results.
 *
 * If ``left`` and ``right`` are not the same size, they will first be broadcast into a
 * common size using ``resize``.
 *
 * Note that this function does **not** cast the input vectors to a common element type. See
 * ``zip_common`` for that functionality.
 */
template<typename F, typename Left, typename Right, typename Output = zip_type<F, Left, Right>>
KERNEL_FLOAT_INLINE vector<Output> zip(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using LeftInput = default_storage_type<vector_value_type<Left>, N>;
    using RightInput = default_storage_type<vector_value_type<Right>, N>;

    return detail::zip_helper<F, Output, LeftInput, RightInput>::call(
        fun,
        broadcast<LeftInput, Left>(std::forward<Left>(left)),
        broadcast<RightInput, Right>(std::forward<Right>(right)));
}

template<typename F, typename L, typename R>
using zip_common_type = default_storage_type<
    result_t<F, common_vector_value_type<L, R>, common_vector_value_type<L, R>>,
    common_vector_size<L, R>>;

/**
 * Applies ``fun`` to each pair of two elements from ``left`` and ``right`` and returns a new
 * vector with the results.
 *
 * If ``left`` and ``right`` are not the same size, they will first be broadcast into a
 * common size using ``resize``.
 *
 * If ``left`` and ``right`` are not of the same type, they will first be case into a common
 * data type. For example, zipping ``float`` and ``double`` first cast vectors to ``double``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {1, 2, 3, 4};
 * vec<long, 1> = {8};
 * vec<long, 5> = zip_common([](auto a, auto b){ return a + b; }, x, y); // [9, 10, 11, 12]
 * ```
 */
template<
    typename F,
    typename Left,
    typename Right,
    typename Output = zip_common_type<F, Left, Right>>
KERNEL_FLOAT_INLINE vector<Output> zip_common(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using C = common_t<vector_value_type<Left>, vector_value_type<Right>>;
    using Input = default_storage_type<C, N>;

    return detail::zip_helper<F, Output, Input, Input>::call(
        fun,
        broadcast<Input, Left>(std::forward<Left>(left)),
        broadcast<Input, Right>(std::forward<Right>(right)));
}

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR)                                                  \
    namespace ops {                                                                             \
    template<typename T>                                                                        \
    struct NAME {                                                                               \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                                     \
            return T(EXPR);                                                                     \
        }                                                                                       \
    };                                                                                          \
    }                                                                                           \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>>               \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> NAME(L&& left, R&& right) { \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right));      \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                               \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const vector<L>& left,                                                    \
        const vector<R>& right) {                                                 \
        return zip_common(ops::NAME<C> {}, left, right);                          \
    }                                                                             \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const vector<L>& left,                                                    \
        const R& right) {                                                         \
        return zip_common(ops::NAME<C> {}, left, right);                          \
    }                                                                             \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const L& left,                                                            \
        const vector<R>& right) {                                                 \
        return zip_common(ops::NAME<C> {}, left, right);                          \
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
template<template<typename T> typename F, typename L, typename R>
static constexpr bool vector_assign_allowed =
    common_vector_size<L, R> == vector_size<L> &&
    is_implicit_convertible<
        result_t<
                F<common_t<vector_value_type<L>, vector_value_type<R>>>,
                vector_value_type<L>,
                vector_value_type<R>
        >,
        vector_value_type<L>
    >;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                                        \
    template<                                                                                 \
        typename L,                                                                           \
        typename R,                                                                           \
        typename T = enabled_t<vector_assign_allowed<ops::NAME, L, R>, vector_value_type<L>>> \
    KERNEL_FLOAT_INLINE vector<L>& operator OP(vector<L>& lhs, const R& rhs) {                \
        using F = ops::NAME<T>;                                                               \
        lhs = zip_common<F, const L&, const R&, L>(F {}, lhs.storage(), rhs);                 \
        return lhs;                                                                           \
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

#endif  //KERNEL_FLOAT_BINOPS_H
