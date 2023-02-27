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
        return Output {fun(left.get(const_index<Is> {}), right.get(const_index<Is> {}))...};
    }
};

template<typename F, typename T, typename L, typename R, size_t N>
struct zip_helper<F, vector_compound<T, N>, vector_compound<L, N>, vector_compound<R, N>> {
    KERNEL_FLOAT_INLINE static vector_compound<T, N>
    call(F fun, const vector_compound<L, N>& left, const vector_compound<R, N>& right) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return {
            zip_helper<
                F,
                vector_storage<T, low_size>,
                vector_storage<L, low_size>,
                vector_storage<R, low_size>>::call(fun, left.low(), right.low()),
            zip_helper<
                F,
                vector_storage<T, high_size>,
                vector_storage<L, high_size>,
                vector_storage<R, high_size>>::call(fun, left.high(), right.high())};
    }
};
};  // namespace detail

template<typename... Ts>
using common_vector_value_type = common_t<vector_value_type<Ts>...>;

template<typename... Ts>
static constexpr size_t common_vector_size = common_size<vector_size<Ts>...>;

template<typename F, typename L, typename R>
using zip_type = vector_storage<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    common_vector_size<L, R>>;

template<typename F, typename Left, typename Right, typename Output = zip_type<F, Left, Right>>
KERNEL_FLOAT_INLINE Output zip(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    return detail::zip_helper<F, Output, into_vector_type<Left>, into_vector_type<Right>>::call(
        fun,
        broadcast<N>(std::forward<Left>(left)),
        broadcast<N>(std::forward<Right>(right)));
}

template<typename F, typename L, typename R>
using zip_common_type = vector_storage<
    result_t<F, common_vector_value_type<L, R>, common_vector_value_type<L, R>>,
    common_vector_size<L, R>>;

template<
    typename F,
    typename Left,
    typename Right,
    typename Output = zip_common_type<F, Left, Right>>
KERNEL_FLOAT_INLINE Output zip_common(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using C = common_t<vector_value_type<Left>, vector_value_type<Right>>;

    return detail::zip_helper<F, Output, vector_storage<C, N>, vector_storage<C, N>>::call(
        fun,
        broadcast<C, N>(std::forward<Left>(left)),
        broadcast<C, N>(std::forward<Right>(right)));
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
    template<typename L, typename R, typename C = common_vector_value_type<L, R>>          \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {    \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                                \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                                            \
    template<                                                                                  \
        typename L,                                                                            \
        typename R,                                                                            \
        typename C = enabled_t<is_vector<L> || is_vector<R>, common_vector_value_type<L, R>>>  \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> operator OP(L&& left, R&& right) { \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right));     \
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
