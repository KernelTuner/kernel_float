#ifndef KERNEL_FLOAT_APPLY_H
#define KERNEL_FLOAT_APPLY_H

#include "base.h"

namespace kernel_float {
namespace detail {

template<typename... Es>
struct broadcast_extent_helper;

template<typename E>
struct broadcast_extent_helper<E> {
    using type = E;
};

template<size_t N>
struct broadcast_extent_helper<extent<N>, extent<N>> {
    using type = extent<N>;
};

template<size_t N>
struct broadcast_extent_helper<extent<1>, extent<N>> {
    using type = extent<N>;
};

template<size_t N>
struct broadcast_extent_helper<extent<N>, extent<1>> {
    using type = extent<N>;
};

template<>
struct broadcast_extent_helper<extent<1>, extent<1>> {
    using type = extent<1>;
};

template<typename A, typename B, typename C, typename... Rest>
struct broadcast_extent_helper<A, B, C, Rest...>:
    broadcast_extent_helper<typename broadcast_extent_helper<A, B>::type, C, Rest...> {};

}  // namespace detail

template<typename... Es>
using broadcast_extent = typename detail::broadcast_extent_helper<Es...>::type;

template<typename... Vs>
using broadcast_vector_extent_type = broadcast_extent<vector_extent_type<Vs>...>;

template<typename From, typename To>
static constexpr bool is_broadcastable = is_same_type<broadcast_extent<From, To>, To>;

template<typename V, typename To>
static constexpr bool is_vector_broadcastable = is_broadcastable<vector_extent_type<V>, To>;

namespace detail {

template<typename T, typename From, typename To>
struct broadcast_impl;

template<typename T, size_t N>
struct broadcast_impl<T, extent<1>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(const vector_storage<T, 1>& input) {
        vector_storage<T, N> output;
        for (size_t i = 0; i < N; i++) {
            output.data()[i] = input.data()[0];
        }
        return output;
    }
};

template<typename T, size_t N>
struct broadcast_impl<T, extent<N>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(vector_storage<T, N> input) {
        return input;
    }
};

template<typename T>
struct broadcast_impl<T, extent<1>, extent<1>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, 1> call(vector_storage<T, 1> input) {
        return input;
    }
};

}  // namespace detail

/**
 * Takes the given vector `input` and extends its size to a length of `N`. This is only valid if the size of `input`
 * is 1 or `N`.
 *
 * Example
 * =======
 * ```
 * vec<float, 1> a = {1.0f};
 * vec<float, 5> x = broadcast<5>(a);  // Returns [1.0f, 1.0f, 1.0f, 1.0f, 1.0f]
 *
 * vec<float, 5> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
 * vec<float, 5> y = broadcast<5>(b);  // Returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f]
 * ```
 */
template<size_t N, typename V>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, extent<N>>
broadcast(const V& input, extent<N> new_size = {}) {
    using T = vector_value_type<V>;
    return detail::broadcast_impl<T, vector_extent_type<V>, extent<N>>::call(
        into_vector_storage(input));
}

/**
 * Takes the given vector `input` and extends its size to the same length as vector `other`. This is only valid if the
 * size of `input` is 1 or the same as `other`.
 */
template<typename V, typename R>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<R>>
broadcast_like(const V& input, const R& other) {
    return broadcast(input, vector_extent_type<R> {});
}

namespace detail {

template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
#pragma unroll
        for (size_t i = 0; i < N; i++) {
            output[i] = fun(args[i]...);
        }
    }
};

template<typename F, size_t N, typename Output, typename... Args>
struct apply_fastmath_impl: apply_impl<F, N, Output, Args...> {};

template<int Deg, typename F, size_t N, typename Output, typename... Args>
struct apply_approx_impl: apply_fastmath_impl<F, N, Output, Args...> {};
}  // namespace detail

struct accurate_policy {
    template<typename F, size_t N, typename Output, typename... Args>
    using type = detail::apply_impl<F, N, Output, Args...>;
};

struct fast_policy {
    template<typename F, size_t N, typename Output, typename... Args>
    using type = detail::apply_fastmath_impl<F, N, Output, Args...>;
};

template<int Degree = -1>
struct approximate_policy {
    template<typename F, size_t N, typename Output, typename... Args>
    using type = detail::apply_approx_impl<Degree, F, N, Output, Args...>;
};

using default_approximate_policy = approximate_policy<>;

#ifdef KERNEL_FLOAT_POLICY
using default_policy = KERNEL_FLOAT_POLICY;
#else
using default_policy = accurate_policy;
#endif

namespace detail {

template<typename Policy, typename F, size_t N, typename Output, typename... Args>
struct map_policy_impl {
    static constexpr size_t packet_size = preferred_vector_size<Output>::value;
    static constexpr size_t remainder = N % packet_size;

    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
        if constexpr (N / packet_size > 0) {
#pragma unroll
            for (size_t i = 0; i < N - remainder; i += packet_size) {
                Policy::template type<F, packet_size, Output, Args...>::call(
                    fun,
                    output + i,
                    (args + i)...);
            }
        }

        if constexpr (remainder > 0) {
#pragma unroll
            for (size_t i = N - remainder; i < N; i++) {
                Policy::template type<F, 1, Output, Args...>::call(fun, output + i, (args + i)...);
            }
        }
    }
};

template<typename F, size_t N, typename Output, typename... Args>
using map_impl = map_policy_impl<default_policy, F, N, Output, Args...>;

}  // namespace detail

template<typename F, typename... Args>
using map_type =
    vector<result_t<F, vector_value_type<Args>...>, broadcast_vector_extent_type<Args...>>;

/**
 * Apply the function `F` to each element from the vector `input` and return the results as a new vector.
 *
 * Examples
 * ========
 * ```
 * vec<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * vec<float, 4> squared = map([](auto x) { return x * x; }, input); // [1.0f, 4.0f, 9.0f, 16.0f]
 * ```
 */
template<typename Accuracy = default_policy, typename F, typename... Args>
KERNEL_FLOAT_INLINE map_type<F, Args...> map(F fun, const Args&... args) {
    using Output = result_t<F, vector_value_type<Args>...>;
    using E = broadcast_vector_extent_type<Args...>;
    vector_storage<Output, extent_size<E>> result;

    detail::map_policy_impl<Accuracy, F, extent_size<E>, Output, vector_value_type<Args>...>::call(
        fun,
        result.data(),
        (detail::broadcast_impl<vector_value_type<Args>, vector_extent_type<Args>, E>::call(
             into_vector_storage(args))
             .data())...);

    return result;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_APPLY_H