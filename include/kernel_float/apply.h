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

/**
 * The accurate_policy is designed for computations where maximum accuracy is essential. This policy ensures that all
 * operations are performed without any approximations or optimizations that could potentially alter the precise
 * outcome of the computations
 */
struct accurate_policy {};

/**
 * The fast_policy is intended for scenarios where performance and execution speed are more critical than achieving
 * the utmost accuracy. This policy leverages optimizations to accelerate computations, which may involve
 * approximations that slightly compromise precision.
 */
struct fast_policy {};

/**
 * This template policy allows developers to specify a custom degree of approximation for their computations. By
 * adjusting the `Level` parameter, you can fine-tune the balance between accuracy and performance to meet the
 * specific needs of your application. Higher values mean more precision.
 */
template<int Level = -1>
struct approx_level_policy {};

/**
 * The approximate_policy serves as the default approximation policy, providing a standard level of approximation
 * without requiring explicit configuration. It balances accuracy and performance, making it suitable for
 * general-purpose use cases where neither extreme precision nor maximum speed is necessary.
 */
using approx_policy = approx_level_policy<>;

#ifndef KERNEL_FLOAT_POLICY
#define KERNEL_FLOAT_POLICY accurate_policy;
#endif

/**
 * The `default_policy` acts as the standard computation policy. It can be configured externally using the
 * `KERNEL_FLOAT_POLICY` macro. If `KERNEL_FLOAT_POLICY` is not defined, it defaults to `accurate_policy`.
 */
using default_policy = KERNEL_FLOAT_POLICY;

namespace detail {

//
template<typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_fallback_impl {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
        static_assert(N > 0, "operation not implemented");
    }
};

template<typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_base_impl: apply_fallback_impl<Policy, F, N, Output, Args...> {};

template<typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_impl: apply_base_impl<Policy, F, N, Output, Args...> {};

// `fast_policy` falls back to `accurate_policy`
template<typename F, size_t N, typename Output, typename... Args>
struct apply_fallback_impl<fast_policy, F, N, Output, Args...>:
    apply_impl<accurate_policy, F, N, Output, Args...> {};

// `approx_policy` falls back to `fast_policy`
template<typename F, size_t N, typename Output, typename... Args>
struct apply_fallback_impl<approx_policy, F, N, Output, Args...>:
    apply_impl<fast_policy, F, N, Output, Args...> {};

// `approx_level_policy` falls back to `approx_policy`
template<int Level, typename F, size_t N, typename Output, typename... Args>
struct apply_fallback_impl<approx_level_policy<Level>, F, N, Output, Args...>:
    apply_impl<approx_policy, F, N, Output, Args...> {};

template<typename F, typename Output, typename... Args>
struct invoke_impl {
    KERNEL_FLOAT_INLINE static Output call(F fun, Args... args) {
        return fun(args...);
    }
};

// Only for `accurate_policy` do we implement `apply_impl`, the others will fall back to `apply_base_impl`.
template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl<accurate_policy, F, N, Output, Args...> {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
#pragma unroll
        for (size_t i = 0; i < N; i++) {
            output[i] = invoke_impl<F, Output, Args...>::call(fun, args[i]...);
        }
    }
};

template<typename Policy, typename F, size_t N, typename Output, typename... Args>
struct map_impl {
    static constexpr size_t packet_size = preferred_vector_size<Output>::value;
    static constexpr size_t remainder = N % packet_size;

    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
        if constexpr (N / packet_size > 0) {
#pragma unroll
            for (size_t i = 0; i < N - remainder; i += packet_size) {
                apply_impl<Policy, F, packet_size, Output, Args...>::call(
                    fun,
                    output + i,
                    (args + i)...);
            }
        }

        if constexpr (remainder > 0) {
#pragma unroll
            for (size_t i = N - remainder; i < N; i++) {
                apply_impl<Policy, F, 1, Output, Args...>::call(fun, output + i, (args + i)...);
            }
        }
    }
};

template<typename F, size_t N, typename Output, typename... Args>
using default_map_impl = map_impl<default_policy, F, N, Output, Args...>;

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

    detail::map_impl<Accuracy, F, extent_size<E>, Output, vector_value_type<Args>...>::call(
        fun,
        result.data(),
        (detail::broadcast_impl<vector_value_type<Args>, vector_extent_type<Args>, E>::call(
             into_vector_storage(args))
             .data())...);

    return result;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_APPLY_H