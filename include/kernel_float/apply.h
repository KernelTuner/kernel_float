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

template<size_t N>
struct apply_recur_impl;

template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* result, const Args*... inputs) {
        apply_recur_impl<N>::call(fun, result, inputs...);
    }
};

template<size_t N>
struct apply_recur_impl {
    static constexpr size_t K = round_up_to_power_of_two(N) / 2;

    template<typename F, typename Output, typename... Args>
    KERNEL_FLOAT_INLINE static void call(F fun, Output* result, const Args*... inputs) {
        apply_impl<F, K, Output, Args...>::call(fun, result, inputs...);
        apply_impl<F, N - K, Output, Args...>::call(fun, result + K, (inputs + K)...);
    }
};

template<>
struct apply_recur_impl<0> {
    template<typename F, typename Output, typename... Args>
    KERNEL_FLOAT_INLINE static void call(F fun, Output* result, const Args*... inputs) {}
};

template<>
struct apply_recur_impl<1> {
    template<typename F, typename Output, typename... Args>
    KERNEL_FLOAT_INLINE static void call(F fun, Output* result, const Args*... inputs) {
        result[0] = fun(inputs[0]...);
    }
};
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
template<typename F, typename... Args>
KERNEL_FLOAT_INLINE map_type<F, Args...> map(F fun, const Args&... args) {
    using Output = result_t<F, vector_value_type<Args>...>;
    using E = broadcast_vector_extent_type<Args...>;
    vector_storage<Output, E::value> result;

    detail::apply_impl<F, E::value, Output, vector_value_type<Args>...>::call(
        fun,
        result.data(),
        (detail::broadcast_impl<vector_value_type<Args>, vector_extent_type<Args>, E>::call(
             into_vector_storage(args))
             .data())...);

    return result;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_APPLY_H