#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "base.h"

namespace kernel_float {

/**
 * Apply the function fun for each element from input.
 *
 * ```
 * for_each(range<int, 3>(), [&](auto i) {
 *    printf("element: %d\n", i);
 * });
 * ```
 */
template<typename V, typename F>
void for_each(V&& input, F fun) {
    auto storage = into_vector_storage(input);

#pragma unroll
    for (size_t i = 0; i < vector_extent<V>; i++) {
        fun(storage.data()[i]);
    }
}

namespace detail {
template<typename T, size_t N>
struct range_helper {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call() {
        vector_storage<T, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = T(i);
        }

        return result;
    }
};
}  // namespace detail

/**
 * Generate vector consisting of the numbers 0...N-1 of type T
 *
 * ```
 * // Returns [0, 1, 2]
 * vector<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range() {
    return detail::range_helper<T, N>::call();
}

/**
 * Takes a vector of size ``N`` and element type ``T`` and returns a new vector consisting of the numbers ``0...N-1``
 * of type ``T``
 *
 * ```
 * // Returns [0.0f, 1.0f, 2.0f]
 * vector<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> range_like(const V& = {}) {
    return detail::range_helper<vector_value_type<V>, vector_extent<V>>::call();
}

/**
 * Takes a vector of size ``N`` and returns a new vector consisting of the numbers ``0...N-1`` of type ``size_t``
 *
 * ```
 * // Returns [0, 1, 2]
 * vector<size_t, 3> vec = enumerate(float3(6, 4, 2));
 * ```
 */
template<typename T = size_t, typename V>
KERNEL_FLOAT_INLINE vector<T, vector_extent_type<V>> enumerate(const V& = {}) {
    return detail::range_helper<T, vector_extent<V>>::call();
}

namespace detail {
template<typename V, typename T = vector_value_type<V>, size_t N = vector_extent<V>>
struct flatten_helper {
    using value_type = typename flatten_helper<T>::value_type;
    static constexpr size_t size = N * flatten_helper<T>::size;

    KERNEL_FLOAT_INLINE
    static void call(const V& input, value_type* output) {
        vector_storage<T, N> storage = into_vector_storage(input);

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            flatten_helper<T>::call(storage.data()[i], output + flatten_helper<T>::size * i);
        }
    }
};

template<typename T>
struct flatten_helper<T, T, 1> {
    using value_type = T;
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE
    static void call(const T& input, T* output) {
        *output = input;
    }
};
}  // namespace detail

template<typename V>
using flatten_value_type = typename detail::flatten_helper<V>::value_type;

template<typename V>
static constexpr size_t flatten_size = detail::flatten_helper<V>::size;

template<typename V>
using flatten_type = vector<flatten_value_type<V>, extent<flatten_size<V>>>;

template<typename V>
KERNEL_FLOAT_INLINE flatten_type<V> flatten(const V& input) {
    vector_storage<flatten_value_type<V>, flatten_size<V>> output;
    detail::flatten_helper<V>::call(input, output.data());
    return output;
}
}  // namespace kernel_float

#endif