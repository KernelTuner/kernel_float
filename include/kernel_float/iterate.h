#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "base.h"

namespace kernel_float {

/**
 * Apply the function fun for each element from input.
 *
 * Example
 * =======
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
 * Generate vector consisting of the numbers `0...N-1` of type `T`
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2]
 * vec<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range() {
    return detail::range_helper<T, N>::call();
}

/**
 * Takes a vector `vec<T, N>` and returns a new vector consisting of the numbers ``0...N-1`` of type ``T``
 *
 * Example
 * =======
 * ```
 * auto input = vec<float, 3>(5.0f, 10.0f, -1.0f);
 * auto indices = range_like(input);  // returns [0.0f, 1.0f, 2.0f]
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> range_like(const V& = {}) {
    return detail::range_helper<vector_value_type<V>, vector_extent<V>>::call();
}

/**
 * Takes a vector of size ``N`` and returns a new vector consisting of the numbers ``0...N-1``. The data type used
 * for the indices is given by the first template argument, which is `size_t` by default. This function is useful when
 * needing to iterate over the indices of a vector.
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2] of type size_t
 * vec<size_t, 3> a = each_index(float3(6, 4, 2));
 *
 * // Returns [0, 1, 2] of type int.
 * vec<int, 3> b = each_index<int>(float3(6, 4, 2));
 *
 * vec<float, 3> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * for (auto index: each_index<int>(input)) {
 *   printf("%d] %f\n", index, input[index]);
 * }
 * ```
 */
template<typename T = size_t, typename V>
KERNEL_FLOAT_INLINE vector<T, vector_extent_type<V>> each_index(const V& = {}) {
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

/**
 * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
 *
 * Example
 * =======
 * ```
 * vec<float2, 3> input = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
 * vec<float, 6> result = flatten(input); // returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE flatten_type<V> flatten(const V& input) {
    vector_storage<flatten_value_type<V>, flatten_size<V>> output;
    detail::flatten_helper<V>::call(input, output.data());
    return output;
}
}  // namespace kernel_float

#endif