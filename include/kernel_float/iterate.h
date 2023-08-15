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

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(const V& input, U* output) {
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

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(const T& input, U* output) {
        *output = ops::cast<T, U> {}(input);
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

namespace detail {
template<typename... Vs>
struct concat_helper {};

template<typename V, typename... Vs>
struct concat_helper<V, Vs...> {
    using value_type = typename promote_type<
        typename flatten_helper<V>::value_type,
        typename concat_helper<Vs...>::value_type>::type;
    static constexpr size_t size = flatten_helper<V>::size + concat_helper<Vs...>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input, const Vs&... rest) {
        flatten_helper<V>::call(input, output);
        concat_helper<Vs...>::call(output + flatten_helper<V>::size, rest...);
    }
};

template<typename V>
struct concat_helper<V> {
    using value_type = typename promote_type<
        typename flatten_helper<V>::value_type,
        typename concat_helper<Vs...>::value_type>::type;
    static constexpr size_t size = flatten_helper<V>::size + concat_helper<Vs...>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input, const Vs&... rest) {
        flatten_helper<V>::call(input, output);
        concat_helper<Vs...>::call(output + flatten_helper<V>::size, rest...);
    }
};
}  // namespace detail

template<typename... Vs>
using concat_value_type = promote_t<typename detail::concat_helper<Vs...>::value_type>;

template<typename... Vs>
static constexpr size_t concat_size = detail::concat_helper<Vs...>::size;

template<typename... Vs>
using concat_type = vector<concat_value_type<Vs...>, extent<concat_size<Vs...>>>;

/**
 * Concatenates the provided input values into a single one-dimensional vector.
 *
 * This function works in three steps:
 * - All input values are converted into vectors using the `into_vector` operation.
 * - The resulting vectors' elements are then promoted into a shared value type.
 * - The resultant vectors are finally concatenated together.
 *
 * For instance, when invoking this function with arguments of types `float, double2, double`:
 * - After the first step: `vec<float, 1>, vec<double, 2>, vec<double, 1>`
 * - After the second step: `vec<double, 1>, vec<double, 2>, vec<double, 1>`
 * - After the third step: `vec<double, 4>`
 *
 * Example
 * =======
 * ```
 * double vec1 = 1.0;
 * double3 vec2 = {3.0, 4.0, 5.0);
 * double4 vec3 = {6.0, 7.0, 8.0, 9.0};
 * vec<double, 9> concatenated = concat(vec1, vec2, vec3); // contains [1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 * int num1 = 42;
 * float num2 = 3.14159;
 * int2 num3 = {-10, 10};
 * vec<float, 3> concatenated = concat(num1, num2, num3); // contains [42, 3.14159, -10, 10]
 * ```
 */
template<typename... Vs>
KERNEL_FLOAT_INLINE concat_type<Vs...> concat(const Vs&... inputs) {
    vector_storage<concat_value_type<Vs...>, concat_size<Vs...>> output;
    detail::concat_helper<Vs...>::call(output.data(), inputs...);
    return output;
}

template<typename V, typename... Is>
using select_type = vector<vector_value_type<V>, extent<concat_size<Is...>>>;

/**
 * Selects elements from the this vector based on the specified indices.
 *
 * Example
 * =======
 * ```
 * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
 * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
 *
 * vec<int, 4> indices = {0, 4, 4, 2};
 * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
 * ```
 */
template<typename V, typename... Is>
KERNEL_FLOAT_INLINE select_type<V, Is...> select(const V& input, const Is&... indices) {
    using T = vector_value_type<V>;
    static constexpr size_t N = vector_extent<V>;
    static constexpr size_t M = concat_size<Is...>;

    vector_storage<size_t, M> index_set;
    detail::concat_helper<Is...>::call(index_set.data(), indices...);

    vector_storage<T, N> inputs = into_vector_storage(input);
    vector_storage<T, M> outputs;
    for (size_t i = 0; i < M; i++) {
        size_t j = index_set.data()[i];

        if (j < N) {
            outputs.data()[i] = inputs.data()[j];
        }
    }

    return outputs;
}

}  // namespace kernel_float

#endif