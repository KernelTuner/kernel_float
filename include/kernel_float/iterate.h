#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "base.h"
#include "conversion.h"

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
KERNEL_FLOAT_INLINE void for_each(V&& input, F fun) {
    auto storage = into_vector_storage(input);

#pragma unroll
    for (size_t i = 0; i < vector_extent<V>; i++) {
        fun(storage.data()[i]);
    }
}

namespace detail {
template<typename T, size_t N>
struct range_impl {
    template<typename F>
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(F fun) {
        vector_storage<T, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = fun(i);
        }

        return result;
    }
};
}  // namespace detail

/**
 * Generate vector consisting of the result `fun(0)...fun(N-1)`
 *
 * Example
 * =======
 * ```
 * // Returns [0.0f, 2.0f, 4.0f]
 * vec<float, 3> vec = range<3>([](auto i){ return float(i * 2.0f); });
 * ```
 */
template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range(F fun) {
    return detail::range_impl<T, N>::call(fun);
}

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
    return detail::range_impl<T, N>::call(ops::cast<size_t, T>());
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
    return range<vector_value_type<V>, vector_extent<V>>();
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
    return range<T, vector_extent<V>>();
}

namespace detail {
template<typename V, typename T = vector_value_type<V>, size_t N = vector_extent<V>>
struct flatten_impl {
    using value_type = typename flatten_impl<T>::value_type;
    static constexpr size_t size = N * flatten_impl<T>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, N> storage = into_vector_storage(input);

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            flatten_impl<T>::call(output + flatten_impl<T>::size * i, storage.data()[i]);
        }
    }
};

template<typename T>
struct flatten_impl<T, T, 1> {
    using value_type = T;
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE
    static void call(T* output, const T& input) {
        *output = input;
    }

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) {
        *output = ops::cast<T, U> {}(input);
    }
};
}  // namespace detail

template<typename V>
using flatten_value_type = typename detail::flatten_impl<V>::value_type;

template<typename V>
static constexpr size_t flatten_size = detail::flatten_impl<V>::size;

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
    detail::flatten_impl<V>::call(output.data(), input);
    return output;
}

namespace detail {
template<typename U, typename V = U, typename T = vector_value_type<V>>
struct concat_base_impl {
    static constexpr size_t size = vector_extent<V>;

    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, size> storage = into_vector_storage(input);

        for (size_t i = 0; i < size; i++) {
            output[i] = ops::cast<T, U> {}(storage.data()[i]);
        }
    }
};

template<typename U, typename T>
struct concat_base_impl<U, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) {
        *output = ops::cast<T, U> {}(input);
    }
};

template<typename T>
struct concat_base_impl<T, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(T* output, const T& input) {
        *output = input;
    }
};

template<typename... Vs>
struct concat_impl {};

template<typename V, typename... Vs>
struct concat_impl<V, Vs...> {
    using value_type =
        typename promote_type<vector_value_type<V>, typename concat_impl<Vs...>::value_type>::type;
    static constexpr size_t size = concat_base_impl<V>::size + concat_impl<Vs...>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input, const Vs&... rest) {
        concat_base_impl<U, V>::call(output, input);
        concat_impl<Vs...>::call(output + concat_base_impl<U, V>::size, rest...);
    }
};

template<>
struct concat_impl<> {
    using value_type = void;
    static constexpr size_t size = 1;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output) {}
};
}  // namespace detail

template<typename... Vs>
using concat_value_type = promote_t<typename detail::concat_impl<Vs...>::value_type>;

template<typename... Vs>
static constexpr size_t concat_size = detail::concat_impl<Vs...>::size;

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
    detail::concat_impl<Vs...>::call(output.data(), inputs...);
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
    detail::concat_impl<Is...>::call(index_set.data(), indices...);

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
