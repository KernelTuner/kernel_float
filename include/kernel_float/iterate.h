#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "storage.h"
#include "unops.h"

namespace kernel_float {

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct range_helper;

template<typename F, typename V, size_t... Is>
struct range_helper<F, V, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static V call(F fun) {
        return vector_traits<V>::create(fun(const_index<Is> {})...);
    }
};
}  // namespace detail

/**
 * Generate vector of length ``N`` by applying the given function ``fun`` to
 * each index ``0...N-1``.
 *
 * Example
 * =======
 * ```
 * // returns [0, 2, 4]
 * vector<float, 3> vec = range<3>([](auto i) { return float(i * 2); });
 * ```
 */
template<
    size_t N,
    typename F,
    typename T = result_t<F, size_t>,
    typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> range(F fun) {
    return detail::range_helper<F, Output>::call(fun);
}

/**
 * Generate vector consisting of the numbers ``0...N-1`` of type ``T``.
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2]
 * vector<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename T, size_t N, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> range() {
    using F = ops::cast<size_t, T>;
    return detail::range_helper<F, Output>::call(F {});
}

/**
 * Generate vector having same size and type as ``V``, but filled with the numbers ``0..N-1``.
 */
template<typename Input, typename Output = into_storage_type<Input>>
KERNEL_FLOAT_INLINE vector<Output> range_like(const Input&) {
    using F = ops::cast<size_t, vector_value_type<Input>>;
    return detail::range_helper<F, Output>::call(F {});
}

/**
 * Generate vector of `N` elements of type `T`
 *
 * Example
 * =======
 * ```
 * // Returns [1.0, 1.0, 1.0]
 * vector<float, 3> = fill(1.0f);
 * ```
 */
template<size_t N = 1, typename T, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> fill(T value) {
    return vector_traits<Output>::fill(value);
}

/**
 * Generate vector having same size and type as ``V``, but filled with the given ``value``.
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> fill_like(const Output&, vector_value_type<Output> value) {
    return vector_traits<Output>::fill(value);
}

/**
 * Generate vector of ``N`` zeros of type ``T``
 *
 * Example
 * =======
 * ```
 * // Returns [0.0, 0.0, 0.0]
 * vector<float, 3> = zeros();
 * ```
 */
template<size_t N = 1, typename T = bool, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> zeros() {
    return vector_traits<Output>::fill(T(0));
}

/**
 * Generate vector having same size and type as ``V``, but filled with zeros.
 *
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> zeros_like(const Output& output = {}) {
    return vector_traits<Output>::fill(0);
}

/**
 * Generate vector of ``N`` ones of type ``T``
 *
 * Example
 * =======
 * ```
 * // Returns [1.0, 1.0, 1.0]
 * vector<float, 3> = ones();
 * ```
 */
template<size_t N = 1, typename T = bool, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> ones() {
    return vector_traits<Output>::fill(T(1));
}

/**
 * Generate vector having same size and type as ``V``, but filled with ones.
 *
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> ones_like(const Output& output = {}) {
    return vector_traits<Output>::fill(1);
}

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct iterate_helper;

template<typename F, typename V>
struct iterate_helper<F, V, index_sequence<>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {}
};

template<typename F, typename V, size_t I, size_t... Rest>
struct iterate_helper<F, V, index_sequence<I, Rest...>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {
        fun(vector_get<I>(input));
        iterate_helper<F, V, index_sequence<Rest...>>::call(fun, input);
    }
};
}  // namespace detail

/**
 * Apply the function ``fun`` for each element from ``input``.
 *
 * Example
 * =======
 * ```
 * for_each(range<3>(), [&](auto i) {
 *    printf("element: %d\n", i);
 * });
 * ```
 */
template<typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(const V& input, F fun) {
    detail::iterate_helper<F, into_storage_type<V>>::call(fun, into_storage(input));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
