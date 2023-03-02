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
        return V {fun(const_index<Is> {})...};
    }
};
}  // namespace detail

/**
 * Generate vector of length ``N`` by applying the given function ``fun`` to
 * each index ``0...N-1``.
 */
template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vector_storage<T, N> range(F fun) {
    return detail::range_helper<F, vector_storage<T, N>>::call(fun);
}

/**
 * Generate vector consisting of the numbers ``0...N-1`` of type ``T``.
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector_storage<T, N> range() {
    return range(ops::cast<size_t, T> {});
}

/**
 * Generate vector of `N` elements of type `T`
 *
 * ```
 * vector<float, 3> = fill(1.0);
 * ```
 */
template<size_t N = 1, typename T>
KERNEL_FLOAT_INLINE vector_storage<T, N> fill(T value) {
    return {value};
}

/**
 * Generate vector of ``N`` zeros of type ``T``
 *
 * ```
 * vector<float, 3> = zeros();
 * ```
 */
template<size_t N = 1, typename T = bool>
KERNEL_FLOAT_INLINE vector_storage<T, N> zeros() {
    return fill<N, T>(T(0));
}

/**
 * Generate vector of ``N`` ones of type ``T``
 *
 * ```
 * vector<float, 3> = ones();
 * ```
 */
template<size_t N = 1, typename T = bool>
KERNEL_FLOAT_INLINE vector_storage<T, N> ones() {
    return fill<N, T>(T(1));
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
        fun(input.get(const_index<I> {}));
        iterate_helper<F, V, index_sequence<Rest...>>::call(fun, input);
    }
};
}  // namespace detail

/**
 * Apply the function ``fun`` for each element from ``input``.
 */
template<typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(V&& input, F fun) {
    detail::iterate_helper<F, into_vector_type<V>>::call(fun, into_vector(input));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
