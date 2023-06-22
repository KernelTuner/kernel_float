#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H

#include "binops.h"

namespace kernel_float {
namespace detail {
template<typename F, size_t N, typename T, typename = void>
struct reduce_helper {
    KERNEL_FLOAT_INLINE static T call(F fun, const tensor_storage<T, N>& input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static T
    call(F fun, const tensor_storage<T, N>& input, index_sequence<0, Is...>) {
        T result = input[0];
#pragma unroll
        for (size_t i = 1; i < N; i++) {
            result = fun(result, input[i]);
        }
        return result;
    }
};
}  // namespace detail

/**
 * Reduce the elements of the given vector ``input`` into a single value using
 * the function ``fun``. This function should be a binary function that takes
 * two elements and returns one element. The order in which the elements
 * are reduced is not specified and depends on the reduction function and
 * the vector type.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 2, 1};
 * int y = reduce(x, [](int a, int b) { return a + b; }); // returns 5+2+1=8
 * ```
 */
template<typename F, typename V>
KERNEL_FLOAT_INLINE tensor_value_type<V> reduce(F fun, const V& input) {
    return detail::reduce_helper<F, tensor_volume<V>, tensor_value_type<V>>::call(
        fun,
        into_tensor_storage(input));
}

/**
 * Find the minimum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 0, 2, 1, 0};
 * int y = min(x);  // Returns 0
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T min(const V& input) {
    return reduce(ops::min<T> {}, input);
}

/**
 * Find the maximum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 0, 2, 1, 0};
 * int y = max(x);  // Returns 5
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T max(const V& input) {
    return reduce(ops::max<T> {}, input);
}

/**
 * Sum the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 8
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T sum(const V& input) {
    return reduce(ops::add<T> {}, input);
}

/**
 * Multiply the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 5*0*2*1*0 = 0
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T product(const V& input) {
    return reduce(ops::multiply<T> {}, input);
}

/**
 * Check if all elements in the given vector ``input`` are non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template<typename V>
KERNEL_FLOAT_INLINE bool all(const V& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

/**
 * Check if any element in the given vector ``input`` is non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template<typename V>
KERNEL_FLOAT_INLINE bool any(const V& input) {
    return reduce(ops::bit_or<bool> {}, cast<bool>(input));
}

/**
 * Count the number of non-zero items in the given vector ``input``. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = count(x);  // Returns 3 (5, 2, 1 are non-zero)
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE int count(const V& input) {
    return sum(cast<int>(cast<bool>(input)));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
