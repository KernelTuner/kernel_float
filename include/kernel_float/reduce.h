#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H

#include "binops.h"

namespace kernel_float {
namespace detail {
template<typename F, size_t N, typename T, typename = void>
struct reduce_impl {
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input, index_sequence<0, Is...>) {
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
 * are reduced is not specified and depends on both the reduction function and
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
KERNEL_FLOAT_INLINE vector_value_type<V> reduce(F fun, const V& input) {
    return detail::reduce_impl<F, vector_extent<V>, vector_value_type<V>>::call(
        fun,
        into_vector_storage(input).data());
}

/**
 * Find the minimum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = min(x);  // Returns 0
 * ```
 */
template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T min(const V& input) {
    return reduce(ops::min<T> {}, input);
}

/**
 * Find the maximum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = max(x);  // Returns 5
 * ```
 */
template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T max(const V& input) {
    return reduce(ops::max<T> {}, input);
}

/**
 * Sum the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 8
 * ```
 */
template<typename V, typename T = vector_value_type<V>>
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
template<typename V, typename T = vector_value_type<V>>
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
template<typename T = int, typename V>
KERNEL_FLOAT_INLINE T count(const V& input) {
    return sum(cast<T>(cast<bool>(input)));
}

namespace detail {
template<typename T, size_t N>
struct dot_impl {
    KERNEL_FLOAT_INLINE
    static T call(const T* left, const T* right) {
        vector_storage<T, N> intermediate;
        detail::apply_impl<ops::multiply<T>, N, T, T, T>::call(
            ops::multiply<T>(),
            intermediate.data(),
            left,
            right);

        return detail::reduce_impl<ops::add<T>, N, T>::call(ops::add<T>(), intermediate.data());
    }
};
}  // namespace detail

/**
 * Compute the dot product of the given vectors ``left`` and ``right``
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {1, 2, 3};
 * vec<int, 3> y = {4, 5, 6};
 * int y = dot(x, y);  // Returns 1*4+2*5+3*6 = 32
 * ```
 */
template<typename L, typename R, typename T = promoted_vector_value_type<L, R>>
KERNEL_FLOAT_INLINE T dot(const L& left, const R& right) {
    using E = broadcast_vector_extent_type<L, R>;
    return detail::dot_impl<T, E::value>::call(
        convert_storage<T>(left, E {}).data(),
        convert_storage<T>(right, E {}).data());
}

namespace detail {
template<typename T, size_t N>
struct magnitude_impl {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) {
        return ops::sqrt<T> {}(detail::dot_impl<T, N>::call(input, input));
    }
};

template<typename T>
struct magnitude_impl<T, 0> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) {
        return T {};
    }
};

template<typename T>
struct magnitude_impl<T, 1> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) {
        return ops::abs<T> {}(input[0]);
    }
};

template<typename T>
struct magnitude_impl<T, 2> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) {
        return ops::hypot<T>()(input[0], input[1]);
    }
};

// The 3-argument overload of hypot is only available on host from C++17
#if defined(__cpp_lib_hypot) && KERNEL_FLOAT_IS_HOST
template<>
struct magnitude_impl<float, 3> {
    static float call(const float* input) {
        return ::hypot(input[0], input[1], input[2]);
    }
};

template<>
struct magnitude_impl<double, 3> {
    static double call(const double* input) {
        return ::hypot(input[0], input[1], input[2]);
    }
};
#endif

}  // namespace detail

/**
 * Compute the magnitude of the given input vector. This calculates the square root of the sum of squares, also
 * known as the Euclidian norm, of a vector.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> x = {2, 3, 6};
 * float y = mag(x);  // Returns sqrt(2*2 + 3*3 + 6*6) = 7
 * ```
 */
template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T mag(const V& input) {
    return detail::magnitude_impl<T, vector_extent<V>>::call(into_vector_storage(input).data());
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
