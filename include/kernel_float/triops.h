#ifndef KERNEL_FLOAT_TRIOPS_H
#define KERNEL_FLOAT_TRIOPS_H

#include "conversion.h"
#include "unops.h"

namespace kernel_float {

namespace ops {
template<typename T>
struct conditional {
    KERNEL_FLOAT_INLINE T operator()(bool cond, T true_value, T false_value) {
        if (cond) {
            return true_value;
        } else {
            return false_value;
        }
    }
};
}  // namespace ops

/**
 * Return elements chosen from `true_values` and `false_values` depending on `cond`.
 *
 * This function broadcasts all arguments to the same size and then promotes the values of `true_values` and
 * `false_values` into the same type. Next, it casts the values of `cond` to booleans and returns a vector where
 * the values are taken from `true_values` where the condition is true and `false_values` otherwise.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @param false_values The vector of values to choose from when the condition is false.
 * @return A vector containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename R,
    typename T = promoted_vector_value_type<L, R>,
    typename E = broadcast_vector_extent_type<C, L, R>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values, const R& false_values) {
    using F = ops::conditional<T>;
    vector_storage<T, E::value> result;

    detail::apply_impl<F, E::value, T, bool, T, T>::call(
        F {},
        result.data(),
        detail::convert_impl<vector_value_type<C>, vector_extent_type<C>, bool, E>::call(
            into_vector_storage(cond))
            .data(),
        detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(true_values))
            .data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(false_values))
            .data());

    return result;
}

/**
 * Selects elements from `true_values` depending on `cond`.
 *
 * This function returns a vector where the values are taken from `true_values` where `cond` is `true` and `0` where
 * `cond is `false`.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @return A vector containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename T = vector_value_type<L>,
    typename E = broadcast_vector_extent_type<C, L>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values) {
    vector<T, extent<1>> false_values = T {};
    return where(cond, true_values, false_values);
}

/**
 * Returns a vector having the value `T(1)` where `cond` is `true` and `T(0)` where `cond` is `false`.
 *
 * @param cond The condition used for selection.
 * @return A vector containing elements as per the condition.
 */
template<typename T = bool, typename C, typename E = vector_extent_type<C>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond) {
    return cast<T>(cast<bool>(cond));
}

namespace ops {
template<typename T>
struct fma {
    KERNEL_FLOAT_INLINE T operator()(T a, T b, T c) {
        return a + b * c;
    }
};

#if KERNEL_FLOAT_IS_DEVICE
template<>
struct fma<float> {
    KERNEL_FLOAT_INLINE float operator()(float a, float b, float c) {
        return __fmaf_rn(a, b, c);
    }
};

template<>
struct fma<double> {
    KERNEL_FLOAT_INLINE double operator()(double a, double b, double c) {
        return __fma_rn(a, b, c);
    }
};
#endif
}  // namespace ops

/**
 * Computes the result of `a * b + c`. This is done in a single operation if possible.
 */
template<
    typename A,
    typename B,
    typename C,
    typename T = promoted_vector_value_type<A, B, C>,
    typename E = broadcast_vector_extent_type<A, B, C>>
KERNEL_FLOAT_INLINE vector<T, E> fma(const A& a, const B& b, const C& c) {
    using F = ops::fma<T>;
    vector_storage<T, E::value> result;

    detail::apply_impl<F, E::value, T, T, T, T>::call(
        F {},
        result.data(),
        detail::convert_impl<vector_value_type<A>, vector_extent_type<A>, T, E>::call(
            into_vector_storage(a))
            .data(),
        detail::convert_impl<vector_value_type<B>, vector_extent_type<B>, T, E>::call(
            into_vector_storage(b))
            .data(),
        detail::convert_impl<vector_value_type<C>, vector_extent_type<C>, T, E>::call(
            into_vector_storage(c))
            .data());

    return result;
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TRIOPS_H
