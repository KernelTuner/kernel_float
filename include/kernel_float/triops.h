#ifndef KERNEL_FLOAT_TRIOPS_H
#define KERNEL_FLOAT_TRIOPS_H

#include "broadcast.h"
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
 * This function broadcasts all arguments to the same shape and it promotes the values of `true_values` and
 * `false_values` into the same type. Next, it casts the values of `cond` to booleans and returns a tensor where
 * the values are taken from `true_values` if the condition is true and `false_values` otherwise.
 *
 * @param cond The condition used for selection.
 * @param true_values The tensor of values to choose from when the condition is true.
 * @param false_values The tensor of values to choose from when the condition is false.
 * @return A tensor containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename R,
    typename T = promoted_tensor_value_type<L, R>,
    typename E = broadcast_extents<tensor_extents<C>, broadcast_tensor_extents<L, R>>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond, const L& true_values, const R& false_values) {
    using F = ops::conditional<T>;

    return detail::apply_impl<F, E::volume, T, bool, T, T>::call(
        F {},
        detail::convert_helper<tensor_value_type<C>, tensor_extents<C>, bool, E>::call(
            into_tensor_storage(cond)),
        detail::convert_helper<tensor_value_type<L>, tensor_extents<L>, T, E>::call(
            into_tensor_storage(true_values)),
        detail::convert_helper<tensor_value_type<R>, tensor_extents<R>, T, E>::call(
            into_tensor_storage(false_values)));
}

/**
 * Selects elements from `true_values` depending on `cond`.
 *
 * This function returns a tensor where the values are taken from `true_values` where `cond` is `true` and `0` where
 * `cond is `false`.
 *
 * @param cond The condition used for selection.
 * @param true_values The tensor of values to choose from when the condition is true.
 * @return A tensor containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename T = tensor_value_type<L>,
    typename E = broadcast_extents<tensor_extents<C>, tensor_extents<L>>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond, const L& true_values) {
    tensor<T, extents<>> false_values = T {};
    return where(cond, true_values, false_values);
}

/**
 * Returns a tensor where the values are `T(1)` where `cond` is `true` and `T(0)` where `cond` is `false`.
 *
 * @param cond The condition used for selection.
 * @return A tensor containing elements as per the condition.
 */
template<typename T = bool, typename C, typename E = tensor_extents<C>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond) {
    tensor<T, extents<>> true_values = T {true};
    tensor<T, extents<>> false_values = T {false};
    return where(cond, true_values, false_values);
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TRIOPS_H
