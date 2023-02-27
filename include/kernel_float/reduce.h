#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H

#include "binops.h"

namespace kernel_float {
namespace detail {
template<typename F, typename V, typename = void>
struct reduce_helper {
    using value_type = vector_value_type<V>;

    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& input) {
        return call(fun, input, make_index_sequence<vector_size<V>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& vector, index_sequence<0, Is...>) {
        return call(fun, vector, vector.get(const_index<0> {}), index_sequence<Is...> {});
    }

    template<size_t I, size_t... Rest>
    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<I, Rest...>) {
        return call(
            fun,
            vector,
            fun(accum, vector.get(const_index<I> {})),
            index_sequence<Rest...> {});
    }

    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<>) {
        return accum;
    }
};

template<typename F, typename T, size_t N>
struct reduce_helper<F, vector_compound<T, N>> {
    KERNEL_FLOAT_INLINE static T call(F fun, const vector_compound<T, N>& input) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return fun(
            reduce_helper<F, vector_storage<T, low_size>>::call(fun, input.low()),
            reduce_helper<F, vector_storage<T, high_size>>::call(fun, input.high()));
    }
};
}  // namespace detail

template<typename F, typename V>
KERNEL_FLOAT_INLINE vector_value_type<V> reduce(F fun, V&& input) {
    return detail::reduce_helper<F, into_vector_type<V>>::call(
        fun,
        into_vector(std::forward<V>(input)));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T min(V&& input) {
    return reduce(ops::min<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T max(V&& input) {
    return reduce(ops::max<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T sum(V&& input) {
    return reduce(ops::add<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T product(V&& input) {
    return reduce(ops::multiply<T> {}, std::forward<V>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE bool all(V&& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE bool any(V&& input) {
    return reduce(ops::bit_or<bool> {}, cast<bool>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE int count(V&& input) {
    return sum(cast<int>(cast<bool>(input)));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
