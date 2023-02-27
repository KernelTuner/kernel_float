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

template<typename V, typename F>
KERNEL_FLOAT_INLINE V range(F fun) {
    return detail::range_helper<F, V>::call(fun);
}

template<typename V>
KERNEL_FLOAT_INLINE V range() {
    return range<V>(ops::cast<size_t, vector_value_type<V>> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector_storage<T, N> range() {
    return range<vector_storage<T, N>>();
}

template<size_t N>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range() {
    return range<vector_storage<size_t, N>>();
}

template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range(F fun) {
    return range<vector_storage<T, N>>(fun);
}

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range(F fun) {
    return range<vector_storage<T, N>>(fun);
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

template<typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(V&& input, F fun) {
    detail::iterate_helper<F, into_vector_type<V>>::call(fun, into_vector(input));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
