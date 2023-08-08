#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "base.h"

namespace kernel_float {

namespace detail {
template<typename V, typename T = vector_value_type<V>, size_t N = vector_extent<V>>
struct flatten_helper {
    using value_type = typename flatten_helper<T>::value_type;
    static constexpr size_t size = N * flatten_helper<T>::size;

    static void call(const V& input, value_type* output) {
        vector_storage<T, N> storage = into_vector_storage(input);

        for (size_t i = 0; i < N; i++) {
            flatten_helper<T>::call(storage.data()[i], output + flatten_helper<T>::size * i);
        }
    }
};

template<typename T>
struct flatten_helper<T, T, 1> {
    using value_type = T;
    static constexpr size_t size = 1;

    static void call(const T& input, T* output) {
        *output = input;
    }
};
}  // namespace detail

template<typename V>
using flatten_value_type = typename detail::flatten_helper<V>::value_type;

template<typename V>
static constexpr size_t flatten_size = detail::flatten_helper<V>::size;

template<typename V>
using flatten_type = vector<flatten_value_type<V>, extent<flatten_size<V>>>;

template<typename V>
flatten_type<V> flatten(const V& input) {
    vector_storage<flatten_value_type<V>, flatten_size<V>> output;
    detail::flatten_helper<V>::call(input, output.data());
    return output;
}
}  // namespace kernel_float

#endif