#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H

#include "base.h"
#include "unops.h"

namespace kernel_float {
namespace detail {

template<typename A, typename B>
struct broadcast_extent_helper;

template<size_t N>
struct broadcast_extent_helper<extent<N>, extent<N>> {
    using type = extent<N>;
};

template<size_t N>
struct broadcast_extent_helper<extent<1>, extent<N>> {
    using type = extent<N>;
};

template<size_t N>
struct broadcast_extent_helper<extent<N>, extent<1>> {
    using type = extent<N>;
};

template<>
struct broadcast_extent_helper<extent<1>, extent<1>> {
    using type = extent<1>;
};

}  // namespace detail

template<typename A, typename B>
using broadcast_extent = typename detail::broadcast_extent_helper<A, B>::type;

template<typename A, typename B>
using broadcast_vector_extent_type = broadcast_extent<vector_extent_type<A>, vector_extent_type<B>>;

template<typename From, typename To>
static constexpr bool is_broadcastable = is_same<broadcast_extent<From, To>, To>;

template<typename V, typename To>
static constexpr bool is_vector_broadcastable = is_broadcastable<vector_extent_type<V>, To>;

namespace detail {

template<typename T, typename From, typename To>
struct broadcast_impl;

template<typename T, size_t N>
struct broadcast_impl<T, extent<1>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(const vector_storage<T, 1>& input) {
        vector_storage<T, N> output;
        for (size_t i = 0; i < N; i++) {
            output.data()[i] = input.data()[0];
        }
        return output;
    }
};

template<typename T, size_t N>
struct broadcast_impl<T, extent<N>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(vector_storage<T, N> input) {
        return input;
    }
};

}  // namespace detail

template<size_t N, typename V>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, extent<N>>
broadcast(const V& input, extent<N> new_size = {}) {
    using T = vector_value_type<V>;
    return detail::broadcast_impl<T, vector_extent_type<V>, extent<N>>::call(
        into_vector_storage(input));
}

template<typename V, typename R>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<R>>
broadcast_like(const V& input, const R&) {
    using T = vector_value_type<V>;
    return detail::broadcast_impl<T, vector_extent_type<V>, vector_extent_type<R>>::call(
        into_vector_storage(input));
}

template<size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> fill(T value = {}, extent<N> = {}) {
    vector_storage<T, 1> input = {value};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> zeros(extent<N> = {}) {
    vector_storage<T, 1> input = {T {}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> ones(extent<N> = {}) {
    vector_storage<T, 1> input = {T {1}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> zeros_like(const V&) {
    return zeros<T>(E {});
}

template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> ones_like(const V&) {
    return ones<T>(E {});
}

namespace detail {
template<typename T, typename E, typename T2, typename E2, RoundingMode M = RoundingMode::ANY>
struct convert_helper {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, E2::value> call(vector_storage<T, E::value> input) {
        using F = ops::cast<T, T2, M>;
        vector_storage<T2, E::value> intermediate =
            detail::apply_impl<F, E::value, T2, T>::call(F {}, input);
        return detail::broadcast_impl<T2, E, E2>::call(intermediate);
    }
};

template<typename T, typename E, RoundingMode M>
struct convert_helper<T, E, T, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, E::value> call(vector_storage<T, E::value> input) {
        return input;
    }
};

template<typename T, typename E, typename E2, RoundingMode M>
struct convert_helper<T, E, T, E2, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, E2::value> call(vector_storage<T, E::value> input) {
        return detail::broadcast_impl<T, E, E2>::call(input);
    }
};

template<typename T, typename E, typename T2, RoundingMode M>
struct convert_helper<T, E, T2, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, E::value> call(vector_storage<T, E::value> input) {
        using F = ops::cast<T, T2, M>;
        return detail::apply_impl<F, E::value, T2, T>::call(F {}, input);
    }
};
}  // namespace detail

template<typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector_storage<R, N> convert_storage(const V& input, extent<N> new_size = {}) {
    return detail::convert_helper<vector_value_type<V>, vector_extent_type<V>, R, extent<N>, M>::
        call(into_vector_storage(input));
}

/**
 * Cast the values of the given input vector to type `R` and then broadcast the result to the given size `N`.
 */
template<typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, extent<N>> convert(const V& input, extent<N> new_size = {}) {
    return convert_storage(input);
}

}  // namespace kernel_float

#endif
