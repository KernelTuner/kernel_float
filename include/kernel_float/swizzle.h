#ifndef KERNEL_FLOAT_SWIZZLE_H
#define KERNEL_FLOAT_SWIZZLE_H

#include "storage.h"

namespace kernel_float {

/**
 * "Swizzles" the vector. Returns a new vector where the elements are provided by the given indices.
 *
 * # Example
 * ```
 * vec<int, 6> x = {0, 1, 2, 3, 4, 5, 6};
 * vec<int, 3> a = swizzle<0, 1, 2>(x);  // 0, 1, 2
 * vec<int, 3> b = swizzle<2, 1, 0>(x);  // 2, 1, 0
 * vec<int, 3> c = swizzle<1, 1, 1>(x);  // 1, 1, 1
 * vec<int, 4> d = swizzle<0, 2, 4, 6>(x);  // 0, 2, 4, 6
 * ```
 */
template<
    size_t... Is,
    typename V,
    typename Output = default_storage_type<vector_value_type<V>, sizeof...(Is)>>
KERNEL_FLOAT_INLINE vector<Output> swizzle(const V& input, index_sequence<Is...> _ = {}) {
    return vector_swizzle<Output, into_storage_type<V>, index_sequence<Is...>>::call(
        into_storage(input));
}

/**
 * Takes the first ``N`` elements from the given vector and returns a new vector of length ``N``.
 *
 * # Example
 * ```
 * vec<int, 6> x = {1, 2, 3, 4, 5, 6};
 * vec<int, 6> y = first<3>(x);  // 1, 2, 3
 * int z = first(x);  // 1
 * ```
 */
template<size_t K = 1, typename V, typename Output = default_storage_type<vector_value_type<V>, K>>
KERNEL_FLOAT_INLINE vector<Output> first(const V& input) {
    static_assert(K <= vector_size<V>, "K cannot exceed vector size");
    using Indices = make_index_sequence<K>;
    return vector_swizzle<Output, into_storage_type<V>, Indices>::call(into_storage(input));
}

namespace detail {
template<size_t Offset, typename Indices>
struct offset_index_sequence_helper;

template<size_t Offset, size_t... Is>
struct offset_index_sequence_helper<Offset, index_sequence<Is...>> {
    using type = index_sequence<Offset + Is...>;
};
}  // namespace detail

/**
 * Takes the last ``N`` elements from the given vector and returns a new vector of length ``N``.
 *
 * # Example
 * ```
 * vec<int, 6> x = {1, 2, 3, 4, 5, 6};
 * vec<int, 6> y = last<3>(x);  // 4, 5, 6
 * int z = last(x);  // 6
 * ```
 */
template<size_t K = 1, typename V, typename Output = default_storage_type<vector_value_type<V>, K>>
KERNEL_FLOAT_INLINE vector<Output> last(const V& input) {
    static_assert(K <= vector_size<V>, "K cannot exceed vector size");
    using Indices = typename detail::offset_index_sequence_helper<  //
        vector_size<V> - K,
        make_index_sequence<K>>::type;

    return vector_swizzle<Output, into_storage_type<V>, Indices>::call(into_storage(input));
}

namespace detail {
template<size_t N, size_t... Is>
struct reverse_index_sequence_helper: reverse_index_sequence_helper<N - 1, Is..., N - 1> {};

template<size_t... Is>
struct reverse_index_sequence_helper<0, Is...> {
    using type = index_sequence<Is...>;
};
}  // namespace detail

/**
 * Reverses the elements in the given vector.
 *
 * # Example
 * ```
 * vec<int, 6> x = {1, 2, 3, 4, 5, 6};
 * vec<int, 6> y = reversed(x);  // 6, 5, 4, 3, 2, 1
 * ```
 */
template<typename V, typename Output = into_storage_type<V>>
KERNEL_FLOAT_INLINE vector<Output> reversed(const V& input) {
    using Indices = typename detail::reverse_index_sequence_helper<vector_size<V>>::type;

    return vector_swizzle<Output, into_storage_type<V>, Indices>::call(into_storage(input));
}

namespace detail {
template<typename I, typename J>
struct concat_index_sequence_helper {};

template<size_t... Is, size_t... Js>
struct concat_index_sequence_helper<index_sequence<Is...>, index_sequence<Js...>> {
    using type = index_sequence<Is..., Js...>;
};
}  // namespace detail

/**
 * Rotate the given vector ``K`` steps to the right. In other words, this move the front element to the back
 * ``K`` times. This is the inverse of ``rotate_left``.
 *
 * # Example
 * ```
 * vec<int, 6> x = {1, 2, 3, 4, 5, 6};
 * vec<int, 6> y = rotate_right<2>(x);  // 5, 6, 1, 2, 3, 4
 * ```
 */
template<size_t K = 1, typename V, typename Output = into_storage_type<V>>
KERNEL_FLOAT_INLINE vector<Output> rotate_right(const V& input) {
    static constexpr size_t N = vector_size<V>;
    static constexpr size_t I = (N > 0) ? (K % N) : 0;

    using First =
        typename detail::offset_index_sequence_helper<N - I, make_index_sequence<I>>::type;
    using Second = make_index_sequence<N - I>;
    using Indices = typename detail::concat_index_sequence_helper<First, Second>::type;

    return vector_swizzle<Output, into_storage_type<V>, Indices>::call(into_storage(input));
}

/**
 * Rotate the given vector ``K`` steps to the left. In other words, this move the back element to the front
 * ``K`` times. This is the inverse of ``rotate_right``.
 *
 * # Example
 * ```
 * vec<int, 6> x = {1, 2, 3, 4, 5, 6};
 * vec<int, 6> y = rotate_left<4>(x);  // 5, 6, 1, 2, 3, 4
 * ```
 */
template<size_t K = 1, typename V, typename Output = into_storage_type<V>>
KERNEL_FLOAT_INLINE vector<Output> rotate_left(const V& input) {
    static constexpr size_t N = vector_size<V>;
    static constexpr size_t K_rev = N > 0 ? (N - K % N) : 0;

    return rotate_right<K_rev, V, Output>(input);
}

namespace detail {
template<
    typename U,
    typename V,
    typename Is = make_index_sequence<vector_size<U>>,
    typename Js = make_index_sequence<vector_size<V>>>
struct concat_helper;

template<typename U, typename V, size_t... Is, size_t... Js>
struct concat_helper<U, V, index_sequence<Is...>, index_sequence<Js...>> {
    using type = default_storage_type<
        common_t<vector_value_type<U>, vector_value_type<V>>,
        vector_size<U> + vector_size<V>>;

    KERNEL_FLOAT_INLINE static type call(const U& left, const V& right) {
        return vector_traits<type>::create(vector_get<Is>(left)..., vector_get<Js>(right)...);
    }
};

template<typename... Ts>
struct recur_concat_helper;

template<typename U>
struct recur_concat_helper<U> {
    using type = U;

    KERNEL_FLOAT_INLINE static U call(U&& input) {
        return input;
    }
};

template<typename U, typename V, typename... Rest>
struct recur_concat_helper<U, V, Rest...> {
    using recur_helper = recur_concat_helper<typename concat_helper<U, V>::type, Rest...>;
    using type = typename recur_helper::type;

    KERNEL_FLOAT_INLINE static type call(const U& left, const V& right, const Rest&... rest) {
        return recur_helper::call(concat_helper<U, V>::call(left, right), rest...);
    }
};
}  // namespace detail

template<typename... Vs>
using concat_type = typename detail::recur_concat_helper<into_storage_type<Vs>...>::type;

/**
 * Concatenate the given vectors into one large vector. For example, given vectors of size 3, size 2 and size 5,
 * this function returns a new vector of size 3+2+5=8. If the vectors are not of the same element type, they
 * will first be cast into a common data type.
 *
 * # Examples
 * ```
 * vec<int, 3> x = {1, 2, 3};
 * int y = 4;
 * vec<int, 4> z = {5, 6, 7, 8};
 * vec<int, 8> xyz = concat(x, y, z);  // 1, 2, 3, 4, 5, 6, 7, 8
 * ```
 */
template<typename... Vs>
KERNEL_FLOAT_INLINE vector<concat_type<Vs...>> concat(const Vs&... inputs) {
    return detail::recur_concat_helper<into_storage_type<Vs>...>::call(into_storage(inputs)...);
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_SWIZZLE_H
