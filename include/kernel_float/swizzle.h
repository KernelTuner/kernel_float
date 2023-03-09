#ifndef KERNEL_FLOAT_SWIZZLE_H
#define KERNEL_FLOAT_SWIZZLE_H

#include "storage.h"

namespace kernel_float {

template<typename Output, typename Input, typename Indices, typename = void>
struct vector_swizzle;

template<typename Output, typename Input, size_t... Is>
struct vector_swizzle<Output, Input, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static Output call(const Input& storage) {
        return Output {storage.get(const_index<Is> {})...};
    }
};

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
template<size_t... Is, typename V>
KERNEL_FLOAT_INLINE vector_storage<vector_value_type<V>, sizeof...(Is)>
swizzle(V&& input, index_sequence < Is... >= {}) {
    using Input = into_vector_type<V>;
    using Output = vector_storage<vector_value_type<V>, sizeof...(Is)>;

    return vector_swizzle<Output, Input, index_sequence<Is...>>::call(
        into_vector(std::forward<V>(input)));
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
template<size_t N = 1, typename V>
KERNEL_FLOAT_INLINE vector_storage<vector_value_type<V>, N> first(V&& input) {
    static_assert(N <= vector_size<V>, "N cannot exceed vector size");
    using Indices = make_index_sequence<N>;
    return swizzle(std::forward<V>(input), Indices {});
}

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
template<size_t N = 1, typename V>
KERNEL_FLOAT_INLINE vector_storage<vector_value_type<V>, N> last(V&& input) {
    static_assert(N <= vector_size<V>, "N cannot exceed vector size");
    using Indices = make_index_sequence<N, (vector_size<V> - N)>;
    return swizzle(std::forward<V>(input), Indices {});
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
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> reversed(V&& input) {
    using Input = into_vector_type<V>;
    using Output = Input;
    using Indices = typename detail::reverse_index_sequence_helper<vector_size<V>>::type;

    return swizzle(std::forward<V>(input), Indices {});
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
template<size_t K = 1, typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> rotate_right(V&& input) {
    static constexpr size_t N = vector_size<V>;
    static constexpr size_t I = (N > 0) ? (K % N) : 0;

    using First = index_sequence<I, N - I>;
    using Second = index_sequence<N - I>;
    using Indices = typename detail::concat_index_sequence_helper<First, Second>::type;

    return swizzle(std::forward<V>(input), Indices {});
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
template<size_t K = 1, typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> rotate_left(V&& input) {
    static constexpr size_t N = vector_size<V>;
    static constexpr size_t K_rev = N > 0 ? (N - K % N) : 0;

    return rotate_right<K_rev>(std::forward<V>(input));
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
    using type = vector_storage<
        common_t<vector_value_type<U>, vector_value_type<V>>,
        vector_size<U> + vector_size<V>>;

    KERNEL_FLOAT_INLINE static type call(U&& left, V&& right) {
        return type {left.get(const_index<Is> {})..., right.get(const_index<Js> {})...};
    }
};

template<typename... Ts>
struct recur_concat_helper;

template<typename U>
struct recur_concat_helper<U> {
    using type = U;

    KERNEL_FLOAT_INLINE static U call(U&& input) {
        return output;
    }
};

template<typename U, typename V, typename... Rest>
struct recur_concat_helper<U, V, Rest...> {
    using recur_helper = recur_concat_helper<typename concat_helper<U, V>::type, Rest...>;
    using type = typename recur_helper::type;

    KERNEL_FLOAT_INLINE static type call(U&& left, V&& right, Rest&&... rest) {
        return recur_helper::call(
            concat_helper<U, V>::call(std::forward<U>(left), std::forward<V>(right)),
            std::forward<Rest>(rest)...);
    }
};
}  // namespace detail

template<typename... Vs>
using concat_type = typename detail::recur_concat_helper<into_vector_type<Vs>...>::type;

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
KERNEL_FLOAT_INLINE concat_type<Vs...> concat(Vs&&... inputs) {
    return detail::recur_concat_helper<into_vector_type<Vs>...>::call(
        into_vector<Vs>(std::forward<Vs>(inputs))...);
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_SWIZZLE_H
