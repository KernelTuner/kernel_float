#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H

#include "base.h"
#include "unops.h"

namespace kernel_float {

namespace detail {
/**
 * Convert vector of element type `T` and extent type `E` to vector of element type `T2` and extent type `E2`.
 *  Specialization exist for the cases where `T==T2` and/or `E==E2`.
 */
template<typename T, typename E, typename T2, typename E2, RoundingMode M = RoundingMode::ANY>
struct convert_impl {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, E2::value> call(vector_storage<T, E::value> input) {
        using F = ops::cast<T, T2, M>;
        vector_storage<T2, E::value> intermediate;
        detail::apply_impl<F, E::value, T2, T>::call(F {}, intermediate.data(), input.data());
        return detail::broadcast_impl<T2, E, E2>::call(intermediate);
    }
};

// T == T2, E == E2
template<typename T, typename E, RoundingMode M>
struct convert_impl<T, E, T, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, E::value> call(vector_storage<T, E::value> input) {
        return input;
    }
};

// T == T2, E != E2
template<typename T, typename E, typename E2, RoundingMode M>
struct convert_impl<T, E, T, E2, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, E2::value> call(vector_storage<T, E::value> input) {
        return detail::broadcast_impl<T, E, E2>::call(input);
    }
};

// T != T2, E == E2
template<typename T, typename E, typename T2, RoundingMode M>
struct convert_impl<T, E, T2, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, E::value> call(vector_storage<T, E::value> input) {
        using F = ops::cast<T, T2, M>;

        vector_storage<T2, E::value> result;
        detail::apply_impl<F, E::value, T2, T>::call(F {}, result.data(), input.data());
        return result;
    }
};
}  // namespace detail

template<typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector_storage<R, N> convert_storage(const V& input, extent<N> new_size = {}) {
    return detail::convert_impl<vector_value_type<V>, vector_extent_type<V>, R, extent<N>, M>::call(
        into_vector_storage(input));
}

/**
 * Cast the values of the given input vector to type `R` and then broadcast the result to the given size `N`.
 *
 * Example
 * =======
 * ```
 * int a = 5;
 * vec<float, 3> x = convert<float, 3>(a);  // returns [5.0f, 5.0f, 5.0f]
 *
 * float b = 5.0f;
 * vec<float, 3> x = convert<float, 3>(b);  // returns [5.0f, 5.0f, 5.0f]
 *
 * vec<int, 3> c = {1, 2, 3};
 * vec<float, 3> x = convert<float, 3>(c);  // returns [1.0f, 2.0f, 3.0f]
 * ```
 */
template<typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, extent<N>> convert(const V& input, extent<N> new_size = {}) {
    return convert_storage(input);
}

template<typename T, RoundingMode M = RoundingMode::ANY>
struct AssignConversionProxy {
    KERNEL_FLOAT_INLINE
    explicit AssignConversionProxy(T* ptr) : ptr_(ptr) {}

    template<typename U>
    KERNEL_FLOAT_INLINE AssignConversionProxy& operator=(U&& values) {
        *ptr_ = detail::convert_impl<
            vector_value_type<U>,
            vector_extent_type<U>,
            vector_value_type<T>,
            vector_extent_type<T>,
            M>::call(into_vector_storage(values));

        return *this;
    }

  private:
    T* ptr_;
};

/**
 * Takes a vector reference and gives back a helper object. This object allows you to assign
 * a vector of a different type to another vector while perofrming implicit type converion.
 *
 * For example, if `x = expression;` does not compile because `x` and `expression` are
 * different vector types, you can use `cast_to(x) = expression;` to make it work.
 *
 * Example
 * =======
 * ```
 * vec<float, 2> x;
 * vec<double, 2> y = {1.0, 2.0};
 * cast_to(x) = y;  // Normally, `x = y;` would give an error, but `cast_to` fixes that.
 * ```
 */
template<typename T, RoundingMode M = RoundingMode::ANY>
KERNEL_FLOAT_INLINE AssignConversionProxy<T, M> cast_to(T& input) {
    return AssignConversionProxy<T, M>(&input);
}

/**
 * Returns a vector containing `N` copies of `value`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = fill<3>(42); // returns [42, 42, 42]
 * ```
 */
template<size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> fill(T value = {}, extent<N> = {}) {
    vector_storage<T, 1> input = {value};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector containing `N` copies of `T(0)`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = zeros<int, 3>(); // returns [0, 0, 0]
 * ```
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> zeros(extent<N> = {}) {
    vector_storage<T, 1> input = {T {}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector containing `N` copies of `T(1)`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = ones<int, 3>(); // returns [1, 1, 1]
 * ```
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> ones(extent<N> = {}) {
    vector_storage<T, 1> input = {T {1}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector filled with `value` having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = fill_like(a, 42); // returns [42, 42, 42]
 * ```
 */
template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> fill_like(const V&, T value) {
    return fill(value, E {});
}

/**
 * Returns a vector filled with zeros having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = zeros_like(a); // returns [0, 0, 0]
 * ```
 */
template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> zeros_like(const V& = {}) {
    return zeros<T>(E {});
}

/**
 * Returns a vector filled with ones having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = ones_like(a); // returns [1, 1, 1]
 * ```
 */
template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> ones_like(const V& = {}) {
    return ones<T>(E {});
}

}  // namespace kernel_float

#endif
