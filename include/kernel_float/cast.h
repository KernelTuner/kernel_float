#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H

#include "storage.h"

namespace kernel_float {
namespace ops {
template<typename T, typename R>
struct cast {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

template<typename T>
struct cast<T, T> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};
}  // namespace ops

namespace detail {

// Cast a vector of type `Input` to type `Output`. Vectors must have the same size.
// The input vector has value type `T`
// The output vector has value type `R`
template<
    typename Input,
    typename Output,
    typename T = vector_value_type<Input>,
    typename R = vector_value_type<Output>>
struct cast_helper {
    static_assert(vector_size<Input> == vector_size<Output>, "sizes must match");
    static constexpr size_t N = vector_size<Input>;

    KERNEL_FLOAT_INLINE static Output call(const Input& input) {
        return call(input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output call(const Input& input, index_sequence<Is...>) {
        ops::cast<T, R> fun;
        return vector_traits<Output>::create(fun(vector_get<Is>(input))...);
    }
};

// Cast a vector of type `Input` to type `Output`.
// The input vector has value type `T` and size `N`.
// The output vector has value type `R` and size `M`.
template<
    typename Input,
    typename Output,
    typename T = vector_value_type<Input>,
    size_t N = vector_size<Input>,
    typename R = vector_value_type<Output>,
    size_t M = vector_size<Output>>
struct broadcast_helper;

// T[1] => T[1]
template<typename Vector, typename T>
struct broadcast_helper<Vector, Vector, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

// T[N] => T[N]
template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, Vector, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

// T[1] => T[N]
template<typename Output, typename Input, typename T, size_t N>
struct broadcast_helper<Input, Output, T, 1, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return vector_traits<Output>::fill(vector_get<0>(input));
    }
};

// T[1] => T[1], but different vector types
template<typename Output, typename Input, typename T>
struct broadcast_helper<Input, Output, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return vector_traits<Output>::create(vector_get<0>(input));
    }
};

// T[N] => T[N], but different vector types
template<typename Input, typename Output, typename T, size_t N>
struct broadcast_helper<Input, Output, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return cast_helper<Input, Output>::call(input);
    }
};

// T[1] => R[N]
template<typename Output, typename Input, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, 1, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return vector_traits<Output>::fill(ops::cast<T, R> {}(vector_get<0>(input)));
    }
};

// T[1] => R[1]
template<typename Output, typename Input, typename T, typename R>
struct broadcast_helper<Input, Output, T, 1, R, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return vector_traits<Output>::create(ops::cast<T, R> {}(vector_get<0>(input)));
    }
};

// T[N] => R[N]
template<typename Input, typename Output, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, N, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input input) {
        return cast_helper<Input, Output>::call(input);
    }
};
}  // namespace detail

/**
 * Cast the elements of the given vector ``input`` to the given type ``R`` and then widen the
 * vector to length ``N``. The cast may lead to a loss in precision if ``R`` is a smaller data
 * type. Widening is only possible if the input vector has size ``1`` or ``N``, other sizes
 * will lead to a compilation error.
 *
 * Example
 * =======
 * ```
 * vec<int, 1> x = {6};
 * vec<double, 3> y = broadcast<double, 3>(x);
 * vec<float, 3> z = broadcast<float, 3>(y);
 * ```
 */
template<typename R, size_t N, typename Input, typename Output = default_storage_type<R, N>>
KERNEL_FLOAT_INLINE vector<Output> broadcast(Input&& input) {
    return detail::broadcast_helper<into_storage_type<Input>, Output>::call(
        into_storage(std::forward<Input>(input)));
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<
    size_t N,
    typename Input,
    typename Output = default_storage_type<vector_value_type<Input>, N>>
KERNEL_FLOAT_INLINE vector<Output> broadcast(Input&& input) {
    return detail::broadcast_helper<into_storage_type<Input>, Output>::call(
        into_storage(std::forward<Input>(input)));
}

template<typename Output, typename Input>
KERNEL_FLOAT_INLINE vector<Output> broadcast(Input&& input) {
    return detail::broadcast_helper<into_storage_type<Input>, Output>::call(
        into_storage(std::forward<Input>(input)));
}
#endif

/**
 * Widen the given vector ``input`` to length ``N``. Widening is only possible if the input vector
 * has size ``1`` or ``N``, other sizes will lead to a compilation error.
 *
 * Example
 * =======
 * ```
 * vec<int, 1> x = {6};
 * vec<int, 3> y = resize<3>(x);
 * ```
 */
template<
    size_t N,
    typename Input,
    typename Output = default_storage_type<vector_value_type<Input>, N>>
KERNEL_FLOAT_INLINE vector<Output> resize(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<typename R, typename Input>
using cast_type = default_storage_type<R, vector_size<Input>>;

/**
 * Cast the elements of given vector ``input`` to the given type ``R``. Note that this cast may
 * lead to a loss in precision if ``R`` is a smaller data type.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> x = {1.0f, 2.0f, 3.0f};
 * vec<double, 3> y = cast<double>(x);
 * vec<int, 3> z = cast<int>(x);
 * ```
 */
template<typename R, typename Input, typename Output = cast_type<R, Input>>
KERNEL_FLOAT_INLINE vector<Output> cast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_CAST_H
