#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

#include "storage.h"

namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Input, typename = void>
struct map_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input) {
        return call(fun, input, make_index_sequence<vector_size<Input>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input, index_sequence<Is...>) {
        return Output {fun(input.get(const_index<Is> {}))...};
    }
};

template<typename F, typename R, typename T, size_t N>
struct map_helper<F, vector_compound<R, N>, vector_compound<T, N>> {
    KERNEL_FLOAT_INLINE static vector_compound<R, N>
    call(F fun, const vector_compound<T, N>& input) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return {
            map_helper<F, vector_storage<R, low_size>, vector_storage<T, low_size>>::call(
                fun,
                input.low()),
            map_helper<F, vector_storage<R, high_size>, vector_storage<T, high_size>>::call(
                fun,
                input.high())};
    }
};
}  // namespace detail

template<typename F, typename Input>
using map_type = vector_storage<result_t<F, vector_value_type<Input>>, vector_size<Input>>;

/**
 * Applies ``fun`` to each element from vector ``input`` and returns a new vector with the results.
 * This function is the basis for all unary operators like ``sin`` and ``sqrt``.
 *
 * Example
 * =======
 * ```
 * vector<int, 3> v = {1, 2, 3};
 * vector<int, 3> w = map([](auto i) { return i * 2; }); // 2, 4, 6
 * ```
 */
template<typename F, typename Input, typename Output = map_type<F, Input>>
KERNEL_FLOAT_INLINE Output map(F fun, Input&& input) {
    return detail::map_helper<F, Output, into_vector_type<Input>>::call(fun, into_vector(input));
}

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
template<
    typename Input,
    typename Output,
    typename T = vector_value_type<Input>,
    size_t N = vector_size<Input>,
    typename R = vector_value_type<Output>,
    size_t M = vector_size<Output>>
struct broadcast_helper;

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, Vector, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

template<typename Vector, typename T>
struct broadcast_helper<Vector, Vector, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, Vector&, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(const Vector& input) {
        return input;
    }
};

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, const Vector&, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(const Vector& input) {
        return input;
    }
};

template<typename Input, typename Output, typename T, size_t N>
struct broadcast_helper<Input, Output, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        using F = ops::cast<T, T>;
        return map_helper<F, Output, into_vector_type<Input>>::call(
            F {},
            into_vector(std::forward<Input>(input)));
    }
};

template<typename Output, typename Input, typename T, size_t N>
struct broadcast_helper<Input, Output, T, 1, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {into_vector(std::forward<Input>(input)).get(const_index<0> {})};
    }
};

template<typename Output, typename Input, typename T>
struct broadcast_helper<Input, Output, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {into_vector(std::forward<Input>(input)).get(const_index<0> {})};
    }
};

template<typename Output, typename Input, typename T, typename R>
struct broadcast_helper<Input, Output, T, 1, R, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {
            ops::cast<T, R> {}(into_vector(std::forward<Input>(input)).get(const_index<0> {}))};
    }
};

template<typename Output, typename Input, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, 1, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {
            ops::cast<T, R> {}(into_vector(std::forward<Input>(input)).get(const_index<0> {}))};
    }
};

template<typename Output, typename Input, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, N, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        using F = ops::cast<T, R>;
        return map_helper<F, Output, into_vector_type<Input>>::call(
            F {},
            into_vector(std::forward<Input>(input)));
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
template<typename R, size_t N, typename Input, typename Output = vector_storage<R, N>>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

#ifdef DOXYGEN_SHOULD_SKIP_THIS
template<size_t N, typename Input, typename Output = vector_storage<vector_value_type<Input>, N>>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<typename Output, typename Input>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
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
template<size_t N, typename Input, typename Output = vector_storage<vector_value_type<Input>, N>>
KERNEL_FLOAT_INLINE Output resize(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<typename R, typename Input>
using cast_type = vector_storage<R, vector_size<Input>>;

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
KERNEL_FLOAT_INLINE Output cast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                   \
    namespace ops {                                                             \
    template<typename T>                                                        \
    struct NAME {                                                               \
        KERNEL_FLOAT_INLINE T operator()(T input) {                             \
            return T(EXPR);                                                     \
        }                                                                       \
    };                                                                          \
    }                                                                           \
    template<typename V>                                                        \
    KERNEL_FLOAT_INLINE into_vector_type<V> NAME(V&& input) {                   \
        return map(ops::NAME<vector_value_type<V>> {}, std::forward<V>(input)); \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                                        \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                                   \
    template<typename V>                                                                    \
    KERNEL_FLOAT_INLINE enabled_t<is_vector<V>, into_vector_type<V>> operator OP(V&& vec) { \
        return NAME(std::forward<V>(vec));                                                  \
    }

KERNEL_FLOAT_DEFINE_UNARY_OP(negate, -, -input)
KERNEL_FLOAT_DEFINE_UNARY_OP(bit_not, ~, ~input)
KERNEL_FLOAT_DEFINE_UNARY_OP(logical_not, !, !bool(input))

#define KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME) KERNEL_FLOAT_DEFINE_UNARY(NAME, ::NAME(input))

KERNEL_FLOAT_DEFINE_UNARY_FUN(acos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(abs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(acosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ceil)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cospi)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcx)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp2)
KERNEL_FLOAT_DEFINE_UNARY_FUN(expm1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(fabs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(floor)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ilogb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(lgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(logb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(nearbyint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(normcdf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rcbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(trunc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y0)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(yn)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(round)
KERNEL_FLOAT_DEFINE_UNARY_FUN(signbit)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isinf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isnan)

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
