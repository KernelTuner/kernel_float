#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H

#include "macros.h"

namespace kernel_float {
template<typename T, size_t N = 1>
struct vec;

template<typename F, typename... Args>
using result_t = typename std::result_of<F(Args...)>::type;

template<bool C, typename T = void>
using enabled_t = typename std::enable_if<C, T>::type;

using float32 = float;
using float64 = double;

template<typename T, typename... Args>
static constexpr bool is_constructible = std::is_constructible<T, Args...>::value;

template<size_t I>
struct constant_index {
    using value_type = size_t;
    static constexpr size_t value = I;

    KERNEL_FLOAT_INLINE constexpr operator std::integral_constant<size_t, I>() const noexcept {
        return {};
    }

    KERNEL_FLOAT_INLINE constexpr operator size_t() const noexcept {
        return I;
    }

    KERNEL_FLOAT_INLINE constexpr size_t operator()() const noexcept {
        return I;
    }
};

template<size_t... Is>
using index_sequence = std::integer_sequence<size_t, Is...>;

template<size_t N>
using make_index_sequence = std::make_index_sequence<N>;

using I0 = constant_index<0>;
using I1 = constant_index<1>;
using I2 = constant_index<2>;
using I3 = constant_index<3>;

template<typename... Ts>
struct common_type;

template<typename T>
struct common_type<T> {
    using type = T;
};

template<typename T>
struct common_type<T, T> {
    using type = T;
};

template<typename T, typename U, size_t N>
struct common_type<vec<T, N>, vec<U, N>> {
    using type = vec<typename common_type<T, U>::type, N>;
};

#define KERNEL_FLOAT_DEFINE_COMMON_TYPE(T, U) \
    template<>                                \
    struct common_type<T, U> {                \
        using type = T;                       \
    };                                        \
    template<>                                \
    struct common_type<U, T> {                \
        using type = T;                       \
    };

KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, double)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, float)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, float)
//KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, half)
//KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, half)

#define KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(T, U)       \
    KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed T, signed U) \
    KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned T, unsigned U)

KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, long)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, int)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, int)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(int, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(int, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(short, char)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed long long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed int, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed short, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed char, bool)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned long long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned int, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned short, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned char, bool)

template<typename T, typename U, typename First, typename... Rest>
struct common_type<T, U, First, Rest...> {
    using type = typename common_type<typename common_type<T, U>::type, First, Rest...>::type;
};

template<typename... Args>
using common_t = typename common_type<Args...>::type;

template<typename From, typename To>
static constexpr bool is_implicit_convertible = std::is_same<common_t<From, To>, To>::value;
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_CORE_H
