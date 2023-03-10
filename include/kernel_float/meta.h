#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H

#include "macros.h"

namespace kernel_float {

template<size_t I>
struct const_index {
    static constexpr size_t value = I;

    KERNEL_FLOAT_INLINE constexpr operator size_t() const noexcept {
        return I;
    }
};

template<size_t... Is>
struct index_sequence {};

namespace detail {
template<size_t N, size_t X, size_t... Is>
struct make_index_sequence_helper: make_index_sequence_helper<N - 1, X + N - 1, Is...> {};

template<size_t... Is, size_t X>
struct make_index_sequence_helper<0, X, Is...> {
    using type = index_sequence<Is...>;
};

}  // namespace detail

template<size_t N, size_t Offset = 0>
using make_index_sequence = typename detail::make_index_sequence_helper<N, Offset>::type;

namespace detail {
template<typename T>
struct decay_helper {
    using type = T;
};

template<typename T>
struct decay_helper<const T> {
    using type = T;
};

template<typename T>
struct decay_helper<const T&> {
    using type = T;
};

template<typename T>
struct decay_helper<T&> {
    using type = T;
};

template<typename T>
struct decay_helper<T&&> {
    using type = T;
};
}  // namespace detail

template<typename T>
using decay_t = typename detail::decay_helper<T>::type;

template<typename T, typename U>
struct common_type;

template<typename T>
struct common_type<T, T> {
    using type = T;
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

KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, bool)

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

namespace detail {
template<typename... Ts>
struct common_type_helper;

template<typename T>
struct common_type_helper<T> {
    using type = T;
};

template<typename T, typename U, typename... Rest>
struct common_type_helper<T, U, Rest...>:
    common_type_helper<typename common_type<T, U>::type, Rest...> {};
}  // namespace detail

template<typename... Ts>
using common_t = typename detail::common_type_helper<decay_t<Ts>...>::type;

namespace detail {
template<size_t...>
struct common_size_helper;

template<>
struct common_size_helper<> {
    static constexpr size_t value = 1;
};

template<size_t N>
struct common_size_helper<N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<N, N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<N, 1> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<1, N> {
    static constexpr size_t value = N;
};

template<>
struct common_size_helper<1, 1> {
    static constexpr size_t value = 1;
};
}  // namespace detail

template<size_t... Ns>
static constexpr size_t common_size = detail::common_size_helper<Ns...>::value;

namespace detail {

template<typename From, typename To, typename Common = To>
struct is_implicit_convertible_helper {
    static constexpr bool value = false;
};

template<typename From, typename To>
struct is_implicit_convertible_helper<From, To, typename common_type<From, To>::type> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename From, typename To>
static constexpr bool is_implicit_convertible =
    detail::is_implicit_convertible_helper<decay_t<From>, decay_t<To>>::value;

namespace detail {
template<typename T>
KERNEL_FLOAT_INLINE T& declval() {
    while (1)
        ;
}
}  // namespace detail

template<typename F, typename... Args>
using result_t = decltype((detail::declval<F>())(detail::declval<Args>()...));

namespace detail {
template<bool, typename T>
struct enabled_helper {};

template<typename T>
struct enabled_helper<true, T> {
    using type = T;
};
}  // namespace detail

template<bool C, typename T = void>
using enabled_t = typename detail::enabled_helper<C, T>::type;

}  // namespace kernel_float

#endif
