#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H

#include "macros.h"

namespace kernel_float {

template<size_t... Is>
struct index_sequence {
    static constexpr size_t size = sizeof...(Is);
};

namespace detail {
template<size_t N>
struct make_index_sequence_impl {};

// Benchmarks show that it is much faster to predefine all possible index sequences instead of doing something
// recursive with variadic templates.
#define KERNEL_FLOAT_INDEX_SEQ(N, ...)            \
    template<>                                    \
    struct make_index_sequence_impl<N> {          \
        using type = index_sequence<__VA_ARGS__>; \
    };

KERNEL_FLOAT_INDEX_SEQ(0)
KERNEL_FLOAT_INDEX_SEQ(1, 0)
KERNEL_FLOAT_INDEX_SEQ(2, 0, 1)
KERNEL_FLOAT_INDEX_SEQ(3, 0, 1, 2)
KERNEL_FLOAT_INDEX_SEQ(4, 0, 1, 2, 3)
KERNEL_FLOAT_INDEX_SEQ(5, 0, 1, 2, 3, 4)
KERNEL_FLOAT_INDEX_SEQ(6, 0, 1, 2, 3, 4, 5)
KERNEL_FLOAT_INDEX_SEQ(7, 0, 1, 2, 3, 4, 5, 6)
KERNEL_FLOAT_INDEX_SEQ(8, 0, 1, 2, 3, 4, 5, 6, 7)
KERNEL_FLOAT_INDEX_SEQ(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
KERNEL_FLOAT_INDEX_SEQ(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
KERNEL_FLOAT_INDEX_SEQ(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
KERNEL_FLOAT_INDEX_SEQ(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
KERNEL_FLOAT_INDEX_SEQ(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
KERNEL_FLOAT_INDEX_SEQ(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
KERNEL_FLOAT_INDEX_SEQ(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
KERNEL_FLOAT_INDEX_SEQ(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
KERNEL_FLOAT_INDEX_SEQ(17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

}  // namespace detail

template<size_t N>
using make_index_sequence = typename detail::make_index_sequence_impl<N>::type;

namespace detail {
template<typename T>
struct decay_impl {
    using type = T;
};

template<typename T>
struct decay_impl<const T> {
    using type = T;
};

template<typename T>
struct decay_impl<const T&> {
    using type = T;
};

template<typename T>
struct decay_impl<T&> {
    using type = T;
};

template<typename T>
struct decay_impl<T&&> {
    using type = T;
};
}  // namespace detail

template<typename T>
using decay_t = typename detail::decay_impl<T>::type;

template<typename A, typename B>
struct promote_type {};

template<typename T>
struct promote_type<T, T> {
    using type = T;
};

template<typename T>
struct promote_type<void, T> {
    using type = T;
};

template<typename T>
struct promote_type<T, void> {
    using type = T;
};

template<>
struct promote_type<void, void> {
    using type = void;
};

#define KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, U) \
    template<>                                  \
    struct promote_type<T, U> {                 \
        using type = T;                         \
    };                                          \
    template<>                                  \
    struct promote_type<U, T> {                 \
        using type = T;                         \
    };

// T + bool becomes T
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(char, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed char, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed short, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed int, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed long, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed long long, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned char, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned short, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned int, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned long, bool)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned long long, bool)

KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, float)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, float)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, double)

#define KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(T)                \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, char)               \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed char)        \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed short)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed int)         \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed long)        \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed long long)   \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned char)      \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned short)     \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned int)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned long)      \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned long long) \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, bool)

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(float)
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(double)
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(long double)

#define KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(T, U)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed T, signed U) \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned T, unsigned U)

KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(short, char)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(int, char)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(int, short)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, char)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, short)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, int)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, char)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, short)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, int)
KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, long)

template<typename T>
struct promote_type<T*, T*> {
    using type = T*;
};

#define KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(I) \
    template<typename T>                        \
    struct promote_type<T*, I> {                \
        using type = T*;                        \
    };                                          \
    template<typename T>                        \
    struct promote_type<I, T*> {                \
        using type = T*;                        \
    };

KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(char)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed char)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed short)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed int)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed long)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed long long)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned char)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned short)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned int)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned long)
KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned long long)

// half precision
//    KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(half)
//    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half, bool)
//    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, half)
//    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, half)
//    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, half)

namespace detail {
template<typename... Ts>
struct multi_promote_type;

template<typename T>
struct multi_promote_type<T> {
    using type = T;
};

template<typename A, typename B>
struct multi_promote_type<A, B>: promote_type<A, B> {};

template<typename A, typename B, typename C, typename... Rest>
struct multi_promote_type<A, B, C, Rest...>:
    multi_promote_type<typename promote_type<A, B>::type, C, Rest...> {};

}  // namespace detail

template<typename... Ts>
using promote_t = typename detail::multi_promote_type<decay_t<Ts>...>::type;

namespace detail {

template<typename A, typename B>
struct is_same_type_impl {
    static constexpr bool value = false;
};

template<typename A>
struct is_same_type_impl<A, A> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename A, typename B>
static constexpr bool is_same_type = detail::is_same_type_impl<A, B>::value;

namespace detail {
template<typename From, typename To, typename Common = To>
struct is_implicit_convertible_impl {
    static constexpr bool value = false;
};

// This has to be done using SFINAE since `promote_type<From, To>` might actually be undefined if `From` is not
// implicitly convertible to `To`.
template<typename From, typename To>
struct is_implicit_convertible_impl<From, To, typename promote_type<From, To>::type> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename From, typename To>
static constexpr bool is_implicit_convertible =
    detail::is_implicit_convertible_impl<decay_t<From>, decay_t<To>>::value;

namespace detail {
template<typename T>
KERNEL_FLOAT_INLINE T& declval() {
    while (1)
        ;
}
}  // namespace detail

namespace detail {
template<bool, typename T>
struct enable_if_impl {};

template<typename T>
struct enable_if_impl<true, T> {
    using type = T;
};
}  // namespace detail

template<bool C, typename T = void>
using enable_if_t = typename detail::enable_if_impl<C, T>::type;

KERNEL_FLOAT_INLINE
constexpr size_t round_up_to_power_of_two(size_t n) {
    size_t result = 1;
    while (result < n) {
        result *= 2;
    }
    return result;
}

}  // namespace kernel_float

#endif
