//================================================================================
// this file has been auto-generated, do not modify its contents!
// date: 2023-07-25 14:50:15.560873
// git hash: df48350ff5f4362e8220188c09f48c37ba9d0335
//================================================================================

#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __CUDACC__
#define KERNEL_FLOAT_CUDA (1)

#ifdef __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE    __forceinline__ __device__
#define KERNEL_FLOAT_IS_DEVICE (1)
#define KERNEL_FLOAT_IS_HOST   (0)
#define KERNEL_FLOAT_CUDA_ARCH (__CUDA_ARCH__)
#else
#define KERNEL_FLOAT_INLINE    __forceinline__ __host__
#define KERNEL_FLOAT_IS_DEVICE (0)
#define KERNEL_FLOAT_IS_HOST   (1)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif
#else
#define KERNEL_FLOAT_INLINE    inline
#define KERNEL_FLOAT_CUDA      (0)
#define KERNEL_FLOAT_IS_HOST   (1)
#define KERNEL_FLOAT_IS_DEVICE (0)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
#define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
#define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
#define KERNEL_FLOAT_FP8_AVAILABLE (0)
#endif

#define KERNEL_FLOAT_ASSERT(expr) \
    do {                          \
    } while (0)
#define KERNEL_FLOAT_UNREACHABLE __builtin_unreachable()

#endif  //KERNEL_FLOAT_MACROS_H
#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H



namespace kernel_float {

template<size_t... Is>
struct index_sequence {
    static constexpr size_t size = sizeof...(Is);
};

namespace detail {
template<size_t N>
struct make_index_sequence_helper {};

// Benchmarks show that it is much faster to predefine all possible index sequences instead of doing something
// recursive with variadic templates.
#define KERNEL_FLOAT_INDEX_SEQ(N, ...)            \
    template<>                                    \
    struct make_index_sequence_helper<N> {        \
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
using make_index_sequence = typename detail::make_index_sequence_helper<N>::type;

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

template<typename A, typename B>
struct promote_type;

template<typename T>
struct promote_type<T, T> {
    using type = T;
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
struct is_same_helper {
    static constexpr bool value = false;
};

template<typename A>
struct is_same_helper<A, A> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename A, typename B>
static constexpr bool is_same = detail::is_same_helper<A, B>::value;

namespace detail {
template<typename From, typename To, typename Common = To>
struct is_implicit_convertible_helper {
    static constexpr bool value = false;
};

template<typename From, typename To>
struct is_implicit_convertible_helper<From, To, typename promote_type<From, To>::type> {
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
#ifndef KERNEL_FLOAT_BASE
#define KERNEL_FLOAT_BASE




namespace kernel_float {

template<typename T, size_t N, size_t Alignment = alignof(T)>
struct alignas(Alignment) array {
    KERNEL_FLOAT_INLINE
    T* data() {
        return items_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items_;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        return items_[i];
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        return items_[i];
    }

    T items_[N];
};

template<typename T, size_t Alignment>
struct array<T, 1, Alignment> {
    KERNEL_FLOAT_INLINE
    array(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    operator T() const {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    T* data() {
        return &value_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return &value_;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t) {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t) const {
        return value_;
    }

    T value_;
};

template<typename T, size_t Alignment>
struct array<T, 0, Alignment> {
    KERNEL_FLOAT_INLINE
    T* data() {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        while (true)
            ;
    }
};

template<size_t N>
using ndindex = array<size_t, N>;

KERNEL_FLOAT_INLINE
static constexpr size_t compute_max_alignment(size_t total_size, size_t min_align) {
    if (total_size % 32 == 0 || min_align >= 32) {
        return 32;
    } else if (total_size % 16 == 0 || min_align == 16) {
        return 16;
    } else if (total_size % 8 == 0 || min_align == 8) {
        return 8;
    } else if (total_size % 4 == 0 || min_align == 4) {
        return 4;
    } else if (total_size % 2 == 0 || min_align == 2) {
        return 2;
    } else {
        return 1;
    }
}

template<typename T, size_t N>
using tensor_storage = array<T, N, compute_max_alignment(sizeof(T) * N, alignof(T))>;

template<typename T, typename D, template<typename, size_t> class S = tensor_storage>
struct tensor;

template<size_t... Ns>
struct extents;

template<>
struct extents<> {
    static constexpr size_t rank = 0;
    static constexpr size_t volume = 1;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<0>) {
        return 0;
    }

    KERNEL_FLOAT_INLINE
    static ndindex<0> unravel_index(size_t i) {
        return {};
    }
};

template<size_t N>
struct extents<N> {
    static constexpr size_t rank = 1;
    static constexpr size_t volume = N;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<1> ind) {
        return ind[0];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<1> unravel_index(size_t i) {
        return {i};
    }
};

template<size_t N, size_t M>
struct extents<N, M> {
    static constexpr size_t rank = 2;
    static constexpr size_t volume = N * M;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : axis == 1 ? M : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return axis == 0 ? M : 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<2> x) {
        return x[0] * M + x[1];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<2> unravel_index(size_t i) {
        return {i / M, i % M};
    }
};

template<size_t N, size_t M, size_t K>
struct extents<N, M, K> {
    static constexpr size_t rank = 3;
    static constexpr size_t volume = N * M * K;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : axis == 1 ? M : axis == 2 ? K : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return axis == 0 ? M * K  //
            : axis == 1  ? K  //
                         : 1;  //
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<3> x) {
        return (x[0] * M + x[1]) * K + x[2];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<3> unravel_index(size_t i) {
        return {i / (K * M), (i / K) % M, i % K};
    }
};

template<typename T>
struct into_tensor_traits {
    using type = tensor<T, extents<>>;

    KERNEL_FLOAT_INLINE
    static type call(const T& input) {
        return tensor_storage<T, 1> {input};
    }
};

template<typename V>
struct into_tensor_traits<const V> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(const V input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<V&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(V& input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<const V&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(const V& input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<V&&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(V&& input) {
        return into_tensor_traits<V>::call(std::move(input));
    }
};

template<typename T, typename D, template<typename, size_t> class S>
struct into_tensor_traits<tensor<T, D, S>> {
    using type = tensor<T, D>;

    KERNEL_FLOAT_INLINE
    static type call(const tensor<T, D, S>& input) {
        return input;
    }
};

template<typename T, size_t N, size_t A>
struct into_tensor_traits<array<T, N, A>> {
    using type = tensor<T, extents<N>>;

    KERNEL_FLOAT_INLINE
    static type call(const array<T, N, A>& input) {
        return input;
    }
};

template<typename V>
struct tensor_traits;

template<typename T, typename D, template<typename, size_t> class S>
struct tensor_traits<tensor<T, D, S>> {
    using value_type = T;
    using extents_type = D;
    using storage_type = S<T, D::volume>;
};

template<typename V>
using into_tensor_type = typename into_tensor_traits<V>::type;

template<typename V>
using tensor_extents = typename tensor_traits<into_tensor_type<V>>::extents_type;

template<typename V>
static constexpr size_t tensor_rank = tensor_extents<V>::rank;

template<typename V>
static constexpr size_t tensor_volume = tensor_extents<V>::volume;

template<typename V>
using tensor_value_type = typename tensor_traits<into_tensor_type<V>>::value_type;

template<typename V>
using tensor_storage_type = tensor_storage<tensor_value_type<V>, tensor_volume<V>>;

template<typename... Vs>
using promoted_tensor_value_type =
    promote_t<typename tensor_traits<into_tensor_type<Vs>>::value_type...>;

template<typename V>
KERNEL_FLOAT_INLINE into_tensor_type<V> into_tensor(V&& input) {
    return into_tensor_traits<V>::call(std::forward<V>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE tensor_storage_type<V> into_tensor_storage(V&& input) {
    return into_tensor_traits<V>::call(std::forward<V>(input)).storage();
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_COMPLEX_TYPE_H
#define KERNEL_FLOAT_COMPLEX_TYPE_H




namespace kernel_float {

template<typename T>
struct alignas(2 * alignof(T)) complex_type_storage {
    T re;
    T im;
};

template<typename T>
struct complex_type: complex_type_storage<T> {
    using base_type = complex_type_storage<T>;

    template<typename T2>
    KERNEL_FLOAT_INLINE complex_type(complex_type<T2> that) : base_type(that.real(), that.imag()) {}

    KERNEL_FLOAT_INLINE
    complex_type(T real = {}, T imag = {}) : base_type(real, im) {}

    KERNEL_FLOAT_INLINE
    T real() const {
        return re;
    }

    KERNEL_FLOAT_INLINE
    T imag() const {
        return im;
    }

    KERNEL_FLOAT_INLINE
    T norm() const {
        return re * re + im * im;
    }

    KERNEL_FLOAT_INLINE
    complex_type conj() const {
        return {re, -im};
    }
};

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> v) {
    return v;
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, complex_type<T> b) {
    return {a.real() + b.real(), a.imag() + b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(T a, complex_type<T> b) {
    return {a + b.real(), b.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, T b) {
    return {a.real() + b, a.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, complex_type<T> b) {
    return (a = a + b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, T b) {
    return (a = a + b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> v) {
    return {-v.real(), -v.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, complex_type<T> b) {
    return {
        a.real() - b.real(), a.imag() - b.imag()
    }
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(T a, complex_type<T> b) {
    return {
        a - b.real(), -b.imag()
    }
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, T b) {
    return {
        a.real() - b, a.imag()
    }
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, complex_type<T> b) {
    return (a = a - b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, T b) {
    return (a = a - b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, complex_type<T> b) {
    return {
        a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()
    }
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, T b) {
    return {a.real() * b, a.imag() * b};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator*=(complex_type<T>& a, complex_type<T> b) {
    return (a = a * b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator*=(complex_type<T>& a, T b) {
    return (a = a * b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(T a, complex_type<T> b) {
    return {
        a * b.real(),
        a * b.imag(),
    };
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, complex_type<T> b) {
    T normi = 1 / b.norm();

    return {
        (a.real() * b.real() + a.imag() * b.imag()) * normi,
        (a.imag() * b.real() - a.real() * b.imag()) * normi};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, T b) {
    return {a.real() * (1 / b), a.imag() * (1 / b)};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(T a, complex_type<T> b) {
    T normi = 1 / b.norm();

    return {a * b.real() * normi, -a * b.imag() * normi};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator/=(complex_type<T>& a, complex_type<T> b) {
    return (a = a / b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator/=(complex_type<T>& a, T b) {
    return (a = a / b);
}

template<typename T>
KERNEL_FLOAT_INLINE T real(complex_type<T> v) {
    return v.real();
}

template<typename T>
KERNEL_FLOAT_INLINE T imag(complex_type<T> v) {
    return v.real();
}

template<typename T>
KERNEL_FLOAT_INLINE T abs(complex_type<T> v) {
    return hypot(v.real(), v.imag());
}

template<typename T>
KERNEL_FLOAT_INLINE T arg(complex_type<T> v) {
    return atan2(v.imag(), v.real());
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> sqrt(complex_type<T> v) {
    T radius = abs(v);
    T cosA = v.real() / radius;

    complex_type<T> out = {
        sqrt(radius * (cosA + T(1)) * T(.5)),
        sqrt(radius * (T(1) - cosA) * T(.5))};

    // signbit should be false if x.y is negative
    if (v.imag() < 0) {
        out = complex_type<T> {out.real, -out.im};
    }

    return out;
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> norm(complex_type<T> v) {
    return v.real() * v.real() + v.imag() * v.imag();
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> conj(complex_type<T> v) {
    return {v.real(), -v.imag()};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> exp(complex_type<T> v) {
    // TODO: Handle nan and inf correctly
    T e = exp(v.real());
    T a = v.imag();
    return complex_type<T>(e * cos(a), e * sin(a));
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> log(complex_type<T> v) {
    return {log(abs(v)), arg(v)};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, T b) {
    return exp(a * log(b));
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, complex_type<T> b) {
    return exp(a * log(b));
}

template<typename L, typename R>
struct promote_type<complex_type<L>, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

template<typename L, typename R>
struct promote_type<complex_type<L>, R> {
    using type = complex_type<promote_t<L, R>>;
};

template<typename L, typename R>
struct promote_type<L, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H



namespace kernel_float {
namespace detail {

template<typename F, size_t N, typename Output, typename... Args>
struct apply_impl {
    KERNEL_FLOAT_INLINE static tensor_storage<Output, N>
    call(F fun, const tensor_storage<Args, N>&... inputs) {
        tensor_storage<Output, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result[i] = fun(inputs[i]...);
        }

        return result;
    }
};
}  // namespace detail

template<typename F, typename V>
using map_type = tensor<result_t<F, tensor_value_type<V>>, tensor_extents<V>>;

template<typename F, typename V>
KERNEL_FLOAT_INLINE map_type<F, V> map(F fun, const V& input) {
    using Input = tensor_value_type<V>;
    using Output = result_t<F, Input>;
    return detail::apply_impl<F, tensor_volume<V>, Output, Input>::call(
        fun,
        into_tensor(input).storage());
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                      \
    namespace ops {                                                \
    template<typename T>                                           \
    struct NAME {                                                  \
        KERNEL_FLOAT_INLINE T operator()(T input) {                \
            return T(EXPR);                                        \
        }                                                          \
    };                                                             \
    }                                                              \
    template<typename V>                                           \
    KERNEL_FLOAT_INLINE into_tensor_type<V> NAME(const V& input) { \
        using F = ops::NAME<tensor_value_type<V>>;                 \
        return map(F {}, input);                                   \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                        \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                   \
    template<typename T, typename D>                                        \
    KERNEL_FLOAT_INLINE tensor<T, D> operator OP(const tensor<T, D>& vec) { \
        return NAME(vec);                                                   \
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

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_DEFINE_UNARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_UNARY(FUN_NAME, ops::OP_NAME<T> {}(input))   \
    namespace ops {                                                  \
    template<>                                                       \
    struct OP_NAME<float> {                                          \
        KERNEL_FLOAT_INLINE float operator()(float input) {          \
            return FLOAT_FUN(input);                                 \
        }                                                            \
    };                                                               \
    }
#else
#define KERNEL_FLOAT_DEFINE_UNARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_UNARY(FUN_NAME, ops::OP_NAME<T> {}(input))
#endif

KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_exp, exp, __expf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_log, log, __logf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_cos, cos, __cosf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_sin, sin, __sinf)
KERNEL_FLOAT_DEFINE_UNARY_FAST(fast_tan, tan, __tanf)

enum struct RoundingMode { ANY, DOWN, UP, NEAREST, TOWARD_ZERO };

namespace ops {
template<typename T, typename R, RoundingMode m = RoundingMode::ANY, typename = void>
struct cast;

template<typename T, typename R>
struct cast<T, R, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

template<typename T, RoundingMode m>
struct cast<T, T, m> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};

template<typename T>
struct cast<T, T, RoundingMode::ANY> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};
}  // namespace ops

template<typename R, RoundingMode Mode = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE tensor<R, tensor_extents<V>> cast(const V& input) {
    using F = ops::cast<tensor_value_type<V>, R, Mode>;
    return map(F {}, input);
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H




namespace kernel_float {
namespace detail {

template<size_t N, size_t M>
struct unify_dimension_helper;

template<>
struct unify_dimension_helper<1, 1> {
    static constexpr size_t value = 1;
};

template<size_t N>
struct unify_dimension_helper<N, N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct unify_dimension_helper<N, 1> {
    static constexpr size_t value = N;
};

template<size_t N>
struct unify_dimension_helper<1, N> {
    static constexpr size_t value = N;
};

template<typename A, typename B>
struct unify_extents_helper;

template<size_t... Ns, size_t... Ms>
struct unify_extents_helper<extents<Ns...>, extents<Ms...>> {
    using type = extents<unify_dimension_helper<Ns, Ms>::value...>;
};

template<typename E, size_t N, typename = void>
struct extents_to_rank {
    using type = E;
};

template<size_t... Ns, size_t N>
struct extents_to_rank<extents<Ns...>, N, enabled_t<(sizeof...(Ns) < N)>>:
    extents_to_rank<extents<1, Ns...>, N> {};

template<typename A, typename B>
struct broadcast_extents_helper {
    using type = typename unify_extents_helper<
        typename extents_to_rank<A, B::rank>::type,  //
        typename extents_to_rank<B, A::rank>::type  //
        >::type;
};

template<typename E>
struct broadcast_extents_helper<E, E> {
    using type = E;
};

}  // namespace detail

template<typename A, typename B>
using broadcast_extents = typename detail::broadcast_extents_helper<A, B>::type;

template<typename A, typename B>
using broadcast_tensor_extents = broadcast_extents<tensor_extents<A>, tensor_extents<B>>;

template<typename From, typename To>
static constexpr bool is_broadcastable = is_same<broadcast_extents<From, To>, To>;

template<typename V, typename To>
static constexpr bool is_tensor_broadcastable = is_broadcastable<tensor_extents<V>, To>;

namespace detail {

template<typename E, typename IS, typename OS>
struct copy_helper;

template<typename IS, typename OS>
struct copy_helper<extents<>, IS, OS> {
    template<typename T>
    KERNEL_FLOAT_INLINE static void call(T* output, const T* input) {
        ndindex<0> x;
        size_t input_index = IS::call(x);
        size_t output_index = OS::call(x);
        output[output_index] = input[input_index];
    }
};

template<size_t N, typename IS, typename OS>
struct copy_helper<extents<N>, IS, OS> {
    template<typename T>
    KERNEL_FLOAT_INLINE static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            ndindex<1> x = {i};
            size_t input_index = IS::call(x);
            size_t output_index = OS::call(x);
            output[output_index] = input[input_index];
        }
    }
};

template<size_t N, size_t M, typename IS, typename OS>
struct copy_helper<extents<N, M>, IS, OS> {
    template<typename T>
    KERNEL_FLOAT_INLINE static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                ndindex<2> x = {i, j};
                size_t input_index = IS::call(x);
                size_t output_index = OS::call(x);
                output[output_index] = input[input_index];
            }
        }
    }
};

template<size_t N, size_t M, size_t K, typename IS, typename OS>
struct copy_helper<extents<N, M, K>, IS, OS> {
    template<typename T>
    KERNEL_FLOAT_INLINE static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                for (size_t k = 0; k < K; k++) {
                    ndindex<3> x = {i, j, k};
                    size_t input_index = IS::call(x);
                    size_t output_index = OS::call(x);
                    output[output_index] = input[input_index];
                }
            }
        }
    }
};

template<typename E>
struct strides_helper;

template<>
struct strides_helper<extents<>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<0>) {
        return 0;
    }
};

template<size_t N>
struct strides_helper<extents<N>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<1> x) {
        return (N != 1 ? x[0] : 0);
    }
};

template<size_t N, size_t M>
struct strides_helper<extents<N, M>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<2> x) {
        return (N != 1 ? x[0] * M : 0) +  //
            (M != 1 ? x[1] : 0);
    }
};

template<size_t N, size_t M, size_t K>
struct strides_helper<extents<N, M, K>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<3> x) {
        return (N != 1 ? x[0] * M * K : 0) +  //
            (M != 1 ? x[1] * K : 0) +  //
            (K != 1 ? x[2] : 0);
    }
};

template<typename T, typename From, typename To>
struct broadcast_impl {
    KERNEL_FLOAT_INLINE static tensor_storage<T, To::volume>
    call(tensor_storage<T, From::volume> input) {
        static_assert(is_broadcastable<From, To>, "cannot broadcast to required shape");
        using IS = strides_helper<typename extents_to_rank<From, To::rank>::type>;
        using OS = strides_helper<To>;

        tensor_storage<T, To::volume> output;
        copy_helper<To, IS, OS>::call(output.data(), input.data());
        return output;
    }
};

template<typename T, typename E>
struct broadcast_impl<T, E, E> {
    KERNEL_FLOAT_INLINE static tensor_storage<T, E::volume>
    call(tensor_storage<T, E::volume> input) {
        return input;
    }
};

}  // namespace detail

template<size_t... Ns, typename V>
KERNEL_FLOAT_INLINE tensor<tensor_value_type<V>, extents<Ns...>>
broadcast(const V& input, extents<Ns...> new_extents = {}) {
    using T = tensor_value_type<V>;
    return detail::broadcast_impl<T, tensor_extents<V>, extents<Ns...>>::call(
        into_tensor(input).storage());
}

template<typename V, typename R>
KERNEL_FLOAT_INLINE tensor<tensor_value_type<V>, tensor_extents<R>>
broadcast_like(const V& input, const R&) {
    using T = tensor_value_type<V>;
    return detail::broadcast_impl<T, tensor_extents<V>, tensor_extents<R>>::call(
        into_tensor(input).storage());
}

template<size_t... Ns, typename T>
KERNEL_FLOAT_INLINE tensor<T, extents<Ns...>> fill(T value = {}, extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {value};
    return detail::broadcast_impl<T, extents<>, extents<Ns...>>::call(input);
}

template<typename T, size_t... Ns>
KERNEL_FLOAT_INLINE tensor<T, extents<Ns...>> zeros(extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {T {}};
    return detail::broadcast_impl<T, extents<>, extents<Ns...>>::call(input);
}

template<typename T, size_t... Ns>
KERNEL_FLOAT_INLINE tensor<T, extents<Ns...>> ones(extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {T {1}};
    return detail::broadcast_impl<T, extents<>, extents<Ns...>>::call(input);
}

template<typename V, typename T = tensor_value_type<V>, typename E = tensor_extents<V>>
KERNEL_FLOAT_INLINE tensor<T, E> zeros_like(const V&) {
    return zeros<T>(E {});
}

template<typename V, typename T = tensor_value_type<V>, typename E = tensor_extents<V>>
KERNEL_FLOAT_INLINE tensor<T, E> ones_like(const V&) {
    return ones<T>(E {});
}

namespace detail {
template<typename T, typename E, typename T2, typename E2, RoundingMode M = RoundingMode::ANY>
struct convert_helper {
    KERNEL_FLOAT_INLINE
    static tensor_storage<T2, E2::volume> call(tensor_storage<T, E::volume> input) {
        using F = ops::cast<T, T2, M>;
        tensor_storage<T2, E::volume> intermediate =
            detail::apply_impl<F, E::volume, T2, T>::call(F {}, input);
        return detail::broadcast_impl<T2, E, E2>::call(intermediate);
    }
};

template<typename T, typename E, RoundingMode M>
struct convert_helper<T, E, T, E, M> {
    KERNEL_FLOAT_INLINE
    static tensor_storage<T, E::volume> call(tensor_storage<T, E::volume> input) {
        return input;
    }
};

template<typename T, typename E, typename E2, RoundingMode M>
struct convert_helper<T, E, T, E2, M> {
    KERNEL_FLOAT_INLINE
    static tensor_storage<T, E2::volume> call(tensor_storage<T, E::volume> input) {
        return detail::broadcast_impl<T, E, E2>::call(input);
    }
};

template<typename T, typename E, typename T2, RoundingMode M>
struct convert_helper<T, E, T2, E, M> {
    KERNEL_FLOAT_INLINE
    static tensor_storage<T2, E::volume> call(tensor_storage<T, E::volume> input) {
        using F = ops::cast<T, T2, M>;
        return detail::apply_impl<F, E::volume, T2, T>::call(F {}, input);
    }
};
}  // namespace detail

/**
 * Cast the values of the given input tensor to type `R` and then broadcast the result to the given shape `(Ns...)`.
 */
template<typename R, size_t... Ns, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE tensor<R, extents<Ns...>>
convert(const V& input, extents<Ns...> new_shape = {}) {
    return detail::convert_helper<tensor_value_type<V>, tensor_extents<V>, R, extents<Ns...>, M>::
        call(into_tensor(input).storage());
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H




namespace kernel_float {

template<typename F, typename L, typename R>
using zip_type =
    tensor<result_t<F, tensor_value_type<L>, tensor_value_type<R>>, broadcast_tensor_extents<L, R>>;

template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_type<F, L, R> zip(F fun, const L& left, const R& right) {
    using A = tensor_value_type<L>;
    using B = tensor_value_type<R>;
    using O = result_t<F, A, B>;
    using E = broadcast_tensor_extents<L, R>;

    return detail::apply_impl<F, E::volume, O, A, B>::call(
        fun,
        broadcast<E>(left).storage(),
        broadcast<E>(right).storage());
}

template<typename F, typename L, typename R>
using zip_common_type = tensor<
    result_t<F, promoted_tensor_value_type<L, R>, promoted_tensor_value_type<L, R>>,
    broadcast_tensor_extents<L, R>>;

template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_common_type<F, L, R> zip_common(F fun, const L& left, const R& right) {
    using T = promoted_tensor_value_type<L, R>;
    using O = result_t<F, T, T>;
    using E = broadcast_tensor_extents<L, R>;

    return detail::apply_impl<F, E::volume, O, T, T>::call(
        fun,
        detail::convert_helper<tensor_value_type<L>, tensor_extents<L>, T, E>::call(
            into_tensor_storage(left)),
        detail::convert_helper<tensor_value_type<R>, tensor_extents<R>, T, E>::call(
            into_tensor_storage(right)));
}

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR)                                             \
    namespace ops {                                                                        \
    template<typename T>                                                                   \
    struct NAME {                                                                          \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                                \
            return T(EXPR);                                                                \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    template<typename L, typename R, typename C = promoted_tensor_value_type<L, R>>        \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {    \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                                               \
    template<typename L, typename R, typename C = promote_t<L, R>, typename E1, typename E2>      \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, tensor<L, E1>, tensor<R, E2>> operator OP(  \
        const tensor<L, E1>& left,                                                                \
        const tensor<R, E2>& right) {                                                             \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<L, tensor_value_type<R>>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, tensor<L, E>, R> operator OP(               \
        const tensor<L, E>& left,                                                                 \
        const R& right) {                                                                         \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<tensor_value_type<L>, R>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, tensor<R, E>> operator OP(               \
        const L& left,                                                                            \
        const tensor<R, E>& right) {                                                              \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }

KERNEL_FLOAT_DEFINE_BINARY_OP(add, +)
KERNEL_FLOAT_DEFINE_BINARY_OP(subtract, -)
KERNEL_FLOAT_DEFINE_BINARY_OP(divide, /)
KERNEL_FLOAT_DEFINE_BINARY_OP(multiply, *)
KERNEL_FLOAT_DEFINE_BINARY_OP(modulo, %)

KERNEL_FLOAT_DEFINE_BINARY_OP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINARY_OP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINARY_OP(less, <)
KERNEL_FLOAT_DEFINE_BINARY_OP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater, >)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater_equal, >=)

KERNEL_FLOAT_DEFINE_BINARY_OP(bit_and, &)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_or, |)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_xor, ^)

// clang-format off
template<template<typename> typename F, typename T, typename E, typename R>
static constexpr bool is_tensor_assign_allowed =
        is_tensor_broadcastable<R, E> &&
        is_implicit_convertible<
            result_t<
                F<promote_t<T, tensor_value_type<R>>>,
                    T,
                    tensor_value_type<R>
                >,
            T
        >;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                               \
    template<                                                                        \
        typename T,                                                                  \
        typename E,                                                                  \
        typename R,                                                                  \
        typename = enabled_t<is_tensor_assign_allowed<ops::NAME, T, E, R>>>          \
    KERNEL_FLOAT_INLINE tensor<T, E>& operator OP(tensor<T, E>& lhs, const R& rhs) { \
        using F = ops::NAME<T>;                                                      \
        lhs = zip_common(F {}, lhs, rhs);                                            \
        return lhs;                                                                  \
    }

KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(add, +=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(subtract, -=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(divide, /=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(multiply, *=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(modulo, %=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_and, &=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_or, |=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_xor, ^=)

#define KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME) KERNEL_FLOAT_DEFINE_BINARY(NAME, ::NAME(left, right))

KERNEL_FLOAT_DEFINE_BINARY_FUN(min)
KERNEL_FLOAT_DEFINE_BINARY_FUN(max)
KERNEL_FLOAT_DEFINE_BINARY_FUN(copysign)
KERNEL_FLOAT_DEFINE_BINARY_FUN(hypot)
KERNEL_FLOAT_DEFINE_BINARY_FUN(modf)
KERNEL_FLOAT_DEFINE_BINARY_FUN(nextafter)
KERNEL_FLOAT_DEFINE_BINARY_FUN(pow)
KERNEL_FLOAT_DEFINE_BINARY_FUN(remainder)

#if KERNEL_FLOAT_CUDA_DEVICE
KERNEL_FLOAT_DEFINE_BINARY_FUN(rhypot)
#endif

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_DEFINE_BINARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN)     \
    KERNEL_FLOAT_DEFINE_BINARY(FUN_NAME, ops::OP_NAME<T> {}(left, right)) \
    namespace ops {                                                       \
    template<>                                                            \
    struct OP_NAME<float> {                                               \
        KERNEL_FLOAT_INLINE float operator()(float left, float right) {   \
            return FLOAT_FUN(left, right);                                \
        }                                                                 \
    };                                                                    \
    }
#else
#define KERNEL_FLOAT_DEFINE_BINARY_FAST(FUN_NAME, OP_NAME, FLOAT_FUN) \
    KERNEL_FLOAT_DEFINE_BINARY(FUN_NAME, ops::OP_NAME<T> {}(left, right))
#endif

KERNEL_FLOAT_DEFINE_BINARY_FAST(fast_div, divide, __fdividef)
KERNEL_FLOAT_DEFINE_BINARY_FAST(fast_pow, pow, __powf)

namespace ops {
template<>
struct add<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left || right;
    }
};

template<>
struct multiply<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left && right;
    }
};

template<>
struct bit_and<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) && bool(right));
    }
};

template<>
struct bit_or<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) ^ bool(right));
    }
};

template<>
struct bit_and<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) && bool(right));
    }
};

template<>
struct bit_or<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) ^ bool(right));
    }
};
};  // namespace ops

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_CONSTANT
#define KERNEL_FLOAT_CONSTANT



namespace kernel_float {

template<typename T = double>
struct constant {
    KERNEL_FLOAT_INLINE
    constexpr constant(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    constexpr T get() const {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    constexpr operator T() const {
        return value_;
    }

  private:
    T value_;
};

template<typename T = double>
KERNEL_FLOAT_INLINE constexpr constant<T> make_constant(T value) {
    return value;
}

template<typename L, typename R>
struct promote_type<constant<L>, constant<R>> {
    using type = typename promote_type<L, R>::type;
};

template<typename L, typename R>
struct promote_type<constant<L>, R> {
    using type = typename promote_type<L, R>::type;
};

template<typename L, typename R>
struct promote_type<L, constant<R>> {
    using type = typename promote_type<L, R>::type;
};

namespace ops {
template<typename T, typename R, RoundingMode m>
struct cast<constant<T>, R, m> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R, m> {}(input);
    }
};
}  // namespace ops

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H



namespace kernel_float {
namespace detail {
template<typename F, size_t N, typename T, typename = void>
struct reduce_helper {
    KERNEL_FLOAT_INLINE static T call(F fun, const tensor_storage<T, N>& input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static T
    call(F fun, const tensor_storage<T, N>& input, index_sequence<0, Is...>) {
        T result = input[0];
#pragma unroll
        for (size_t i = 1; i < N; i++) {
            result = fun(result, input[i]);
        }
        return result;
    }
};
}  // namespace detail

/**
 * Reduce the elements of the given vector ``input`` into a single value using
 * the function ``fun``. This function should be a binary function that takes
 * two elements and returns one element. The order in which the elements
 * are reduced is not specified and depends on the reduction function and
 * the vector type.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 2, 1};
 * int y = reduce(x, [](int a, int b) { return a + b; }); // returns 5+2+1=8
 * ```
 */
template<typename F, typename V>
KERNEL_FLOAT_INLINE tensor_value_type<V> reduce(F fun, const V& input) {
    return detail::reduce_helper<F, tensor_volume<V>, tensor_value_type<V>>::call(
        fun,
        into_tensor_storage(input));
}

/**
 * Find the minimum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = min(x);  // Returns 0
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T min(const V& input) {
    return reduce(ops::min<T> {}, input);
}

/**
 * Find the maximum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = max(x);  // Returns 5
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T max(const V& input) {
    return reduce(ops::max<T> {}, input);
}

/**
 * Sum the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 8
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T sum(const V& input) {
    return reduce(ops::add<T> {}, input);
}

/**
 * Compute the dot product of the given vectors ``left`` and ``right``
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {1, 2, 3};
 * vec<int, 3> y = {4, 5, 6};
 * int y = dot(x, y);  // Returns 1*4+2*5+3*6 = 32
 * ```
 */
template<typename L, typename R, typename T = promoted_tensor_value_type<L, R>>
KERNEL_FLOAT_INLINE T dot(const L& left, const R& right) {
    return reduce(ops::add<T> {}, zip_common(ops::multiply<T> {}, left, right));
}

/**
 * Multiply the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 5*0*2*1*0 = 0
 * ```
 */
template<typename V, typename T = tensor_value_type<V>>
KERNEL_FLOAT_INLINE T product(const V& input) {
    return reduce(ops::multiply<T> {}, input);
}

/**
 * Check if all elements in the given vector ``input`` are non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template<typename V>
KERNEL_FLOAT_INLINE bool all(const V& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

/**
 * Check if any element in the given vector ``input`` is non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template<typename V>
KERNEL_FLOAT_INLINE bool any(const V& input) {
    return reduce(ops::bit_or<bool> {}, cast<bool>(input));
}

/**
 * Count the number of non-zero items in the given vector ``input``. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = count(x);  // Returns 3 (5, 2, 1 are non-zero)
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE int count(const V& input) {
    return sum(cast<int>(cast<bool>(input)));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
#ifndef KERNEL_FLOAT_BASE_H
#define KERNEL_FLOAT_BASE_H







namespace kernel_float {

template<typename Derived, typename T, size_t N>
struct tensor_extension {};

template<typename T, typename E, template<typename, size_t> class S>
struct tensor: tensor_extension<tensor<T, E, S>, T, E::volume> {
    static constexpr size_t rank = E::rank;
    static constexpr size_t volume = E::volume;

    using value_type = T;
    using extents_type = E;
    using ndindex_type = ndindex<rank>;
    using storage_type = S<T, volume>;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return E::volume;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return E::size(axis);
    }

    KERNEL_FLOAT_INLINE
    static constexpr extents_type shape() {
        return {};
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return E::stride(axis);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t linearize_index(ndindex_type index) {
        return E::ravel_index(index);
    }

    tensor(const tensor&) = default;

    KERNEL_FLOAT_INLINE
    tensor(storage_type storage) : storage_(storage) {}

    template<typename... Args, enabled_t<sizeof...(Args) == volume && volume >= 2, int> = 0>
    KERNEL_FLOAT_INLINE tensor(Args&&... args) : storage_ {std::forward<Args>(args)...} {}

    template<
        typename U,
        typename F,
        enabled_t<
            is_implicit_convertible<U, value_type> && is_tensor_broadcastable<F, extents_type>,
            int> = 0>
    KERNEL_FLOAT_INLINE tensor(const tensor<U, F>& input) :
        tensor(convert<T>(input, extents_type {})) {}

    template<
        typename U,
        typename F,
        enabled_t<
            !is_implicit_convertible<U, value_type> && is_tensor_broadcastable<F, extents_type>,
            int> = 0>
    explicit KERNEL_FLOAT_INLINE tensor(const tensor<U, F>& input) :
        tensor(convert<T>(input, extents_type {})) {}

    KERNEL_FLOAT_INLINE tensor(const value_type& input = {}) :
        tensor(convert<T>(input, extents_type {})) {}

    KERNEL_FLOAT_INLINE
    storage_type& storage() {
        return storage_;
    }

    KERNEL_FLOAT_INLINE
    const storage_type& storage() const {
        return storage_;
    }

    KERNEL_FLOAT_INLINE
    T* data() {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    const T* cdata() const {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    T* begin() {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    const T* begin() const {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    const T* cbegin() const {
        return storage_.data();
    }

    KERNEL_FLOAT_INLINE
    T* end() {
        return storage_.data() + E::volume;
    }

    KERNEL_FLOAT_INLINE
    const T* end() const {
        return storage_.data() + E::volume;
    }

    KERNEL_FLOAT_INLINE
    const T* cend() const {
        return storage_.data() + E::volume;
    }

    KERNEL_FLOAT_INLINE
    T& at(ndindex_type x) {
        return *(data() + linearize_index(x));
    }

    KERNEL_FLOAT_INLINE
    const T& at(ndindex_type x) const {
        return *(data() + linearize_index(x));
    }

    KERNEL_FLOAT_INLINE
    T get(ndindex_type x) const {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    void set(ndindex_type x, T value) {
        at(x) = std::move(value);
    }

    KERNEL_FLOAT_INLINE
    T& operator[](ndindex_type x) {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](ndindex_type x) const {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    T& operator()(ndindex_type x) {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    const T& operator()(ndindex_type x) const {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    tensor<T, extents<volume>> flatten() const {
        return storage_;
    }

    template<size_t... Ns>
    KERNEL_FLOAT_INLINE tensor<T, extents<Ns...>> reshape(extents<Ns...> = {}) const {
        static_assert(extents<Ns...>::volume == volume, "invalid reshape shape");
        return storage_;
    }

    template<size_t... Ns>
    KERNEL_FLOAT_INLINE tensor<T, extents<Ns...>> broadcast(extents<Ns...> new_shape = {}) const {
        return kernel_float::broadcast(*this, new_shape);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE tensor<result_t<F, T>, E> map(F fun = {}) const {
        return kernel_float::map(fun, *this);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun = {}) const {
        return kernel_float::reduce(fun, *this);
    }

  private:
    storage_type storage_;
};

template<typename Derived, typename T>
struct tensor_extension<Derived, T, 1> {
    KERNEL_FLOAT_INLINE
    T get() const {
        return static_cast<const Derived*>(this)->get({});
    }

    KERNEL_FLOAT_INLINE
    void set(T value) {
        static_cast<Derived*>(this)->set({}, value);
    }

    KERNEL_FLOAT_INLINE
    operator T() const {
        return get();
    }
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T1, T2, T3, T4)    \
    template<>                                                \
    struct into_tensor_traits<::T2> {                         \
        using type = tensor<T, extents<2>>;                   \
                                                              \
        KERNEL_FLOAT_INLINE                                   \
        static type call(::T2 v) {                            \
            return tensor_storage<T, 2> {v.x, v.y};           \
        }                                                     \
    };                                                        \
                                                              \
    template<>                                                \
    struct into_tensor_traits<::T3> {                         \
        using type = tensor<T, extents<3>>;                   \
                                                              \
        KERNEL_FLOAT_INLINE                                   \
        static type call(::T3 v) {                            \
            return tensor_storage<T, 3> {v.x, v.y, v.z};      \
        }                                                     \
    };                                                        \
                                                              \
    template<>                                                \
    struct into_tensor_traits<::T4> {                         \
        using type = tensor<T, extents<4>>;                   \
                                                              \
        KERNEL_FLOAT_INLINE                                   \
        static type call(::T4 v) {                            \
            return tensor_storage<T, 4> {v.x, v.y, v.z, v.w}; \
        }                                                     \
    };

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(char, char1, char2, char3, char4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(short, short1, short2, short3, short4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(int, int1, int2, int3, int4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long, long1, long2, long3, long4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long long, longlong1, longlong2, longlong3, longlong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned char, uchar1, uchar2, uchar3, uchar4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned short, ushort1, ushort2, ushort3, ushort4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned int, uint1, uint2, uint3, uint4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long, ulong1, ulong2, ulong3, ulong4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong1, ulonglong2, ulonglong3, ulonglong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(float, float1, float2, float3, float4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(double, double1, double2, double3, double4)

template<typename T>
using scalar = tensor<T, extents<>>;

template<typename T, size_t N>
using vec = tensor<T, extents<N>>;

template<typename T, size_t N, size_t M>
using mat = tensor<T, extents<N, M>>;

template<typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return tensor_storage<T, sizeof...(Args)> {T {args}...};
};

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H



#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>



namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__half)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __half)

template<>
struct into_tensor_traits<__half2> {
    using type = tensor<__half, extents<2>>;

    KERNEL_FLOAT_INLINE
    static type call(__half2 input) {
        return tensor_storage<__half, 2> {input.x, input.y};
    }
};

namespace detail {
template<typename F>
struct map_halfx2 {
    KERNEL_FLOAT_INLINE
    static __half2 call(F fun, __half2 input) {
        __half a = fun(input.x);
        __half b = fun(input.y);
        return {a, b};
    }
};

template<typename F>
struct zip_halfx2 {
    KERNEL_FLOAT_INLINE
    static __half2 call(F fun, __half2 left, __half2 right) {
        __half a = fun(left.x, left.y);
        __half b = fun(right.y, right.y);
        return {a, b};
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __half, __half> {
    KERNEL_FLOAT_INLINE static tensor_storage<__half, N>
    call(F fun, const tensor_storage<__half, N>& input) {
        tensor_storage<__half, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i += 2) {
            __half2 a = {input[i], input[i + 1]};
            __half2 b = map_halfx2<F>::call(fun, a);
            result[i + 0] = b.x;
            result[i + 1] = b.y;
        }

        if (N % 2 != 0) {
            result[N - 1] = fun(input[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __half, __half, __half> {
    KERNEL_FLOAT_INLINE static tensor_storage<__half, N>
    call(F fun, const tensor_storage<__half, N>& left, const tensor_storage<__half, N>& right) {
        tensor_storage<__half, N> result;
#pragma unroll
        for (size_t i = 0; i < N; i += 2) {
            __half2 a = {left[i], left[i + 1]};
            __half2 b = {right[i], right[i + 1]};
            __half2 c = zip_halfx2<F>::call(fun, a, b);
            result[i + 0] = c.x;
            result[i + 1] = c.y;
        }

        if (N % 2 != 0) {
            result[N - 1] = fun(left[N - 1], right[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct reduce_helper<F, N, __half, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __half call(F fun, const tensor_storage<__half, N>& input) {
        __half2 accum = {input[0], input[1]};

#pragma unroll
        for (size_t i = 2; i < N; i += 2) {
            __half2 a = {input[i], input[i + 1]};
            accum = zip_halfx2<F>::call(fun, accum, a);
        }

        __half result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input[N - 1]);
        }

        return result;
    }
};

};  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                               \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__half> {                                                           \
        KERNEL_FLOAT_INLINE __half operator()(__half input) {                       \
            return FUN1(input);                                                     \
        }                                                                           \
    };                                                                              \
    }                                                                               \
    namespace detail {                                                              \
    template<>                                                                      \
    struct map_halfx2<ops::NAME<__half>> {                                          \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 input) { \
            return FUN2(input);                                                     \
        }                                                                           \
    };                                                                              \
    }

KERNEL_FLOAT_FP16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_FP16_UNARY_FUN(negate, ::__hneg, ::__hneg2)
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_FP16_UNARY_FUN(cos, ::hcos, ::h2cos)
KERNEL_FLOAT_FP16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_FP16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_FP16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(log10, ::hlog10, ::h2log2)
KERNEL_FLOAT_FP16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)

KERNEL_FLOAT_FP16_UNARY_FUN(fast_exp, ::hexp, ::h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_log, ::hlog, ::h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_cos, ::hcos, ::h2cos)
KERNEL_FLOAT_FP16_UNARY_FUN(fast_sin, ::hsin, ::h2sin)

#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                               \
    template<>                                                                                    \
    struct NAME<__half> {                                                                         \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {                  \
            return FUN1(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    namespace detail {                                                                            \
    template<>                                                                                    \
    struct zip_halfx2<ops::NAME<__half>> {                                                        \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 left, __half2 right) { \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_FP16_BINARY_FUN(fast_div, __hdiv, __h2div)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_FP16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater_equal, __hge, __hgt2)

#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)    \
    namespace ops {                                      \
    template<>                                           \
    struct cast<T, __half> {                             \
        KERNEL_FLOAT_INLINE __half operator()(T input) { \
            return TO_HALF;                              \
        }                                                \
    };                                                   \
    template<>                                           \
    struct cast<__half, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__half input) { \
            return FROM_HALF;                            \
        }                                                \
    };                                                   \
    }

KERNEL_FLOAT_FP16_CAST(double, __double2half(input), double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float, __float2half(input), __half2float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_FP16_CAST(char, __int2half_rn(input), (char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed char, __int2half_rn(input), (signed char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned char, __int2half_rn(input), (unsigned char)__half2int_rz(input));

KERNEL_FLOAT_FP16_CAST(signed short, __half2short_rz(input), __short2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed int, __half2int_rz(input), __int2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned short, __half2ushort_rz(input), __ushort2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned int, __half2uint_rz(input), __uint2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

using half = __half;
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H



#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>




namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_bfloat16)

template<>
struct into_tensor_traits<__nv_bfloat162> {
    using type = tensor<__nv_bfloat16, extents<2>>;

    KERNEL_FLOAT_INLINE
    static type call(__nv_bfloat162 input) {
        return tensor_storage<__nv_bfloat16, 2> {input.x, input.y};
    }
};

namespace detail {
template<typename F>
struct map_bfloat16x2 {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 call(F fun, __nv_bfloat162 input) {
        __nv_bfloat16 a = fun(input.x);
        __nv_bfloat16 b = fun(input.y);
        return {a, b};
    }
};

template<typename F>
struct zip_bfloat16x2 {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 call(F fun, __nv_bfloat162 left, __nv_bfloat162 right) {
        __nv_bfloat16 a = fun(left.x, left.y);
        __nv_bfloat16 b = fun(right.y, right.y);
        return {a, b};
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __nv_bfloat16, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static tensor_storage<__nv_bfloat16, N>
    call(F fun, const tensor_storage<__nv_bfloat16, N>& input) {
        tensor_storage<__nv_bfloat16, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i += 2) {
            __nv_bfloat162 a = {input[i], input[i + 1]};
            __nv_bfloat162 b = map_bfloat16x2<F>::call(fun, a);
            result[i + 0] = b.x;
            result[i + 1] = b.y;
        }

        if (N % 2 != 0) {
            result[N - 1] = fun(input[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static tensor_storage<__nv_bfloat16, N> call(
        F fun,
        const tensor_storage<__nv_bfloat16, N>& left,
        const tensor_storage<__nv_bfloat16, N>& right) {
        tensor_storage<__nv_bfloat16, N> result;
#pragma unroll
        for (size_t i = 0; i < N; i += 2) {
            __nv_bfloat162 a = {left[i], left[i + 1]};
            __nv_bfloat162 b = {right[i], right[i + 1]};
            __nv_bfloat162 c = zip_bfloat16x2<F>::call(fun, a, b);
            result[i + 0] = c.x;
            result[i + 1] = c.y;
        }

        if (N % 2 != 0) {
            result[N - 1] = fun(left[N - 1], right[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct reduce_helper<F, N, __nv_bfloat16, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __nv_bfloat16
    call(F fun, const tensor_storage<__nv_bfloat16, N>& input) {
        __nv_bfloat162 accum = {input[0], input[1]};

#pragma unroll
        for (size_t i = 2; i < N; i += 2) {
            __nv_bfloat162 a = {input[i], input[i + 1]};
            accum = zip_bfloat16x2<F>::call(fun, accum, a);
        }

        __nv_bfloat16 result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input[N - 1]);
        }

        return result;
    }
};
}  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                       \
    namespace ops {                                                         \
    template<>                                                              \
    struct NAME<__nv_bfloat16> {                                            \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) { \
            return FUN1(input);                                             \
        }                                                                   \
    };                                                                      \
    }                                                                       \
    namespace detail {                                                      \
    template<>                                                              \
    struct map_bfloat16x2<ops::NAME<__nv_bfloat16>> {                       \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                           \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 input) {              \
            return FUN2(input);                                             \
        }                                                                   \
    };                                                                      \
    }

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2)
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos)
KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2)
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)

KERNEL_FLOAT_BF16_UNARY_FUN(fast_exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_cos, ::hcos, ::h2cos)
KERNEL_FLOAT_BF16_UNARY_FUN(fast_sin, ::hsin, ::h2sin)

#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                              \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__nv_bfloat16> {                                                    \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                           \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {                 \
            return FUN1(left, right);                                               \
        }                                                                           \
    };                                                                              \
    }                                                                               \
    namespace detail {                                                              \
    template<>                                                                      \
    struct zip_bfloat16x2<ops::NAME<__nv_bfloat16>> {                               \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                   \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 left, __nv_bfloat162 right) { \
            return FUN2(left, right);                                               \
        }                                                                           \
    };                                                                              \
    }

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(fast_div, __hdiv, __h2div)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)

#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)           \
    namespace ops {                                             \
    template<>                                                  \
    struct cast<T, __nv_bfloat16> {                             \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(T input) { \
            return TO_HALF;                                     \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<__nv_bfloat16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__nv_bfloat16 input) { \
            return FROM_HALF;                                   \
        }                                                       \
    };                                                          \
    }

KERNEL_FLOAT_BF16_CAST(double, __double2bfloat16(input), double(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float, __float2bfloat16(input), __bfloat162float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    signed char,
    __int2bfloat16_rn(input),
    (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned char,
    __int2bfloat16_rn(input),
    (unsigned char)__bfloat162int_rz(input));

KERNEL_FLOAT_BF16_CAST(signed short, __bfloat162short_rz(input), __short2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(signed int, __bfloat162int_rz(input), __int2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    signed long,
    __ll2bfloat16_rn(input),
    (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __bfloat162ushort_rz(input), __ushort2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __bfloat162uint_rz(input), __uint2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned long,
    __ull2bfloat16_rn(input),
    (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));

using bfloat16 = __nv_bfloat16;
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __nv_bfloat16)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __nv_bfloat16)

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE


namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  //KERNEL_FLOAT_BF16_H
#ifndef KERNEL_FLOAT_PRELUDE_H
#define KERNEL_FLOAT_PRELUDE_H




namespace kernel_float {
namespace prelude {
namespace kf = ::kernel_float;

template<typename T>
using kscalar = tensor<T, extents<>>;

template<typename T, size_t N>
using kvec = tensor<T, extents<N>>;

template<typename T, size_t N, size_t M>
using kmat = tensor<T, extents<N, M>>;

template<typename T, size_t... Ns>
using ktensor = tensor<T, extents<Ns...>>;

// clang-format off
template<typename T> using kvec1 = kvec<T, 1>;
template<typename T> using kvec2 = kvec<T, 2>;
template<typename T> using kvec3 = kvec<T, 3>;
template<typename T> using kvec4 = kvec<T, 4>;
template<typename T> using kvec5 = kvec<T, 5>;
template<typename T> using kvec6 = kvec<T, 6>;
template<typename T> using kvec7 = kvec<T, 7>;
template<typename T> using kvec8 = kvec<T, 8>;
// clang-format on

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)          \
    using k##NAME = tensor<T, extents<>>;         \
    template<size_t... Ns>                        \
    using k##NAME##N = tensor<T, extents<Ns...>>; \
    using k##NAME##1 = vec<T, 1>;                 \
    using k##NAME##2 = vec<T, 2>;                 \
    using k##NAME##3 = vec<T, 3>;                 \
    using k##NAME##4 = vec<T, 4>;                 \
    using k##NAME##5 = vec<T, 5>;                 \
    using k##NAME##6 = vec<T, 6>;                 \
    using k##NAME##7 = vec<T, 7>;                 \
    using k##NAME##8 = vec<T, 8>;

KERNEL_FLOAT_TYPE_ALIAS(char, char)
KERNEL_FLOAT_TYPE_ALIAS(short, short)
KERNEL_FLOAT_TYPE_ALIAS(int, int)
KERNEL_FLOAT_TYPE_ALIAS(long, long)
KERNEL_FLOAT_TYPE_ALIAS(longlong, long long)

KERNEL_FLOAT_TYPE_ALIAS(uchar, unsigned char)
KERNEL_FLOAT_TYPE_ALIAS(ushort, unsigned short)
KERNEL_FLOAT_TYPE_ALIAS(uint, unsigned int)
KERNEL_FLOAT_TYPE_ALIAS(ulong, unsigned long)
KERNEL_FLOAT_TYPE_ALIAS(ulonglong, unsigned long long)

KERNEL_FLOAT_TYPE_ALIAS(float, float)
KERNEL_FLOAT_TYPE_ALIAS(f32x, float)
KERNEL_FLOAT_TYPE_ALIAS(float32x, float)

KERNEL_FLOAT_TYPE_ALIAS(double, double)
KERNEL_FLOAT_TYPE_ALIAS(f64x, double)
KERNEL_FLOAT_TYPE_ALIAS(float64x, double)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(bfloat16, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bf16, __nv_bfloat16)
#endif

template<size_t... Ns>
static constexpr extents<Ns...> kshape = {};

template<typename... Args>
KERNEL_FLOAT_INLINE kvec<promote_t<Args...>, sizeof...(Args)> make_kvec(Args&&... args) {
    return make_vec(std::forward<Args>(args)...);
};

template<typename T = double>
using kconstant = constant<T>;

template<typename T = double>
KERNEL_FLOAT_INLINE constexpr kconstant<T> kconst(T value) {
    return value;
}

}  // namespace prelude
}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_TRIOPS_H
#define KERNEL_FLOAT_TRIOPS_H




namespace kernel_float {

namespace ops {
template<typename T>
struct conditional {
    KERNEL_FLOAT_INLINE T operator()(bool cond, T true_value, T false_value) {
        if (cond) {
            return true_value;
        } else {
            return false_value;
        }
    }
};
}  // namespace ops

/**
 * Return elements chosen from `true_values` and `false_values` depending on `cond`.
 *
 * This function broadcasts all arguments to the same shape and it promotes the values of `true_values` and
 * `false_values` into the same type. Next, it casts the values of `cond` to booleans and returns a tensor where
 * the values are taken from `true_values` if the condition is true and `false_values` otherwise.
 *
 * @param cond The condition used for selection.
 * @param true_values The tensor of values to choose from when the condition is true.
 * @param false_values The tensor of values to choose from when the condition is false.
 * @return A tensor containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename R,
    typename T = promoted_tensor_value_type<L, R>,
    typename E = broadcast_extents<tensor_extents<C>, broadcast_tensor_extents<L, R>>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond, const L& true_values, const R& false_values) {
    using F = ops::conditional<T>;

    return detail::apply_impl<F, E::volume, T, bool, T, T>::call(
        F {},
        detail::convert_helper<tensor_value_type<C>, tensor_extents<C>, bool, E>::call(
            into_tensor_storage(cond)),
        detail::convert_helper<tensor_value_type<L>, tensor_extents<L>, T, E>::call(
            into_tensor_storage(true_values)),
        detail::convert_helper<tensor_value_type<R>, tensor_extents<R>, T, E>::call(
            into_tensor_storage(false_values)));
}

/**
 * Selects elements from `true_values` depending on `cond`.
 *
 * This function returns a tensor where the values are taken from `true_values` where `cond` is `true` and `0` where
 * `cond is `false`.
 *
 * @param cond The condition used for selection.
 * @param true_values The tensor of values to choose from when the condition is true.
 * @return A tensor containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename T = tensor_value_type<L>,
    typename E = broadcast_extents<tensor_extents<C>, tensor_extents<L>>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond, const L& true_values) {
    tensor<T, extents<>> false_values = T {};
    return where(cond, true_values, false_values);
}

/**
 * Returns a tensor where the values are `T(1)` where `cond` is `true` and `T(0)` where `cond` is `false`.
 *
 * @param cond The condition used for selection.
 * @return A tensor containing elements as per the condition.
 */
template<typename T = bool, typename C, typename E = tensor_extents<C>>
KERNEL_FLOAT_INLINE tensor<T, E> where(const C& cond) {
    tensor<T, extents<>> true_values = T {true};
    tensor<T, extents<>> false_values = T {false};
    return where(cond, true_values, false_values);
}

namespace ops {
template<typename T>
struct fma {
    KERNEL_FLOAT_INLINE T operator()(T a, T b, T c) {
        return a + b * c;
    }
};

#if KERNEL_FLOAT_IS_DEVICE
template<>
struct fma<float> {
    KERNEL_FLOAT_INLINE float operator()(float a, float b, float c) {
        return __fmaf_rn(a, b, c);
    }
};

template<>
struct fma<double> {
    KERNEL_FLOAT_INLINE double operator()(double a, double b, double c) {
        return __fma_rn(a, b, c);
    }
};
#endif
}  // namespace ops

/**
 * Computes the result of `a * b + c`. This is done in a single operation if possible.
 */
template<
    typename A,
    typename B,
    typename C,
    typename T = promoted_tensor_value_type<A, B, C>,
    typename E = broadcast_extents<tensor_extents<A>, broadcast_tensor_extents<B, C>>>
KERNEL_FLOAT_INLINE tensor<T, E> fma(const A& a, const B& b, const C& c) {
    using F = ops::fma<T>;

    return detail::apply_impl<F, E::volume, T, T, T, T>::call(
        F {},
        detail::convert_helper<tensor_value_type<A>, tensor_extents<A>, T, E>::call(
            into_tensor_storage(a)),
        detail::convert_helper<tensor_value_type<B>, tensor_extents<B>, T, E>::call(
            into_tensor_storage(b)),
        detail::convert_helper<tensor_value_type<C>, tensor_extents<C>, T, E>::call(
            into_tensor_storage(c)));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TRIOPS_H
