//================================================================================
// this file has been auto-generated, do not modify its contents!
// date: 2023-08-16 12:43:52.493856
// git hash: b236a521d5decdd59b17361febbb7ee39803b715
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
#ifndef KERNEL_FLOAT_BASE_H
#define KERNEL_FLOAT_BASE_H




namespace kernel_float {

template<typename T, size_t N, size_t Alignment = alignof(T)>
struct alignas(Alignment) aligned_array {
    KERNEL_FLOAT_INLINE
    T* data() {
        return items_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items_;
    }

    T items_[N] = {};
};

template<typename T, size_t Alignment>
struct aligned_array<T, 0, Alignment> {
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
};

template<typename T, size_t Alignment>
struct alignas(Alignment) aligned_array<T, 1, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T value = {}) : x(value) {}

    KERNEL_FLOAT_INLINE
    operator T() const {
        return x;
    }

    KERNEL_FLOAT_INLINE
    T* data() {
        return &x;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return &x;
    }

    T x;
};

template<typename T, size_t Alignment>
struct alignas(Alignment) aligned_array<T, 2, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T x, T y) : x(x), y(y) {}

    KERNEL_FLOAT_INLINE
    aligned_array() : aligned_array(T {}, T {}) {}

    KERNEL_FLOAT_INLINE
    T* data() {
        return items;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items;
    }

    union {
        T items[2];
        struct {
            T x;
            T y;
        };
    };
};

template<typename T, size_t Alignment>
struct alignas(Alignment) aligned_array<T, 3, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T x, T y, T z) : x(x), y(y), z(z) {}

    KERNEL_FLOAT_INLINE
    aligned_array() : aligned_array(T {}, T {}, T {}) {}

    KERNEL_FLOAT_INLINE
    T* data() {
        return items;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items;
    }

    union {
        T items[3];
        struct {
            T x;
            T y;
            T z;
        };
    };
};

template<typename T, size_t Alignment>
struct alignas(Alignment) aligned_array<T, 4, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

    KERNEL_FLOAT_INLINE
    aligned_array() : aligned_array(T {}, T {}, T {}, T {}) {}

    KERNEL_FLOAT_INLINE
    T* data() {
        return items;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items;
    }

    union {
        T items[4];
        struct {
            T x;
            T y;
            T z;
            T w;
        };
    };
};

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
using vector_storage = aligned_array<T, N, compute_max_alignment(sizeof(T) * N, alignof(T))>;

template<size_t... Ns>
struct extent;

template<size_t N>
struct extent<N> {
    static constexpr size_t value = N;
    static constexpr size_t size = N;
};

template<typename T>
struct into_vector_traits {
    using value_type = T;
    using extent_type = extent<1>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, 1> call(const T& input) {
        return vector_storage<T, 1> {input};
    }
};

template<typename T, size_t N>
struct into_vector_traits<T[N]> {
    using value_type = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const T (&input)[N]) {
        return call(input, make_index_sequence<N>());
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static vector_storage<T, N>
    call(const T (&input)[N], index_sequence<Is...>) {
        return {input[Is]...};
    }
};

template<typename V>
struct into_vector_traits<const V>: into_vector_traits<V> {};

template<typename V>
struct into_vector_traits<V&>: into_vector_traits<V> {};

template<typename V>
struct into_vector_traits<const V&>: into_vector_traits<V> {};

template<typename V>
struct into_vector_traits<V&&>: into_vector_traits<V> {};

template<typename T, size_t N, size_t A>
struct into_vector_traits<aligned_array<T, N, A>> {
    using value_type = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const aligned_array<T, N, A>& input) {
        return input;
    }
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T1, T2, T3, T4) \
    template<>                                             \
    struct into_vector_traits<::T2> {                      \
        using value_type = T;                              \
        using extent_type = extent<2>;                     \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static vector_storage<T, 2> call(::T2 v) {         \
            return {v.x, v.y};                             \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct into_vector_traits<::T3> {                      \
        using value_type = T;                              \
        using extent_type = extent<3>;                     \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static vector_storage<T, 3> call(::T3 v) {         \
            return {v.x, v.y, v.z};                        \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct into_vector_traits<::T4> {                      \
        using value_type = T;                              \
        using extent_type = extent<4>;                     \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static vector_storage<T, 4> call(::T4 v) {         \
            return {v.x, v.y, v.z, v.w};                   \
        }                                                  \
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

template<typename T, typename E, typename S = vector_storage<T, E::size>>
struct vector;

template<typename T, typename E, typename S>
struct into_vector_traits<vector<T, E, S>> {
    using value_type = T;
    using extent_type = E;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, E::value> call(const vector<T, E, S>& input) {
        return input.storage();
    }
};

template<typename V>
struct vector_traits;

template<typename T, typename E, typename S>
struct vector_traits<vector<T, E, S>> {
    using value_type = T;
    using extent_type = E;
    using storage_type = S;
    using vector_type = vector<T, E, S>;
};

template<typename V>
using vector_value_type = typename into_vector_traits<V>::value_type;

template<typename V>
using vector_extent_type = typename into_vector_traits<V>::extent_type;

template<typename V>
static constexpr size_t vector_extent = vector_extent_type<V>::value;

template<typename V>
using into_vector_type = vector<vector_value_type<V>, vector_extent_type<V>>;

template<typename V>
using vector_storage_type = vector_storage<vector_value_type<V>, vector_extent<V>>;

template<typename... Vs>
using promoted_vector_value_type = promote_t<vector_value_type<Vs>...>;

template<typename V>
KERNEL_FLOAT_INLINE vector_storage_type<V> into_vector_storage(V&& input) {
    return into_vector_traits<V>::call(std::forward<V>(input));
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
    T normi = T(1) / b.norm();

    return {
        (a.real() * b.real() + a.imag() * b.imag()) * normi,
        (a.imag() * b.real() - a.real() * b.imag()) * normi};
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, T b) {
    return a * (T(1) / b);
}

template<typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(T a, complex_type<T> b) {
    T normi = T(1) / b.norm();

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
    return v.imag();
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
    KERNEL_FLOAT_INLINE static vector_storage<Output, N>
    call(F fun, const vector_storage<Args, N>&... inputs) {
        vector_storage<Output, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = fun(inputs.data()[i]...);
        }

        return result;
    }
};
}  // namespace detail

template<typename F, typename V>
using map_type = vector<result_t<F, vector_value_type<V>>, vector_extent_type<V>>;

/**
 * Apply the function `F` to each element from the vector `input` and return the results as a new vector.
 *
 * Examples
 * ========
 * ```
 * vec<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * vec<float, 4> squared = map([](auto x) { return x * x; }, input); // [1.0f, 4.0f, 9.0f, 16.0f]
 * ```
 */
template<typename F, typename V>
KERNEL_FLOAT_INLINE map_type<F, V> map(F fun, const V& input) {
    using Input = vector_value_type<V>;
    using Output = result_t<F, Input>;
    return detail::apply_impl<F, vector_extent<V>, Output, Input>::call(
        fun,
        into_vector_storage(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                                      \
    namespace ops {                                                                                \
    template<typename T>                                                                           \
    struct NAME {                                                                                  \
        KERNEL_FLOAT_INLINE T operator()(T input) {                                                \
            return T(EXPR);                                                                        \
        }                                                                                          \
    };                                                                                             \
    }                                                                                              \
    template<typename V>                                                                           \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> NAME(const V& input) { \
        using F = ops::NAME<vector_value_type<V>>;                                                 \
        return map(F {}, input);                                                                   \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                           \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                      \
    template<typename T, typename E, typename S>                               \
    KERNEL_FLOAT_INLINE vector<T, E> operator OP(const vector<T, E, S>& vec) { \
        return NAME(vec);                                                      \
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

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H




namespace kernel_float {

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

/**
 * Cast the elements of the given vector `input` to a different type `R`.
 *
 * This function casts each element of the input vector to a different data type specified by
 * template parameter `R`.
 *
 * Optionally, the rounding mode can be set using the `Mode` template parameter. The default mode is `ANY`, which
 * uses the fastest rounding mode available.
 *
 * Example
 * =======
 * ```
 * vec<float, 4> input {1.2f, 2.7f, 3.5f, 4.9f};
 * auto casted = cast<int>(input); // [1, 2, 3, 4]
 * ```
 */
template<typename R, RoundingMode Mode = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, vector_extent_type<V>> cast(const V& input) {
    using F = ops::cast<vector_value_type<V>, R, Mode>;
    return map(F {}, input);
}

namespace detail {

template<typename... Es>
struct broadcast_extent_helper;

template<typename E>
struct broadcast_extent_helper<E> {
    using type = E;
};

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

template<typename A, typename B, typename C, typename... Rest>
struct broadcast_extent_helper<A, B, C, Rest...>:
    broadcast_extent_helper<typename broadcast_extent_helper<A, B>::type, C, Rest...> {};

}  // namespace detail

template<typename... Es>
using broadcast_extent = typename detail::broadcast_extent_helper<Es...>::type;

template<typename... Vs>
using broadcast_vector_extent_type = broadcast_extent<vector_extent_type<Vs>...>;

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

template<typename T>
struct broadcast_impl<T, extent<1>, extent<1>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, 1> call(vector_storage<T, 1> input) {
        return input;
    }
};

}  // namespace detail

/**
 * Takes the given vector `input` and extends its size to a length of `N`. This is only valid if the size of `input`
 * is 1 or `N`.
 *
 * Example
 * =======
 * ```
 * vec<float, 1> a = {1.0f};
 * vec<float, 5> x = broadcast<5>(a);  // Returns [1.0f, 1.0f, 1.0f, 1.0f, 1.0f]
 *
 * vec<float, 5> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
 * vec<float, 5> y = broadcast<5>(b);  // Returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f]
 * ```
 */
template<size_t N, typename V>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, extent<N>>
broadcast(const V& input, extent<N> new_size = {}) {
    using T = vector_value_type<V>;
    return detail::broadcast_impl<T, vector_extent_type<V>, extent<N>>::call(
        into_vector_storage(input));
}

/**
 * Takes the given vector `input` and extends its size to the same length as vector `other`. This is only valid if the
 * size of `input` is 1 or the same as `other`.
 */
template<typename V, typename R>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<R>>
broadcast_like(const V& input, const R& other) {
    return broadcast(input, vector_extent_type<R> {});
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

/**
 * Returns a vector containing `N` copies of `value`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = fill<3>(42); // return [42, 42, 42]
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
 * vec<int, 3> a = zeros<int, 3>(); // return [0, 0, 0]
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
 * vec<int, 3> a = ones<int, 3>(); // return [1, 1, 1]
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
 * vec<int, 3> b = fill_like(a, 42); // return [42, 42, 42]
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
 * vec<int, 3> b = zeros_like(a); // return [0, 0, 0]
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
 * vec<int, 3> b = ones_like(a); // return [1, 1, 1]
 * ```
 */
template<typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> ones_like(const V& = {}) {
    return ones<T>(E {});
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H




namespace kernel_float {

template<typename F, typename L, typename R>
using zip_type = vector<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    broadcast_vector_extent_type<L, R>>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements.
 *
 * Example
 * =======
 * ```
 * vec<bool, 3> make_negative = {true, false, true};
 * vec<int, 3> input = {1, 2, 3};
 * vec<int, 3> output = zip([](bool b, int n){ return b ? -n : +n; }, make_negative, input); // returns [-1, 2, -3]
 * ```
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_type<F, L, R> zip(F fun, const L& left, const R& right) {
    using A = vector_value_type<L>;
    using B = vector_value_type<R>;
    using O = result_t<F, A, B>;
    using E = broadcast_vector_extent_type<L, R>;

    return detail::apply_impl<F, E::value, O, A, B>::call(
        fun,
        detail::broadcast_impl<A, vector_extent_type<L>, E>::call(into_vector_storage(left)),
        detail::broadcast_impl<B, vector_extent_type<R>, E>::call(into_vector_storage(right)));
}

template<typename F, typename L, typename R>
using zip_common_type = vector<
    result_t<F, promoted_vector_value_type<L, R>, promoted_vector_value_type<L, R>>,
    broadcast_vector_extent_type<L, R>>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements. The elements are promoted to a common type before applying
 * the binary function.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1.0f, 2.0f, 3.0f};
 * vec<int, 3> b = {4, 5, 6};
 * vec<int, 3> c = zip_common([](float x, float y){ return x + y; }, a, b); // returns [5.0f, 7.0f, 9.0f]
 * ```
 */
template<typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_common_type<F, L, R> zip_common(F fun, const L& left, const R& right) {
    using T = promoted_vector_value_type<L, R>;
    using O = result_t<F, T, T>;
    using E = broadcast_vector_extent_type<L, R>;

    return detail::apply_impl<F, E::value, O, T, T>::call(
        fun,
        detail::convert_helper<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(left)),
        detail::convert_helper<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(right)));
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
    template<typename L, typename R, typename C = promoted_vector_value_type<L, R>>        \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {    \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                                               \
    template<typename L, typename R, typename C = promote_t<L, R>, typename E1, typename E2>      \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E1>, vector<R, E2>> operator OP(  \
        const vector<L, E1>& left,                                                                \
        const vector<R, E2>& right) {                                                             \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<L, vector_value_type<R>>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E>, R> operator OP(               \
        const vector<L, E>& left,                                                                 \
        const R& right) {                                                                         \
        return zip_common(ops::NAME<C> {}, left, right);                                          \
    }                                                                                             \
    template<typename L, typename R, typename C = promote_t<vector_value_type<L>, R>, typename E> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, vector<R, E>> operator OP(               \
        const L& left,                                                                            \
        const vector<R, E>& right) {                                                              \
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
static constexpr bool is_vector_assign_allowed =
        is_vector_broadcastable<R, E> &&
        is_implicit_convertible<
            result_t<
                F<promote_t<T, vector_value_type<R>>>,
                    T,
                    vector_value_type<R>
                >,
            T
        >;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                               \
    template<                                                                        \
        typename T,                                                                  \
        typename E,                                                                  \
        typename R,                                                                  \
        typename = enabled_t<is_vector_assign_allowed<ops::NAME, T, E, R>>>          \
    KERNEL_FLOAT_INLINE vector<T, E>& operator OP(vector<T, E>& lhs, const R& rhs) { \
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

namespace detail {
template<typename T>
struct cross_helper {
    KERNEL_FLOAT_INLINE
    static vector<T, extent<3>>
    call(const vector_storage<T, 3>& av, const vector_storage<T, 3>& bv) {
        auto a = av.data();
        auto b = bv.data();
        vector<T, extent<6>> v0 = {a[1], a[2], a[0], a[2], a[0], a[1]};
        vector<T, extent<6>> v1 = {b[2], b[0], b[1], b[1], b[2], b[0]};
        vector<T, extent<6>> rv = v0 * v1;

        auto r = rv.data();
        vector<T, extent<3>> r0 = {r[0], r[1], r[2]};
        vector<T, extent<3>> r1 = {r[3], r[4], r[5]};
        return r0 - r1;
    }
};
};  // namespace detail

/**
 * Calculates the cross-product between two vectors of length 3.
 */
template<
    typename L,
    typename R,
    typename T = promoted_vector_value_type<L, R>,
    typename =
        enabled_t<is_vector_broadcastable<L, extent<3>> && is_vector_broadcastable<R, extent<3>>>>
KERNEL_FLOAT_INLINE vector<T, extent<3>> cross(const L& left, const R& right) {
    return detail::cross_helper<T>::call(convert_storage<T, 3>(left), convert_storage<T, 3>(right));
}

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
template<typename T, typename R>
struct cast<constant<T>, R> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R> {}(input);
    }
};

template<typename T, typename R, RoundingMode m>
struct cast<constant<T>, R, m> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R, m> {}(input);
    }
};
}  // namespace ops

#define KERNEL_FLOAT_CONSTANT_DEFINE_OP(OP)                                      \
    template<typename L, typename R>                                             \
    R operator OP(const constant<L>& left, const R& right) {                     \
        using T = vector_value_type<R>;                                          \
        return operator OP(T(left.get()), right);                                \
    }                                                                            \
                                                                                 \
    template<typename L, typename R>                                             \
    L operator OP(const L& left, const constant<R>& right) {                     \
        using T = vector_value_type<L>;                                          \
        return operator OP(left, T(right.get()));                                \
    }                                                                            \
                                                                                 \
    template<typename L, typename R, typename T = promote_t<L, R>>               \
    constant<T> operator OP(const constant<L>& left, const constant<R>& right) { \
        return constant<T>(operator OP(T(left.get()), T(right.get())));          \
    }

//KERNEL_FLOAT_CONSTANT_DEFINE_OP(+)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(-)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(*)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(/)

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H




namespace kernel_float {

/**
 * Apply the function fun for each element from input.
 *
 * Example
 * =======
 * ```
 * for_each(range<int, 3>(), [&](auto i) {
 *    printf("element: %d\n", i);
 * });
 * ```
 */
template<typename V, typename F>
void for_each(V&& input, F fun) {
    auto storage = into_vector_storage(input);

#pragma unroll
    for (size_t i = 0; i < vector_extent<V>; i++) {
        fun(storage.data()[i]);
    }
}

namespace detail {
template<typename T, size_t N>
struct range_helper {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call() {
        vector_storage<T, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = T(i);
        }

        return result;
    }
};
}  // namespace detail

/**
 * Generate vector consisting of the numbers `0...N-1` of type `T`
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2]
 * vec<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range() {
    return detail::range_helper<T, N>::call();
}

/**
 * Takes a vector `vec<T, N>` and returns a new vector consisting of the numbers ``0...N-1`` of type ``T``
 *
 * Example
 * =======
 * ```
 * auto input = vec<float, 3>(5.0f, 10.0f, -1.0f);
 * auto indices = range_like(input);  // returns [0.0f, 1.0f, 2.0f]
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> range_like(const V& = {}) {
    return detail::range_helper<vector_value_type<V>, vector_extent<V>>::call();
}

/**
 * Takes a vector of size ``N`` and returns a new vector consisting of the numbers ``0...N-1``. The data type used
 * for the indices is given by the first template argument, which is `size_t` by default. This function is useful when
 * needing to iterate over the indices of a vector.
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2] of type size_t
 * vec<size_t, 3> a = each_index(float3(6, 4, 2));
 *
 * // Returns [0, 1, 2] of type int.
 * vec<int, 3> b = each_index<int>(float3(6, 4, 2));
 *
 * vec<float, 3> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * for (auto index: each_index<int>(input)) {
 *   printf("%d] %f\n", index, input[index]);
 * }
 * ```
 */
template<typename T = size_t, typename V>
KERNEL_FLOAT_INLINE vector<T, vector_extent_type<V>> each_index(const V& = {}) {
    return detail::range_helper<T, vector_extent<V>>::call();
}

namespace detail {
template<typename V, typename T = vector_value_type<V>, size_t N = vector_extent<V>>
struct flatten_helper {
    using value_type = typename flatten_helper<T>::value_type;
    static constexpr size_t size = N * flatten_helper<T>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, N> storage = into_vector_storage(input);

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            flatten_helper<T>::call(output + flatten_helper<T>::size * i, storage.data()[i]);
        }
    }
};

template<typename T>
struct flatten_helper<T, T, 1> {
    using value_type = T;
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE
    static void call(T* output, const T& input) {
        *output = input;
    }

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) {
        *output = ops::cast<T, U> {}(input);
    }
};
}  // namespace detail

template<typename V>
using flatten_value_type = typename detail::flatten_helper<V>::value_type;

template<typename V>
static constexpr size_t flatten_size = detail::flatten_helper<V>::size;

template<typename V>
using flatten_type = vector<flatten_value_type<V>, extent<flatten_size<V>>>;

/**
 * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
 *
 * Example
 * =======
 * ```
 * vec<float2, 3> input = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
 * vec<float, 6> result = flatten(input); // returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]
 * ```
 */
template<typename V>
KERNEL_FLOAT_INLINE flatten_type<V> flatten(const V& input) {
    vector_storage<flatten_value_type<V>, flatten_size<V>> output;
    detail::flatten_helper<V>::call(output.data(), input);
    return output;
}

namespace detail {
template<typename U, typename V = U, typename T = vector_value_type<V>>
struct concat_base_helper {
    static constexpr size_t size = vector_extent<V>;

    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, size> storage = into_vector_storage(input);

        for (size_t i = 0; i < size; i++) {
            output[i] = ops::cast<T, U> {}(storage.data()[i]);
        }
    }
};

template<typename U, typename T>
struct concat_base_helper<U, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) {
        *output = ops::cast<T, U> {}(input);
    }
};

template<typename T>
struct concat_base_helper<T, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(T* output, const T& input) {
        *output = input;
    }
};

template<typename... Vs>
struct concat_helper {};

template<typename V, typename... Vs>
struct concat_helper<V, Vs...> {
    using value_type =
        typename promote_type<vector_value_type<V>, typename concat_helper<Vs...>::value_type>::
            type;
    static constexpr size_t size = concat_base_helper<V>::size + concat_helper<Vs...>::size;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input, const Vs&... rest) {
        concat_base_helper<U, V>::call(output, input);
        concat_helper<Vs...>::call(output + concat_base_helper<U, V>::size, rest...);
    }
};

template<>
struct concat_helper<> {
    using value_type = void;
    static constexpr size_t size = 1;

    template<typename U>
    KERNEL_FLOAT_INLINE static void call(U* output) {}
};
}  // namespace detail

template<typename... Vs>
using concat_value_type = promote_t<typename detail::concat_helper<Vs...>::value_type>;

template<typename... Vs>
static constexpr size_t concat_size = detail::concat_helper<Vs...>::size;

template<typename... Vs>
using concat_type = vector<concat_value_type<Vs...>, extent<concat_size<Vs...>>>;

/**
 * Concatenates the provided input values into a single one-dimensional vector.
 *
 * This function works in three steps:
 * - All input values are converted into vectors using the `into_vector` operation.
 * - The resulting vectors' elements are then promoted into a shared value type.
 * - The resultant vectors are finally concatenated together.
 *
 * For instance, when invoking this function with arguments of types `float, double2, double`:
 * - After the first step: `vec<float, 1>, vec<double, 2>, vec<double, 1>`
 * - After the second step: `vec<double, 1>, vec<double, 2>, vec<double, 1>`
 * - After the third step: `vec<double, 4>`
 *
 * Example
 * =======
 * ```
 * double vec1 = 1.0;
 * double3 vec2 = {3.0, 4.0, 5.0);
 * double4 vec3 = {6.0, 7.0, 8.0, 9.0};
 * vec<double, 9> concatenated = concat(vec1, vec2, vec3); // contains [1, 2, 3, 4, 5, 6, 7, 8, 9]
 *
 * int num1 = 42;
 * float num2 = 3.14159;
 * int2 num3 = {-10, 10};
 * vec<float, 3> concatenated = concat(num1, num2, num3); // contains [42, 3.14159, -10, 10]
 * ```
 */
template<typename... Vs>
KERNEL_FLOAT_INLINE concat_type<Vs...> concat(const Vs&... inputs) {
    vector_storage<concat_value_type<Vs...>, concat_size<Vs...>> output;
    detail::concat_helper<Vs...>::call(output.data(), inputs...);
    return output;
}

template<typename V, typename... Is>
using select_type = vector<vector_value_type<V>, extent<concat_size<Is...>>>;

/**
 * Selects elements from the this vector based on the specified indices.
 *
 * Example
 * =======
 * ```
 * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
 * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
 *
 * vec<int, 4> indices = {0, 4, 4, 2};
 * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
 * ```
 */
template<typename V, typename... Is>
KERNEL_FLOAT_INLINE select_type<V, Is...> select(const V& input, const Is&... indices) {
    using T = vector_value_type<V>;
    static constexpr size_t N = vector_extent<V>;
    static constexpr size_t M = concat_size<Is...>;

    vector_storage<size_t, M> index_set;
    detail::concat_helper<Is...>::call(index_set.data(), indices...);

    vector_storage<T, N> inputs = into_vector_storage(input);
    vector_storage<T, M> outputs;
    for (size_t i = 0; i < M; i++) {
        size_t j = index_set.data()[i];

        if (j < N) {
            outputs.data()[i] = inputs.data()[j];
        }
    }

    return outputs;
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_MEMORY_H
#define KERNEL_FLOAT_MEMORY_H

/*




namespace kernel_float {

    namespace detail {
        template <typename T, size_t N, typename Is = make_index_sequence_helper<N>>
        struct load_helper;

        template <typename T, size_t N, size_t... Is>
        struct load_helper<T, N, index_sequence<Is...>> {
            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets
            ) {
                return {base[offsets.data()[Is]]...};
            }

            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<bool, N> mask
            ) {
                if (all(mask)) {
                    return call(base, offsets);
                } else {
                    return {
                            (mask.data()[Is] ? base[offsets.data()[Is]] : T())...
                    };
                }
            }
        };
    }

    template <
            typename T,
            typename I,
            typename M,
            typename E = broadcast_vector_extent_type<I, M>
    >
    KERNEL_FLOAT_INLINE
    vector<T, E> load(const T* ptr, const I& indices, const M& mask) {
        static constexpr E new_size = {};

        return detail::load_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<bool>(mask, new_size)
        );
    }

    template <typename T, typename I>
    KERNEL_FLOAT_INLINE
    vector<T, vector_extent<I>> load(const T* ptr, const I& indices) {
        return detail::load_helper<T, vector_extent<I>::value>::call(
                ptr,
                cast<ptrdiff_t>(indices)
        );
    }

    template <size_t N, typename T>
    KERNEL_FLOAT_INLINE
    vector<T, extent<N>> load(const T* ptr, ptrdiff_t length) {
        using index_type = vector_value_type<I>;
        return load_masked(ptr, range<ptrdiff_t, N>(), range<ptrdiff_t, N>() < length);
    }

    template <size_t N, typename T>
    KERNEL_FLOAT_INLINE
    vector<T, extent<N>> load(const T* ptr) {
        return load(ptr, range<ptrdiff_t, N>());
    }

    namespace detail {
        template <typename T, size_t N>
        struct store_helper {
            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<bool, N> mask,
                    vector_storage<T, N> values
            ) {
                for (size_t i = 0; i < N; i++) {
                    if (mask.data()[i]) {
                        base[offset.data()[i]] = values.data()[i];
                    }
                }
            }

            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<T, N> values
            ) {
                for (size_t i = 0; i < N; i++) {
                    base[offset.data()[i]] = values.data()[i];
                }
            }
        };
    }

    template <
            typename T,
            typename I,
            typename M,
            typename V,
            typename E = broadcast_extent<vector_extent_type<V>, broadcast_vector_extent_type<M, I>>>
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const M& mask, const V& values) {
        static constexpr E new_size = {};

        return detail::store_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<bool>(mask, new_size),
                convert_storage<T>(values, new_size)
        );
    }

    template <
            typename T,
            typename I,
            typename V,
            typename E = broadcast_vector_extent_type<V, I>
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const V& values) {
        static constexpr E new_size = {};

        return detail::store_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<T>(values, new_size)
        );
    }


    template <
            typename T,
            typename V
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const V& values) {
        using E = vector_extent<V>;
        return store(ptr, range<ptrdiff_t, E::value>(), values);
    }

    template <typename T, typename I, typename S, typename V>
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const S& length, const V& values) {
        using index_type = vector_value_type<I>;
        return store(ptr, indices, (indices >= I(0)) & (indices < length), values);
    }


    template <typename T, size_t alignment>
    struct aligned_ptr_base {
        static_assert(alignof(T) % alignment == 0, "invalid alignment, must be multiple of alignment of `T`");

        KERNEL_FLOAT_INLINE
        aligned_ptr_base(): ptr_(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr_base(T* ptr): ptr_(ptr) {}

        KERNEL_FLOAT_INLINE
        T* get() const {
            // TOOD: check if this way is support across all compilers
#if defined(__has_builtin) && __has_builtin(__builtin_assume_aligned)
            return __builtin_assume_aligned(ptr_, alignment);
#else
            return ptr_;
#endif
        }

        KERNEL_FLOAT_INLINE
        operator T*() const {
            return get();
        }

        KERNEL_FLOAT_INLINE
        T& operator*() const {
            return *get();
        }

        template <typename I>
        KERNEL_FLOAT_INLINE
        T& operator[](I index) const {
            return get()[index);
        }

    private:
        T* ptr_ = nullptr;
    };

    template <typename T, size_t alignment = 256>
    struct aligned_ptr;

    template <typename T, size_t alignment>
    struct aligned_ptr: aligned_ptr_base<T, alignment> {
        using base_type = aligned_ptr_base<T, alignment>;

        KERNEL_FLOAT_INLINE
        aligned_ptr(): base_type(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<T, alignment> ptr): base_type(ptr.get()) {}
    };

    template <typename T, size_t alignment>
    struct aligned_ptr<const T, alignment>: aligned_ptr_base<const T, alignment> {
        using base_type = aligned_ptr_base<const T, alignment>;

        KERNEL_FLOAT_INLINE
        aligned_ptr(): base_type(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(const T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<T, alignment> ptr): base_type(ptr.get()) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<const T, alignment> ptr): base_type(ptr.get()) {}
    };


    template <typename T, size_t alignment>
    KERNEL_FLOAT_INLINE
    T* operator+(aligned_ptr<T, alignment> ptr, ptrdiff_t index) {
        return ptr.get() + index;
    }

    template <typename T, size_t alignment>
    KERNEL_FLOAT_INLINE
    T* operator+(ptrdiff_t index, aligned_ptr<T, alignment> ptr) {
        return ptr.get() + index;
    }

    template <typename T, size_t alignment, size_t alignment2>
    KERNEL_FLOAT_INLINE
    ptrdiff_t operator-(aligned_ptr<T, alignment> left, aligned_ptr<T, alignment2> right) {
        return left.get() - right.get();
    }

    template <typename T>
    using unaligned_ptr = aligned_ptr<T, alignof(T)>;

}
*/

#endif  //KERNEL_FLOAT_MEMORY_H
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H



namespace kernel_float {
namespace detail {
template<typename F, size_t N, typename T, typename = void>
struct reduce_helper {
    KERNEL_FLOAT_INLINE static T call(F fun, const vector_storage<T, N>& input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static T
    call(F fun, const vector_storage<T, N>& input, index_sequence<0, Is...>) {
        T result = input.data()[0];
#pragma unroll
        for (size_t i = 1; i < N; i++) {
            result = fun(result, input.data()[i]);
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
KERNEL_FLOAT_INLINE vector_value_type<V> reduce(F fun, const V& input) {
    return detail::reduce_helper<F, vector_extent<V>, vector_value_type<V>>::call(
        fun,
        into_vector_storage(input));
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
template<typename V, typename T = vector_value_type<V>>
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
template<typename V, typename T = vector_value_type<V>>
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
template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T sum(const V& input) {
    return reduce(ops::add<T> {}, input);
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
template<typename V, typename T = vector_value_type<V>>
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
template<typename T = int, typename V>
KERNEL_FLOAT_INLINE T count(const V& input) {
    return sum(cast<T>(cast<bool>(input)));
}

namespace detail {
template<typename T, size_t N>
struct dot_helper {
    KERNEL_FLOAT_INLINE
    static T call(const vector_storage<T, N>& left, const vector_storage<T, N>& right) {
        return sum(zip(ops::multiply<T> {}, left, right));
    }
};
}  // namespace detail

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
template<typename L, typename R, typename T = promoted_vector_value_type<L, R>>
KERNEL_FLOAT_INLINE T dot(const L& left, const R& right) {
    using E = broadcast_vector_extent_type<L, R>;
    return detail::dot_helper<T, E::value>::call(
        convert_storage<T>(left, E {}),
        convert_storage<T>(right, E {}));
}

namespace detail {
template<typename T, size_t N>
struct magnitude_helper {
    KERNEL_FLOAT_INLINE
    static T call(const vector_storage<T, N>& input) {
        return ops::sqrt<T> {}(detail::dot_helper<T, N>::call(input, input));
    }
};

template<typename T>
struct magnitude_helper<T, 0> {
    KERNEL_FLOAT_INLINE
    static T call(const vector_storage<T, 0>& input) {
        return T {};
    }
};

template<typename T>
struct magnitude_helper<T, 1> {
    KERNEL_FLOAT_INLINE
    static T call(const vector_storage<T, 1>& input) {
        return ops::abs<T> {}(input);
    }
};

template<typename T>
struct magnitude_helper<T, 2> {
    KERNEL_FLOAT_INLINE
    static T call(const vector_storage<T, 2>& input) {
        return ops::hypot<T> {}(input.data()[0], input.data()[1]);
    }
};

// The 3-argument overload of hypot is only available from C++17
#ifdef __cpp_lib_hypot
template<>
struct magnitude_helper<float, 3> {
    KERNEL_FLOAT_INLINE
    static float call(const vector_storage<float, 3>& input) {
        return std::hypot(input.data()[0], input.data()[1], input.data()[2]);
    }
};

template<>
struct magnitude_helper<double, 3> {
    KERNEL_FLOAT_INLINE
    static float call(const vector_storage<double, 3>& input) {
        return std::hypot(input.data()[0], input.data()[1], input.data()[2]);
    }
};
#endif

}  // namespace detail

/**
 * Compute the magnitude of the given input vector. This calculates the square root of the sum of squares, also
 * known as the Euclidian norm of the vector.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> x = {2, 3, 6};
 * float y = mag(x);  // Returns sqrt(2*2 + 3*3 + 6*6) = 7
 * ```
 */
template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T mag(const V& input) {
    return detail::magnitude_helper<T, vector_extent<V>>::call(into_vector_storage(input));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
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
 * This function broadcasts all arguments to the same size and then promotes the values of `true_values` and
 * `false_values` into the same type. Next, it casts the values of `cond` to booleans and returns a vector where
 * the values are taken from `true_values` where the condition is true and `false_values` otherwise.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @param false_values The vector of values to choose from when the condition is false.
 * @return A vector containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename R,
    typename T = promoted_vector_value_type<L, R>,
    typename E = broadcast_vector_extent_type<C, L, R>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values, const R& false_values) {
    using F = ops::conditional<T>;

    return detail::apply_impl<F, E::value, T, bool, T, T>::call(
        F {},
        detail::convert_helper<vector_value_type<C>, vector_extent_type<C>, bool, E>::call(
            into_vector_storage(cond)),
        detail::convert_helper<vector_value_type<L>, vector_extent_type<L>, T, E>::call(
            into_vector_storage(true_values)),
        detail::convert_helper<vector_value_type<R>, vector_extent_type<R>, T, E>::call(
            into_vector_storage(false_values)));
}

/**
 * Selects elements from `true_values` depending on `cond`.
 *
 * This function returns a vector where the values are taken from `true_values` where `cond` is `true` and `0` where
 * `cond is `false`.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @return A vector containing selected elements as per the condition.
 */
template<
    typename C,
    typename L,
    typename T = vector_value_type<L>,
    typename E = broadcast_vector_extent_type<C, L>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values) {
    vector<T, extent<1>> false_values = T {};
    return where(cond, true_values, false_values);
}

/**
 * Returns a vector having the value `T(1)` where `cond` is `true` and `T(0)` where `cond` is `false`.
 *
 * @param cond The condition used for selection.
 * @return A vector containing elements as per the condition.
 */
template<typename T = bool, typename C, typename E = vector_extent_type<C>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond) {
    return cast<T>(cast<bool>(cond));
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
    typename T = promoted_vector_value_type<A, B, C>,
    typename E = broadcast_vector_extent_type<A, B, C>>
KERNEL_FLOAT_INLINE vector<T, E> fma(const A& a, const B& b, const C& c) {
    using F = ops::fma<T>;

    return detail::apply_impl<F, E::value, T, T, T, T>::call(
        F {},
        detail::convert_helper<vector_value_type<A>, vector_extent_type<A>, T, E>::call(
            into_vector_storage(a)),
        detail::convert_helper<vector_value_type<B>, vector_extent_type<B>, T, E>::call(
            into_vector_storage(b)),
        detail::convert_helper<vector_value_type<C>, vector_extent_type<C>, T, E>::call(
            into_vector_storage(c)));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TRIOPS_H
#ifndef KERNEL_FLOAT_VECTOR_H
#define KERNEL_FLOAT_VECTOR_H








namespace kernel_float {

/**
 * Container that stores ``N`` elements of type ``T``.
 *
 * It is not recommended to use this class directly, but instead, use the type `vec<T, N>` which is an alias for
 * `vector<T, extent<N>, vector_storage<T, E>>`.
 *
 * @tparam T The type of the values stored within the vector.
 * @tparam E The size of this vector. Should be of type `extent<N>`.
 * @tparam S The object's storage class. Should be the type `vector_storage<T, E>`
 */
template<typename T, typename E, class S>
struct vector: public S {
    using value_type = T;
    using extent_type = E;
    using storage_type = S;

    // Copy another `vector<T, E>`
    vector(const vector&) = default;

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const storage_type& storage) : storage_type(storage) {}

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const value_type& input = {}) :
        storage_type(detail::broadcast_impl<T, extent<1>, E>::call(input)) {}

    // For all other arguments, we convert it using `convert_storage` according to broadcast rules
    template<typename U, enabled_t<is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE vector(U&& input) :
        storage_type(convert_storage<T>(input, extent_type {})) {}

    template<typename U, enabled_t<!is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE explicit vector(U&& input) :
        storage_type(convert_storage<T>(input, extent_type {})) {}

    // List of `N` (where N >= 2), simply pass forward to the storage
    template<
        typename A,
        typename B,
        typename... Rest,
        typename = enabled_t<sizeof...(Rest) + 2 == E::size>>
    KERNEL_FLOAT_INLINE vector(const A& a, const B& b, const Rest&... rest) :
        storage_type {a, b, rest...} {}

    /**
     * Returns the number of elements in this vector.
     */
    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return E::size;
    }

    KERNEL_FLOAT_INLINE
    storage_type& storage() {
        return *this;
    }

    KERNEL_FLOAT_INLINE
    const storage_type& storage() const {
        return *this;
    }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    T* data() {
        return storage().data();
    }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    const T* data() const {
        return storage().data();
    }

    KERNEL_FLOAT_INLINE
    const T* cdata() const {
        return this->data();
    }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& at(size_t i) {
        return *(this->data() + i);
    }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& at(size_t i) const {
        return *(this->data() + i);
    }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        return at(i);
    }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        return at(i);
    }

    KERNEL_FLOAT_INLINE
    T& operator()(size_t i) {
        return at(i);
    }

    KERNEL_FLOAT_INLINE
    const T& operator()(size_t i) const {
        return at(i);
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    T* begin() {
        return this->data();
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* begin() const {
        return this->data();
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* cbegin() const {
        return this->data();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    T* end() {
        return this->data() + size();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* end() const {
        return this->data() + size();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* cend() const {
        return this->data() + size();
    }

    /**
     * Copy the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T get(size_t x) const {
        return at(x);
    }

    /**
     * Set the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    void set(size_t x, T value) {
        at(x) = std::move(value);
    }

    /**
     * Selects elements from the this vector based on the specified indices.
     *
     * Example
     * =======
     * ```
     * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
     * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
     *
     * vec<int, 4> indices = {0, 4, 4, 2};
     * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
     * ```
     */
    template<typename V, typename... Is>
    KERNEL_FLOAT_INLINE select_type<V, Is...> select(const Is&... indices) {
        return kernel_float::select(*this, indices...);
    }

    /**
     * Cast the elements of this vector to type `R` and returns a new vector.
     */
    template<typename R, RoundingMode Mode = RoundingMode::ANY>
    KERNEL_FLOAT_INLINE vector<R, extent_type> cast() const {
        return kernel_float::cast<R, Mode>(*this);
    }

    /**
     * Broadcast this vector into a new size `(Ns...)`.
     */
    template<size_t... Ns>
    KERNEL_FLOAT_INLINE vector<T, extent<Ns...>> broadcast(extent<Ns...> new_size = {}) const {
        return kernel_float::broadcast(*this, new_size);
    }

    /**
     * Apply the given function `F` to each element of this vector and returns a new vector with the results.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE vector<result_t<F, T>, E> map(F fun) const {
        return kernel_float::map(fun, *this);
    }

    /**
     * Reduce the elements of the given vector input into a single value using the function `F`.
     *
     * This function should be a binary function that takes two elements and returns one element. The order in which
     * the elements are reduced is not specified and depends on the reduction function and the vector type.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun) const {
        return kernel_float::reduce(fun, *this);
    }

    /**
     * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
     */
    KERNEL_FLOAT_INLINE flatten_type<vector> flatten() const {
        return kernel_float::flatten(*this);
    }

    /**
     * Apply the given function `F` to each element of this vector.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const {
        return kernel_float::for_each(*this, std::move(fun));
    }
};

/**
 * Convert the given `input` into a vector. This function can perform one of the following actions:
 *
 * - For vectors `vec<T, N>`, it simply returns the original vector.
 * - For primitive types `T` (e.g., `int`, `float`, `double`), it returns a `vec<T, 1>`.
 * - For array-like types (e.g., `std::array<T, N>`, `T[N]`), it returns `vec<T, N>`.
 * - For vector-like types (e.g., `int2`, `dim3`), it returns `vec<T, N>`.
 */
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_vector(V&& input) {
    return into_vector_traits<V>::call(std::forward<V>(input));
}

template<typename T>
using scalar = vector<T, extent<1>>;

template<typename T, size_t N>
using vec = vector<T, extent<N>>;

// clang-format off
template<typename T> using vec1 = vec<T, 1>;
template<typename T> using vec2 = vec<T, 2>;
template<typename T> using vec3 = vec<T, 3>;
template<typename T> using vec4 = vec<T, 4>;
template<typename T> using vec5 = vec<T, 5>;
template<typename T> using vec6 = vec<T, 6>;
template<typename T> using vec7 = vec<T, 7>;
template<typename T> using vec8 = vec<T, 8>;
// clang-format on

/**
 * Create a vector from a variable number of input values.
 *
 * The resulting vector type is determined by promoting the types of the input values into a common type.
 * The number of input values determines the dimension of the resulting vector.
 *
 * Example
 * =======
 * ```
 * auto v1 = make_vec(1.0f, 2.0f, 3.0f); // Creates a vec<float, 3> [1.0f, 2.0f, 3.0f]
 * auto v2 = make_vec(1, 2, 3, 4);       // Creates a vec<int, 4> [1, 2, 3, 4]
 * ```
 */
template<typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return vector_storage<T, sizeof...(Args)> {T {args}...};
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
struct into_vector_traits<__half2> {
    using value_type = __half;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__half, 2> call(__half2 input) {
        return {input.x, input.y};
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
    KERNEL_FLOAT_INLINE static vector_storage<__half, N>
    call(F fun, const vector_storage<__half, N>& input) {
        vector_storage<__half, N> result;

#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __half2 a = {input.data()[i], input.data()[i + 1]};
            __half2 b = map_halfx2<F>::call(fun, a);
            result.data()[i + 0] = b.x;
            result.data()[i + 1] = b.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(input.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __half, __half, __half> {
    KERNEL_FLOAT_INLINE static vector_storage<__half, N>
    call(F fun, const vector_storage<__half, N>& left, const vector_storage<__half, N>& right) {
        vector_storage<__half, N> result;
#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __half2 a = {left.data()[i], left.data()[i + 1]};
            __half2 b = {right.data()[i], right.data()[i + 1]};
            __half2 c = zip_halfx2<F>::call(fun, a, b);
            result.data()[i + 0] = c.x;
            result.data()[i + 1] = c.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(left.data()[N - 1], right.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct reduce_helper<F, N, __half, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __half call(F fun, const vector_storage<__half, N>& input) {
        __half2 accum = {input.data()[0], input.data()[1]};

#pragma unroll
        for (size_t i = 2; i + 2 <= N; i += 2) {
            __half2 a = {input.data()[i], input.data()[i + 1]};
            accum = zip_halfx2<F>::call(fun, accum, a);
        }

        __half result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input.data()[N - 1]);
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

#if KERNEL_FLOAT_IS_DEVICE
namespace detail {
template<size_t N>
struct dot_helper<__half, N> {
    KERNEL_FLOAT_INLINE
    static __half
    call(const vector_storage<__half, N>& left, const vector_storage<__half, N>& right) {
        if (N == 0) {
            return __half(0);
        } else if (N == 1) {
            return __hmul(left.data()[0], right.data()[0]);
        } else {
            __half2 first_a = {left.data()[0], left.data()[1]};
            __half2 first_b = {right.data()[0], right.data()[1]};
            __half2 accum = __hmul2(first_a, first_b);

#pragma unroll
            for (size_t i = 2; i + 2 <= N; i += 2) {
                __half2 a = {left.data()[i], left.data()[i + 1]};
                __half2 b = {right.data()[i], right.data()[i + 1]};
                accum = __hfma2(a, b, accum);
            }

            __half result = __hadd(accum.x, accum.y);

            if (N % 2 != 0) {
                __half a = left.data()[N - 1];
                __half b = right.data()[N - 1];
                result = __hfma(a, b, result);
            }

            return result;
        }
    }
};
}  // namespace detail
#endif

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
struct into_vector_traits<__nv_bfloat162> {
    using value_type = __nv_bfloat16;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<__nv_bfloat16, 2> call(__nv_bfloat162 input) {
        return {input.x, input.y};
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
    KERNEL_FLOAT_INLINE static vector_storage<__nv_bfloat16, N>
    call(F fun, const vector_storage<__nv_bfloat16, N>& input) {
        vector_storage<__nv_bfloat16, N> result;

#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {input.data()[i], input.data()[i + 1]};
            __nv_bfloat162 b = map_bfloat16x2<F>::call(fun, a);
            result.data()[i + 0] = b.x;
            result.data()[i + 1] = b.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(input.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct apply_impl<F, N, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> {
    KERNEL_FLOAT_INLINE static vector_storage<__nv_bfloat16, N> call(
        F fun,
        const vector_storage<__nv_bfloat16, N>& left,
        const vector_storage<__nv_bfloat16, N>& right) {
        vector_storage<__nv_bfloat16, N> result;
#pragma unroll
        for (size_t i = 0; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {left.data()[i], left.data()[i + 1]};
            __nv_bfloat162 b = {right.data()[i], right.data()[i + 1]};
            __nv_bfloat162 c = zip_bfloat16x2<F>::call(fun, a, b);
            result.data()[i + 0] = c.x;
            result.data()[i + 1] = c.y;
        }

        if (N % 2 != 0) {
            result.data()[N - 1] = fun(left.data()[N - 1], right.data()[N - 1]);
        }

        return result;
    }
};

template<typename F, size_t N>
struct reduce_helper<F, N, __nv_bfloat16, enabled_t<(N >= 2)>> {
    KERNEL_FLOAT_INLINE static __nv_bfloat16
    call(F fun, const vector_storage<__nv_bfloat16, N>& input) {
        __nv_bfloat162 accum = {input.data()[0], input.data()[1]};

#pragma unroll
        for (size_t i = 2; i + 2 <= N; i += 2) {
            __nv_bfloat162 a = {input.data()[i], input.data()[i + 1]};
            accum = zip_bfloat16x2<F>::call(fun, accum, a);
        }

        __nv_bfloat16 result = fun(accum.x, accum.y);

        if (N % 2 != 0) {
            result = fun(result, input.data()[N - 1]);
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

#if KERNEL_FLOAT_IS_DEVICE
namespace detail {
template<size_t N>
struct dot_helper<__nv_bfloat16, N> {
    KERNEL_FLOAT_INLINE
    static __nv_bfloat16 call(
        const vector_storage<__nv_bfloat16, N>& left,
        const vector_storage<__nv_bfloat16, N>& right) {
        if (N == 0) {
            return __nv_bfloat16(0);
        } else if (N == 1) {
            return __hmul(left.data()[0], right.data()[0]);
        } else {
            __nv_bfloat162 first_a = {left.data()[0], left.data()[1]};
            __nv_bfloat162 first_b = {right.data()[0], right.data()[1]};
            __nv_bfloat162 accum = __hmul2(first_a, first_b);

#pragma unroll
            for (size_t i = 2; i + 2 <= N; i += 2) {
                __nv_bfloat162 a = {left.data()[i], left.data()[i + 1]};
                __nv_bfloat162 b = {right.data()[i], right.data()[i + 1]};
                accum = __hfma2(a, b, accum);
            }

            __nv_bfloat16 result = __hadd(accum.x, accum.y);

            if (N % 2 != 0) {
                __nv_bfloat16 a = left.data()[N - 1];
                __nv_bfloat16 b = right.data()[N - 1];
                result = __hfma(a, b, result);
            }

            return result;
        }
    }
};
}  // namespace detail
#endif

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
using kscalar = vector<T, extent<1>>;

template<typename T, size_t N>
using kvec = vector<T, extent<N>>;

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

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)  \
    template<size_t N>                    \
    using k##NAME = vector<T, extent<N>>; \
    using k##NAME##1 = vec<T, 1>;         \
    using k##NAME##2 = vec<T, 2>;         \
    using k##NAME##3 = vec<T, 3>;         \
    using k##NAME##4 = vec<T, 4>;         \
    using k##NAME##5 = vec<T, 5>;         \
    using k##NAME##6 = vec<T, 6>;         \
    using k##NAME##7 = vec<T, 7>;         \
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

template<size_t N>
static constexpr extent<N> kextent = {};

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

KERNEL_FLOAT_INLINE
static constexpr kconstant<double> operator""_c(long double v) {
    return static_cast<double>(v);
}

KERNEL_FLOAT_INLINE
static constexpr kconstant<long long int> operator""_c(unsigned long long int v) {
    return static_cast<long long int>(v);
}

}  // namespace prelude
}  // namespace kernel_float

#endif
