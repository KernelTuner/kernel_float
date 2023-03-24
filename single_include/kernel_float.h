//================================================================================
// this file has been auto-generated, do not modify its contents!
// date: 2023-03-24 15:02:06.799324
// git hash: de4a7013b7b2d6ab8c473998306f51790422c3a0
//================================================================================


#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __CUDACC__
#define KERNEL_FLOAT_CUDA (1)

#ifdef __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE    __forceinline__ __device__
#define KERNEL_FLOAT_ON_DEVICE (1)
#define KERNEL_FLOAT_ON_HOST   (0)
#define KERNEL_FLOAT_CUDA_ARCH (__CUDA_ARCH__)
#else
#define KERNEL_FLOAT_INLINE    __forceinline__ __host__
#define KERNEL_FLOAT_ON_DEVICE (0)
#define KERNEL_FLOAT_ON_HOST   (1)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif
#else
#define KERNEL_FLOAT_INLINE    inline
#define KERNEL_FLOAT_CUDA      (0)
#define KERNEL_FLOAT_ON_HOST   (1)
#define KERNEL_FLOAT_ON_DEVICE (0)
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

template<size_t I>
struct const_index {
    static constexpr size_t value = I;

    KERNEL_FLOAT_INLINE constexpr operator size_t() const noexcept {
        return I;
    }
};

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

template<typename T, typename U>
struct common_type_helper<T, U> {
    using type = typename common_type<T, U>::type;
};

template<typename T, typename U, typename R, typename... Rest>
struct common_type_helper<T, U, R, Rest...>:
    common_type_helper<typename common_type<T, U>::type, R, Rest...> {};
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
#ifndef KERNEL_FLOAT_STORAGE
#define KERNEL_FLOAT_STORAGE



namespace kernel_float {

template<typename V>
struct vector_traits {
    using value_type = V;
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE
    static V fill(value_type value) {
        return value;
    }

    KERNEL_FLOAT_INLINE
    static V create(value_type value) {
        return value;
    }

    KERNEL_FLOAT_INLINE
    static value_type get(const V& self, size_t index) {
        KERNEL_FLOAT_ASSERT(index == 0);
        return self;
    }

    KERNEL_FLOAT_INLINE
    static void set(V& self, size_t index, value_type value) {
        KERNEL_FLOAT_ASSERT(index == 0);
        self = value;
    }
};

template<typename V>
struct into_storage_traits {
    using type = V;

    KERNEL_FLOAT_INLINE
    static constexpr type call(V self) {
        return self;
    }
};

template<typename V>
struct into_storage_traits<V&>: into_storage_traits<V> {};

template<typename V>
struct into_storage_traits<const V&>: into_storage_traits<V> {};

template<typename V>
struct into_storage_traits<V&&>: into_storage_traits<V> {};

template<typename V>
using into_storage_type = typename into_storage_traits<V>::type;

template<typename V>
KERNEL_FLOAT_INLINE into_storage_type<V> into_storage(V&& input) {
    return into_storage_traits<V>::call(input);
}

template<typename V>
static constexpr size_t vector_size = vector_traits<into_storage_type<V>>::size;

template<typename V>
using vector_value_type = typename vector_traits<into_storage_type<V>>::value_type;

template<typename V, size_t I>
struct vector_index {
    using value_type = vector_value_type<V>;

    KERNEL_FLOAT_INLINE
    static value_type get(const V& self) {
        return vector_traits<V>::get(self, I);
    }

    KERNEL_FLOAT_INLINE
    static void set(V& self, value_type value) {
        return vector_traits<V>::set(self, I, value);
    }
};

template<typename V>
KERNEL_FLOAT_INLINE vector_value_type<V> vector_get(const V& self, size_t index) {
    return vector_traits<V>::get(self, index);
}

template<size_t I, typename V>
KERNEL_FLOAT_INLINE vector_value_type<V> vector_get(const V& self, const_index<I> = {}) {
    return vector_index<V, I>::get(self);
}

template<typename Output, typename Input, typename Indices, typename = void>
struct vector_swizzle;

template<typename Output, typename Input, size_t... Is>
struct vector_swizzle<Output, Input, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static Output call(const Input& storage) {
        return vector_traits<Output>::create(vector_get<Is>(storage)...);
    }
};

template<typename V>
struct vector;

template<typename T, size_t N, size_t alignment = alignof(T)>
struct alignas(alignment) array {
    T items_[N];

    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        KERNEL_FLOAT_ASSERT(i < N);
        return items_[i];
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        KERNEL_FLOAT_ASSERT(i < N);
        return items_[i];
    }
};

template<typename T, size_t N, size_t A>
struct vector_traits<array<T, N, A>> {
    using self_type = array<T, N, A>;
    using value_type = T;
    static constexpr size_t size = N;

    template<typename... Args>
    KERNEL_FLOAT_INLINE static self_type create(Args&&... args) {
        return {args...};
    }

    KERNEL_FLOAT_INLINE
    static self_type fill(value_type value) {
        self_type result;
        for (size_t i = 0; i < N; i++) {
            result[i] = value;
        }
        return result;
    }

    KERNEL_FLOAT_INLINE
    static value_type get(const self_type& self, size_t index) {
        KERNEL_FLOAT_ASSERT(index < N);
        return self[index];
    }

    KERNEL_FLOAT_INLINE
    static void set(self_type& self, size_t index, value_type value) {
        KERNEL_FLOAT_ASSERT(index < N);
        self[index] = value;
    }
};

template<typename T, size_t A>
struct array<T, 0, A> {};

template<typename T, size_t A>
struct vector_traits<array<T, 0, A>> {
    using self_type = array<T, 0, A>;
    using value_type = T;
    static constexpr size_t size = 0;

    KERNEL_FLOAT_INLINE
    static self_type create() {
        return {};
    }

    KERNEL_FLOAT_INLINE
    static self_type fill(value_type value) {
        return {};
    }

    KERNEL_FLOAT_INLINE
    static value_type get(const self_type& self, size_t index) {
        KERNEL_FLOAT_UNREACHABLE;
    }

    KERNEL_FLOAT_INLINE
    static void set(self_type& self, size_t index, value_type value) {
        KERNEL_FLOAT_UNREACHABLE;
    }
};

enum struct Alignment {
    Minimum,
    Packed,
    Maximum,
};

constexpr size_t calculate_alignment(Alignment required, size_t min_alignment, size_t total_size) {
    if (required == Alignment::Packed) {
        if (total_size <= 1) {
            return 1;
        } else if (total_size <= 2) {
            return 2;
        } else if (total_size <= 4) {
            return 4;
        } else if (total_size <= 8) {
            return 8;
        } else {
            return 16;
        }
    } else if (required == Alignment::Maximum) {
        if (total_size % 16 == 0) {
            return 16;
        } else if (total_size % 8 == 0) {
            return 8;
        } else if (total_size % 4 == 0) {
            return 4;
        } else if (total_size % 2 == 0) {
            return 2;
        } else {
            return 1;
        }
    }

    else {
        return min_alignment;
    }
}

template<typename T, size_t N, Alignment A, typename = void>
struct default_storage {
    using type = array<T, N, calculate_alignment(A, alignof(T), sizeof(T) * N)>;
};

template<typename T, Alignment A>
struct default_storage<T, 1, A> {
    using type = T;
};

template<typename T, size_t N, Alignment A = Alignment::Maximum>
using default_storage_type = typename default_storage<T, N, A>::type;

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T1, T2, T3, T4) \
    template<>                                             \
    struct vector_traits<T1> {                             \
        using value_type = T;                              \
        static constexpr size_t size = 1;                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T1 create(T x) {                            \
            return {x};                                    \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T1 fill(T v) {                              \
            return {v};                                    \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T get(const T1& self, size_t index) {       \
            switch (index) {                               \
                case 0:                                    \
                    return self.x;                         \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static void set(T1& self, size_t index, T value) { \
            switch (index) {                               \
                case 0:                                    \
                    self.x = value;                        \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct vector_traits<T2> {                             \
        using value_type = T;                              \
        static constexpr size_t size = 2;                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T2 create(T x, T y) {                       \
            return {x, y};                                 \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T2 fill(T v) {                              \
            return {v, v};                                 \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T get(const T2& self, size_t index) {       \
            switch (index) {                               \
                case 0:                                    \
                    return self.x;                         \
                case 1:                                    \
                    return self.y;                         \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static void set(T2& self, size_t index, T value) { \
            switch (index) {                               \
                case 0:                                    \
                    self.x = value;                        \
                case 1:                                    \
                    self.y = value;                        \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct vector_traits<T3> {                             \
        using value_type = T;                              \
        static constexpr size_t size = 3;                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T3 create(T x, T y, T z) {                  \
            return {x, y, z};                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T3 fill(T v) {                              \
            return {v, v, v};                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T get(const T3& self, size_t index) {       \
            switch (index) {                               \
                case 0:                                    \
                    return self.x;                         \
                case 1:                                    \
                    return self.y;                         \
                case 2:                                    \
                    return self.z;                         \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static void set(T3& self, size_t index, T value) { \
            switch (index) {                               \
                case 0:                                    \
                    self.x = value;                        \
                    return;                                \
                case 1:                                    \
                    self.y = value;                        \
                    return;                                \
                case 2:                                    \
                    self.z = value;                        \
                    return;                                \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct vector_traits<T4> {                             \
        using value_type = T;                              \
        static constexpr size_t size = 4;                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T4 create(T x, T y, T z, T w) {             \
            return {x, y, z, w};                           \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T4 fill(T v) {                              \
            return {v, v, v, v};                           \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static T get(const T4& self, size_t index) {       \
            switch (index) {                               \
                case 0:                                    \
                    return self.x;                         \
                case 1:                                    \
                    return self.y;                         \
                case 2:                                    \
                    return self.z;                         \
                case 3:                                    \
                    return self.w;                         \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
        }                                                  \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static void set(T4& self, size_t index, T value) { \
            switch (index) {                               \
                case 0:                                    \
                    self.x = value;                        \
                    return;                                \
                case 1:                                    \
                    self.y = value;                        \
                    return;                                \
                case 2:                                    \
                    self.z = value;                        \
                    return;                                \
                case 3:                                    \
                    self.w = value;                        \
                    return;                                \
                default:                                   \
                    KERNEL_FLOAT_UNREACHABLE;              \
            }                                              \
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

template<typename V, size_t N>
struct nested_array {
    static constexpr size_t num_packets = N / vector_size<V>;

    V packets[num_packets];

    KERNEL_FLOAT_INLINE
    V& operator[](size_t i) {
        KERNEL_FLOAT_ASSERT(i < num_packets);
        return packets[i];
    }

    KERNEL_FLOAT_INLINE
    const V& operator[](size_t i) const {
        KERNEL_FLOAT_ASSERT(i < num_packets);
        return packets[i];
    }
};

template<typename V, size_t N>
struct vector_traits<nested_array<V, N>> {
    using self_type = nested_array<V, N>;
    using value_type = vector_value_type<V>;
    static constexpr size_t size = N;

    template<typename... Args>
    KERNEL_FLOAT_INLINE static self_type create(Args&&... args) {
        value_type items[N] = {args...};
        self_type output;

        size_t i = 0;
        for (; i + vector_size<V> - 1 < N; i += vector_size<V>) {
            // How to generalize this?
            output.packets[i / vector_size<V>] = vector_traits<V>::create(items[i], items[i + 1]);
        }

        for (; i < N; i++) {
            vector_traits<V>::set(output.packets[i / vector_size<V>], i % vector_size<V>, items[i]);
        }

        return output;
    }

    KERNEL_FLOAT_INLINE
    static self_type fill(value_type value) {
        self_type output;

        for (size_t i = 0; i < self_type::num_packets; i++) {
            output.packets[i] = vector_traits<V>::fill(value);
        }

        return output;
    }

    KERNEL_FLOAT_INLINE
    static value_type get(const self_type& self, size_t index) {
        KERNEL_FLOAT_ASSERT(index < N);
        return vector_traits<V>::get(self.packets[index / vector_size<V>], index % vector_size<V>);
    }

    KERNEL_FLOAT_INLINE
    static void set(self_type& self, size_t index, value_type value) {
        KERNEL_FLOAT_ASSERT(index < N);
        vector_traits<V>::set(self.packets[index / vector_size<V>], index % vector_size<V>, value);
    }
};

};  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H



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
#ifndef KERNEL_FLOAT_INTERFACE_H
#define KERNEL_FLOAT_INTERFACE_H



namespace kernel_float {

template<typename Output, typename Input>
KERNEL_FLOAT_INLINE vector<Output> broadcast(Input&& input);

template<typename V, typename I>
struct index_proxy {
    using value_type = typename vector_traits<V>::value_type;

    KERNEL_FLOAT_INLINE
    index_proxy(V& storage, I index) : storage_(storage), index_(index) {}

    KERNEL_FLOAT_INLINE
    index_proxy& operator=(value_type value) {
        vector_traits<V>::set(storage_, index_, value);
        return *this;
    }

    KERNEL_FLOAT_INLINE
    operator value_type() const {
        return vector_traits<V>::get(storage_, index_);
    }

  private:
    V& storage_;
    I index_;
};

template<typename V, size_t I>
struct index_proxy<V, const_index<I>> {
    using value_type = typename vector_traits<V>::value_type;

    KERNEL_FLOAT_INLINE
    index_proxy(V& storage, const_index<I>) : storage_(storage) {}

    KERNEL_FLOAT_INLINE
    index_proxy& operator=(value_type value) {
        vector_index<V, I>::set(storage_, value);
        return *this;
    }

    KERNEL_FLOAT_INLINE
    operator value_type() const {
        return vector_index<V, I>::get(storage_);
    }

  private:
    V& storage_;
};

template<typename V>
struct vector {
    using storage_type = V;
    using traits_type = vector_traits<V>;
    using value_type = typename traits_type::value_type;
    static constexpr size_t const_size = traits_type::size;

    vector(const vector&) = default;
    vector(vector&) = default;
    vector(vector&&) = default;

    vector& operator=(const vector&) = default;
    vector& operator=(vector&) = default;
    vector& operator=(vector&&) = default;

    KERNEL_FLOAT_INLINE
    vector() : storage_(traits_type::fill(value_type {})) {}

    KERNEL_FLOAT_INLINE
    vector(storage_type storage) : storage_(storage) {}

    template<
        typename U,
        enabled_t<is_implicit_convertible<vector_value_type<U>, value_type>, int> = 0>
    KERNEL_FLOAT_INLINE vector(U&& init) : vector(broadcast<V, U>(std::forward<U>(init))) {}

    template<typename... Args, enabled_t<sizeof...(Args) == const_size, int> = 0>
    KERNEL_FLOAT_INLINE vector(Args&&... args) : storage_(traits_type::create(args...)) {}

    KERNEL_FLOAT_INLINE
    operator storage_type() const {
        return storage_;
    }

    KERNEL_FLOAT_INLINE
    storage_type& storage() {
        return storage_;
    }

    KERNEL_FLOAT_INLINE
    const storage_type& storage() const {
        return storage_;
    }

    KERNEL_FLOAT_INLINE
    value_type get(size_t index) const {
        return traits_type::get(storage_, index);
    }

    KERNEL_FLOAT_INLINE
    void set(size_t index, value_type value) {
        traits_type::set(storage_, index, value);
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE value_type get(const_index<I>) const {
        return vector_index<V, I>::get(storage_);
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE void set(const_index<I>, value_type value) {
        return vector_index<V, I>::set(storage_, value);
    }

    KERNEL_FLOAT_INLINE
    value_type operator[](size_t index) const {
        return get(index);
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE value_type operator[](const_index<I>) const {
        return get(const_index<I> {});
    }

    KERNEL_FLOAT_INLINE
    index_proxy<V, size_t> operator[](size_t index) {
        return {storage_, index};
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE index_proxy<V, const_index<I>> operator[](const_index<I>) {
        return {storage_, const_index<I> {}};
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return const_size;
    }

  private:
    storage_type storage_;
};

template<typename V>
struct vector_traits<vector<V>> {
    using value_type = vector_value_type<V>;
    static constexpr size_t size = vector_size<V>;

    KERNEL_FLOAT_INLINE
    static vector<V> fill(value_type value) {
        return vector_traits<V>::fill(value);
    }

    template<typename... Args>
    KERNEL_FLOAT_INLINE static vector<V> create(Args... args) {
        return vector_traits<V>::create(args...);
    }

    KERNEL_FLOAT_INLINE
    static value_type get(const vector<V>& self, size_t index) {
        return vector_traits<V>::get(self.storage(), index);
    }

    KERNEL_FLOAT_INLINE
    static void set(vector<V>& self, size_t index, value_type value) {
        vector_traits<V>::set(self.storage(), index, value);
    }
};

template<typename V, size_t I>
struct vector_index<vector<V>, I> {
    using value_type = vector_value_type<V>;

    KERNEL_FLOAT_INLINE
    static value_type get(const vector<V>& self) {
        return vector_index<V, I>::get(self.storage());
    }

    KERNEL_FLOAT_INLINE
    static void set(vector<V>& self, value_type value) {
        vector_index<V, I>::set(self.storage(), value);
    }
};

template<typename V>
struct into_storage_traits<vector<V>> {
    using type = V;

    KERNEL_FLOAT_INLINE
    static constexpr type call(const vector<V>& self) {
        return self.storage();
    }
};

template<typename Output, typename Input, size_t... Is>
struct vector_swizzle<Output, vector<Input>, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static Output call(const vector<Input>& self) {
        return vector_swizzle<Output, Input, index_sequence<Is...>>::call(self.storage());
    }
};

template<typename T, size_t N>
using vec = vector<default_storage_type<T, N, Alignment::Packed>>;

template<typename T, size_t N>
using unaligned_vec = vector<default_storage_type<T, N, Alignment::Minimum>>;

template<typename... Args>
KERNEL_FLOAT_INLINE vec<common_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using value_type = common_t<Args...>;
    using vector_type = default_storage_type<value_type, sizeof...(Args), Alignment::Packed>;
    return vector_traits<vector_type>::create(value_type(args)...);
}

template<typename V>
KERNEL_FLOAT_INLINE vector<into_storage_type<V>> into_vec(V&& input) {
    return into_storage(input);
}

using float32 = float;
using float64 = double;

template<typename T>
using vec1 = vec<T, 1>;
template<typename T>
using vec2 = vec<T, 2>;
template<typename T>
using vec3 = vec<T, 3>;
template<typename T>
using vec4 = vec<T, 4>;
template<typename T>
using vec5 = vec<T, 5>;
template<typename T>
using vec6 = vec<T, 6>;
template<typename T>
using vec7 = vec<T, 7>;
template<typename T>
using vec8 = vec<T, 8>;

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)             \
    template<size_t N>                               \
    using NAME##N = vec<T, N>;                       \
    using NAME##1 = vec<T, 1>;                       \
    using NAME##2 = vec<T, 2>;                       \
    using NAME##3 = vec<T, 3>;                       \
    using NAME##4 = vec<T, 4>;                       \
    using NAME##5 = vec<T, 5>;                       \
    using NAME##6 = vec<T, 6>;                       \
    using NAME##7 = vec<T, 7>;                       \
    using NAME##8 = vec<T, 8>;                       \
    template<size_t N>                               \
    using unaligned_##NAME##X = unaligned_vec<T, N>; \
    using unaligned_##NAME##1 = unaligned_vec<T, 1>; \
    using unaligned_##NAME##2 = unaligned_vec<T, 2>; \
    using unaligned_##NAME##3 = unaligned_vec<T, 3>; \
    using unaligned_##NAME##4 = unaligned_vec<T, 4>; \
    using unaligned_##NAME##5 = unaligned_vec<T, 5>; \
    using unaligned_##NAME##6 = unaligned_vec<T, 6>; \
    using unaligned_##NAME##7 = unaligned_vec<T, 7>; \
    using unaligned_##NAME##8 = unaligned_vec<T, 8>;

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

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_INTERFACE_H
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H



#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>



namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__half, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __half)

template<>
struct vector_traits<__half2> {
    using value_type = __half;
    static constexpr size_t size = 2;

    KERNEL_FLOAT_INLINE
    static __half2 fill(__half value) {
#if KERNEL_FLOAT_ON_DEVICE
        return __half2half2(value);
#else
        return {value, value};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __half2 create(__half low, __half high) {
#if KERNEL_FLOAT_ON_DEVICE
        return __halves2half2(low, high);
#else
        return {low, high};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __half get(__half2 self, size_t index) {
#if KERNEL_FLOAT_ON_DEVICE
        if (index == 0) {
            return __low2half(self);
        } else {
            return __high2half(self);
        }
#else
        if (index == 0) {
            return self.x;
        } else {
            return self.y;
        }
#endif
    }

    KERNEL_FLOAT_INLINE
    static void set(__half2& self, size_t index, __half value) {
        if (index == 0) {
            self.x = value;
        } else {
            self.y = value;
        }
    }
};

template<size_t N>
struct default_storage<__half, N, Alignment::Maximum, enabled_t<(N >= 2)>> {
    using type = nested_array<__half2, N>;
};

template<size_t N>
struct default_storage<__half, N, Alignment::Packed, enabled_t<(N >= 2 && N % 2 == 0)>> {
    using type = nested_array<__half2, N>;
};

#if KERNEL_FLOAT_ON_DEVICE
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
    struct map_helper<ops::NAME<__half>, __half2, __half2> {                        \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 input) { \
            return FUN2(input);                                                     \
        }                                                                           \
    };                                                                              \
    }

KERNEL_FLOAT_FP16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_FP16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_FP16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_FP16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_FP16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_FP16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_FP16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_FP16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

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
    struct zip_helper<ops::NAME<__half>, __half2, __half2, __half2> {                             \
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

KERNEL_FLOAT_FP16_CAST(signed short, __short2half_rn(input), __half2short_rz(input));
KERNEL_FLOAT_FP16_CAST(signed int, __int2half_rn(input), __half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned int, __uint2half_rn(input), __half2uint_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned short, __ushort2half_rn(input), __half2ushort_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

using half = __half;
using float16 = __half;
//KERNEL_FLOAT_TYPE_ALIAS(half, __half)
//KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
//KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
#ifndef KERNEL_FLOAT_SWIZZLE_H
#define KERNEL_FLOAT_SWIZZLE_H



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
#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H




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
        return vector_traits<Output>::create(fun(vector_get<Is>(input))...);
    }
};

template<typename F, typename V, size_t N>
struct map_helper<F, nested_array<V, N>, nested_array<V, N>> {
    KERNEL_FLOAT_INLINE static nested_array<V, N> call(F fun, const nested_array<V, N>& input) {
        return call(fun, input, make_index_sequence<nested_array<V, N>::num_packets> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static nested_array<V, N>
    call(F fun, const nested_array<V, N>& input, index_sequence<Is...>) {
        return {map_helper<F, V, V>::call(fun, input[Is])...};
    }
};
}  // namespace detail

template<typename F, typename Input>
using map_type = default_storage_type<result_t<F, vector_value_type<Input>>, vector_size<Input>>;

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
KERNEL_FLOAT_INLINE Output map(F fun, const Input& input) {
    return detail::map_helper<F, Output, into_storage_type<Input>>::call(fun, into_storage(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                            \
    namespace ops {                                                                      \
    template<typename T>                                                                 \
    struct NAME {                                                                        \
        KERNEL_FLOAT_INLINE T operator()(T input) {                                      \
            return T(EXPR);                                                              \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    template<typename V>                                                                 \
    KERNEL_FLOAT_INLINE vector<into_storage_type<V>> NAME(const V& input) {              \
        return map<ops::NAME<vector_value_type<V>>, V, into_storage_type<V>>({}, input); \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                  \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                             \
    template<typename V>                                              \
    KERNEL_FLOAT_INLINE vector<V> operator OP(const vector<V>& vec) { \
        return NAME(vec);                                             \
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
#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H



namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Left, typename Right, typename = void>
struct zip_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Left& left, const Right& right) {
        return call_with_indices(fun, left, right, make_index_sequence<vector_size<Output>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output
    call_with_indices(F fun, const Left& left, const Right& right, index_sequence<Is...> = {}) {
        return vector_traits<Output>::create(fun(vector_get<Is>(left), vector_get<Is>(right))...);
    }
};

template<typename F, typename V, size_t N>
struct zip_helper<F, nested_array<V, N>, nested_array<V, N>, nested_array<V, N>> {
    KERNEL_FLOAT_INLINE static nested_array<V, N>
    call(F fun, const nested_array<V, N>& left, const nested_array<V, N>& right) {
        return call(fun, left, right, make_index_sequence<nested_array<V, N>::num_packets> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static nested_array<V, N> call(
        F fun,
        const nested_array<V, N>& left,
        const nested_array<V, N>& right,
        index_sequence<Is...>) {
        return {zip_helper<F, V, V, V>::call(fun, left[Is], right[Is])...};
    }
};
};  // namespace detail

template<typename... Ts>
using common_vector_value_type = common_t<vector_value_type<Ts>...>;

template<typename... Ts>
static constexpr size_t common_vector_size = common_size<vector_size<Ts>...>;

template<typename F, typename L, typename R>
using zip_type = default_storage_type<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    common_vector_size<L, R>>;

/**
 * Applies ``fun`` to each pair of two elements from ``left`` and ``right`` and returns a new
 * vector with the results.
 *
 * If ``left`` and ``right`` are not the same size, they will first be broadcast into a
 * common size using ``resize``.
 *
 * Note that this function does **not** cast the input vectors to a common element type. See
 * ``zip_common`` for that functionality.
 */
template<typename F, typename Left, typename Right, typename Output = zip_type<F, Left, Right>>
KERNEL_FLOAT_INLINE vector<Output> zip(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using LeftInput = default_storage_type<vector_value_type<Left>, N>;
    using RightInput = default_storage_type<vector_value_type<Right>, N>;

    return detail::zip_helper<F, Output, LeftInput, RightInput>::call(
        fun,
        broadcast<LeftInput, Left>(std::forward<Left>(left)),
        broadcast<RightInput, Right>(std::forward<Right>(right)));
}

template<typename F, typename L, typename R>
using zip_common_type = default_storage_type<
    result_t<F, common_vector_value_type<L, R>, common_vector_value_type<L, R>>,
    common_vector_size<L, R>>;

/**
 * Applies ``fun`` to each pair of two elements from ``left`` and ``right`` and returns a new
 * vector with the results.
 *
 * If ``left`` and ``right`` are not the same size, they will first be broadcast into a
 * common size using ``resize``.
 *
 * If ``left`` and ``right`` are not of the same type, they will first be case into a common
 * data type. For example, zipping ``float`` and ``double`` first cast vectors to ``double``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {1, 2, 3, 4};
 * vec<long, 1> = {8};
 * vec<long, 5> = zip_common([](auto a, auto b){ return a + b; }, x, y); // [9, 10, 11, 12]
 * ```
 */
template<
    typename F,
    typename Left,
    typename Right,
    typename Output = zip_common_type<F, Left, Right>>
KERNEL_FLOAT_INLINE vector<Output> zip_common(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using C = common_t<vector_value_type<Left>, vector_value_type<Right>>;
    using Input = default_storage_type<C, N>;

    return detail::zip_helper<F, Output, Input, Input>::call(
        fun,
        broadcast<Input, Left>(std::forward<Left>(left)),
        broadcast<Input, Right>(std::forward<Right>(right)));
}

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR)                                                  \
    namespace ops {                                                                             \
    template<typename T>                                                                        \
    struct NAME {                                                                               \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                                     \
            return T(EXPR);                                                                     \
        }                                                                                       \
    };                                                                                          \
    }                                                                                           \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>>               \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> NAME(L&& left, R&& right) { \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right));      \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                               \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const vector<L>& left,                                                    \
        const vector<R>& right) {                                                 \
        return zip_common(ops::NAME<C> {}, left, right);                          \
    }                                                                             \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const vector<L>& left,                                                    \
        const R& right) {                                                         \
        return zip_common(ops::NAME<C> {}, left, right);                          \
    }                                                                             \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE vector<zip_common_type<ops::NAME<C>, L, R>> operator OP(  \
        const L& left,                                                            \
        const vector<R>& right) {                                                 \
        return zip_common(ops::NAME<C> {}, left, right);                          \
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
template<template<typename T> typename F, typename L, typename R>
static constexpr bool vector_assign_allowed =
    common_vector_size<L, R> == vector_size<L> &&
    is_implicit_convertible<
        result_t<
                F<common_t<vector_value_type<L>, vector_value_type<R>>>,
                vector_value_type<L>,
                vector_value_type<R>
        >,
        vector_value_type<L>
    >;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                                        \
    template<                                                                                 \
        typename L,                                                                           \
        typename R,                                                                           \
        typename T = enabled_t<vector_assign_allowed<ops::NAME, L, R>, vector_value_type<L>>> \
    KERNEL_FLOAT_INLINE vector<L>& operator OP(vector<L>& lhs, const R& rhs) {                \
        using F = ops::NAME<T>;                                                               \
        lhs = zip_common<F, const L&, const R&, L>(F {}, lhs.storage(), rhs);                 \
        return lhs;                                                                           \
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

#endif  //KERNEL_FLOAT_BINOPS_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H



#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>







namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(__nv_bfloat16, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_bfloat16)

template<>
struct vector_traits<__nv_bfloat162> {
    using value_type = __nv_bfloat16;
    static constexpr size_t size = 2;

    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 fill(__nv_bfloat16 value) {
#if KERNEL_FLOAT_ON_DEVICE
        return __bfloat162bfloat162(value);
#else
        return {value, value};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __nv_bfloat162 create(__nv_bfloat16 low, __nv_bfloat16 high) {
#if KERNEL_FLOAT_ON_DEVICE
        return __halves2bfloat162(low, high);
#else
        return {low, high};
#endif
    }

    KERNEL_FLOAT_INLINE
    static __nv_bfloat16 get(__nv_bfloat162 self, size_t index) {
#if KERNEL_FLOAT_ON_DEVICE
        if (index == 0) {
            return __low2bfloat16(self);
        } else {
            return __high2bfloat16(self);
        }
#else
        if (index == 0) {
            return self.x;
        } else {
            return self.y;
        }
#endif
    }

    KERNEL_FLOAT_INLINE
    static void set(__nv_bfloat162& self, size_t index, __nv_bfloat16 value) {
        if (index == 0) {
            self.x = value;
        } else {
            self.y = value;
        }
    }
};

template<size_t N>
struct default_storage<__nv_bfloat16, N, Alignment::Maximum, enabled_t<(N >= 2)>> {
    using type = nested_array<__nv_bfloat162, N>;
};

template<size_t N>
struct default_storage<__nv_bfloat16, N, Alignment::Packed, enabled_t<(N >= 2 && N % 2 == 0)>> {
    using type = nested_array<__nv_bfloat162, N>;
};

#if KERNEL_FLOAT_ON_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                             \
    namespace ops {                                                               \
    template<>                                                                    \
    struct NAME<__nv_bfloat16> {                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) {       \
            return FUN1(input);                                                   \
        }                                                                         \
    };                                                                            \
    }                                                                             \
    namespace detail {                                                            \
    template<>                                                                    \
    struct map_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat162, __nv_bfloat162> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                 \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 input) {                    \
            return FUN2(input);                                                   \
        }                                                                         \
    };                                                                            \
    }

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                               \
    template<>                                                                                    \
    struct NAME<__nv_bfloat16> {                                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                                         \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {                               \
            return FUN1(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    namespace detail {                                                                            \
    template<>                                                                                    \
    struct zip_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat162, __nv_bfloat162, __nv_bfloat162> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                                 \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 left, __nv_bfloat162 right) {               \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

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
//KERNEL_FLOAT_TYPE_ALIAS(half, __nv_bfloat16)
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
#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H




namespace kernel_float {

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct range_helper;

template<typename F, typename V, size_t... Is>
struct range_helper<F, V, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static V call(F fun) {
        return vector_traits<V>::create(fun(const_index<Is> {})...);
    }
};
}  // namespace detail

/**
 * Generate vector of length ``N`` by applying the given function ``fun`` to
 * each index ``0...N-1``.
 *
 * Example
 * =======
 * ```
 * // returns [0, 2, 4]
 * vector<float, 3> vec = range<3>([](auto i) { return float(i * 2); });
 * ```
 */
template<
    size_t N,
    typename F,
    typename T = result_t<F, size_t>,
    typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> range(F fun) {
    return detail::range_helper<F, Output>::call(fun);
}

/**
 * Generate vector consisting of the numbers ``0...N-1`` of type ``T``.
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2]
 * vector<float, 3> vec = range<float, 3>();
 * ```
 */
template<typename T, size_t N, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> range() {
    using F = ops::cast<size_t, T>;
    return detail::range_helper<F, Output>::call(F {});
}

/**
 * Generate vector having same size and type as ``V``, but filled with the numbers ``0..N-1``.
 */
template<typename Input, typename Output = into_storage_type<Input>>
KERNEL_FLOAT_INLINE vector<Output> range_like(const Input&) {
    using F = ops::cast<size_t, vector_value_type<Input>>;
    return detail::range_helper<F, Output>::call(F {});
}

/**
 * Generate vector of `N` elements of type `T`
 *
 * Example
 * =======
 * ```
 * // Returns [1.0, 1.0, 1.0]
 * vector<float, 3> = fill(1.0f);
 * ```
 */
template<size_t N = 1, typename T, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> fill(T value) {
    return vector_traits<Output>::fill(value);
}

/**
 * Generate vector having same size and type as ``V``, but filled with the given ``value``.
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> fill_like(const Output&, vector_value_type<Output> value) {
    return vector_traits<Output>::fill(value);
}

/**
 * Generate vector of ``N`` zeros of type ``T``
 *
 * Example
 * =======
 * ```
 * // Returns [0.0, 0.0, 0.0]
 * vector<float, 3> = zeros();
 * ```
 */
template<size_t N = 1, typename T = bool, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> zeros() {
    return vector_traits<Output>::fill(T(0));
}

/**
 * Generate vector having same size and type as ``V``, but filled with zeros.
 *
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> zeros_like(const Output& output = {}) {
    return vector_traits<Output>::fill(0);
}

/**
 * Generate vector of ``N`` ones of type ``T``
 *
 * Example
 * =======
 * ```
 * // Returns [1.0, 1.0, 1.0]
 * vector<float, 3> = ones();
 * ```
 */
template<size_t N = 1, typename T = bool, typename Output = default_storage_type<T, N>>
KERNEL_FLOAT_INLINE vector<Output> ones() {
    return vector_traits<Output>::fill(T(1));
}

/**
 * Generate vector having same size and type as ``V``, but filled with ones.
 *
 */
template<typename Output>
KERNEL_FLOAT_INLINE vector<Output> ones_like(const Output& output = {}) {
    return vector_traits<Output>::fill(1);
}

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct iterate_helper;

template<typename F, typename V>
struct iterate_helper<F, V, index_sequence<>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {}
};

template<typename F, typename V, size_t I, size_t... Rest>
struct iterate_helper<F, V, index_sequence<I, Rest...>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {
        fun(vector_get<I>(input));
        iterate_helper<F, V, index_sequence<Rest...>>::call(fun, input);
    }
};
}  // namespace detail

/**
 * Apply the function ``fun`` for each element from ``input``.
 *
 * Example
 * =======
 * ```
 * for_each(range<3>(), [&](auto i) {
 *    printf("element: %d\n", i);
 * });
 * ```
 */
template<typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(const V& input, F fun) {
    detail::iterate_helper<F, into_storage_type<V>>::call(fun, into_storage(input));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H



namespace kernel_float {
namespace detail {
template<typename F, typename V, typename = void>
struct reduce_helper {
    using value_type = vector_value_type<V>;

    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& input) {
        return call(fun, input, make_index_sequence<vector_size<V>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& vector, index_sequence<0, Is...>) {
        return call(fun, vector, vector_get<0>(vector), index_sequence<Is...> {});
    }

    template<size_t I, size_t... Rest>
    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<I, Rest...>) {
        return call(fun, vector, fun(accum, vector_get<I>(vector)), index_sequence<Rest...> {});
    }

    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<>) {
        return accum;
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
    return detail::reduce_helper<F, into_storage_type<V>>::call(fun, into_storage(input));
}

/**
 * Find the minimum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 0, 2, 1, 0};
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
 * vec<int, 3> x = {5, 0, 2, 1, 0};
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
 * vec<int, 3> x = {5, 0, 2, 1, 0};
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
KERNEL_FLOAT_INLINE bool all(V&& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

/**
 * Check if any element in the given vector ``input`` is non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template<typename V>
KERNEL_FLOAT_INLINE bool any(V&& input) {
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
KERNEL_FLOAT_INLINE int count(V&& input) {
    return sum(cast<int>(cast<bool>(input)));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
