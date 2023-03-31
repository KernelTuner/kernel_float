#ifndef KERNEL_FLOAT_STORAGE
#define KERNEL_FLOAT_STORAGE

#include "meta.h"

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
    static constexpr size_t num_packets = (N + vector_size<V> - 1) / vector_size<V>;
    static_assert(num_packets * vector_size<V> >= N, "internal error");

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