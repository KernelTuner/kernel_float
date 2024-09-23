#ifndef KERNEL_FLOAT_BASE_H
#define KERNEL_FLOAT_BASE_H

#include "macros.h"
#include "meta.h"

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
struct alignas(Alignment) aligned_array<T, 1, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T item = {}) : item_(item) {}

    KERNEL_FLOAT_INLINE
    operator T() const {
        return item_;
    }

    KERNEL_FLOAT_INLINE
    T* data() {
        return &item_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return &item_;
    }

    T item_ = {};
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
    static constexpr size_t number_of_elements = N;
};

template<typename E>
static constexpr size_t extent_size = E::number_of_elements;

namespace detail {
// Indicates that elements of type `T` offer less precision than floats, thus operations
// on elements of type `T` can be performed by upcasting them to ` float`.
template<typename T>
struct allow_float_fallback {
    static constexpr bool value = false;
};

template<>
struct allow_float_fallback<float> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename T>
struct into_vector_impl {
    using value_type = T;
    using extent_type = extent<1>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, 1> call(const T& input) {
        return vector_storage<T, 1> {input};
    }
};

template<typename T, size_t N>
struct into_vector_impl<T[N]> {
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
struct into_vector_impl<const V>: into_vector_impl<V> {};

template<typename V>
struct into_vector_impl<V&>: into_vector_impl<V> {};

template<typename V>
struct into_vector_impl<const V&>: into_vector_impl<V> {};

template<typename V>
struct into_vector_impl<V&&>: into_vector_impl<V> {};

template<typename T, size_t N, size_t A>
struct into_vector_impl<aligned_array<T, N, A>> {
    using value_type = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const aligned_array<T, N, A>& input) {
        return input;
    }
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T1, T2, T3, T4) \
    template<>                                             \
    struct into_vector_impl<::T1> {                        \
        using value_type = T;                              \
        using extent_type = extent<1>;                     \
                                                           \
        KERNEL_FLOAT_INLINE                                \
        static vector_storage<T, 1> call(::T1 v) {         \
            return {v.x};                                  \
        }                                                  \
    };                                                     \
                                                           \
    template<>                                             \
    struct into_vector_impl<::T2> {                        \
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
    struct into_vector_impl<::T3> {                        \
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
    struct into_vector_impl<::T4> {                        \
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

template<typename T, typename E, typename S = vector_storage<T, E::number_of_elements>>
struct vector;

template<typename T, typename E, typename S>
struct into_vector_impl<vector<T, E, S>> {
    using value_type = T;
    using extent_type = E;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, extent_size<E>> call(const vector<T, E, S>& input) {
        return input.storage();
    }
};

template<typename T>
struct preferred_vector_size {
    static constexpr size_t value = 1;
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
using vector_value_type = typename into_vector_impl<V>::value_type;

template<typename V>
using vector_extent_type = typename into_vector_impl<V>::extent_type;

template<typename V>
static constexpr size_t vector_size = extent_size<vector_extent_type<V>>;

template<typename V>
using into_vector_type = vector<vector_value_type<V>, vector_extent_type<V>>;

template<typename V>
using vector_storage_type = vector_storage<vector_value_type<V>, vector_size<V>>;

template<typename... Vs>
using promoted_vector_value_type = promote_t<vector_value_type<Vs>...>;

template<typename V>
KERNEL_FLOAT_INLINE vector_storage_type<V> into_vector_storage(V&& input) {
    return into_vector_impl<V>::call(std::forward<V>(input));
}

}  // namespace kernel_float

#endif
