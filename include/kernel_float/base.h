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

template<size_t N>
struct extent {
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

template<typename V>
struct vector_traits;

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
