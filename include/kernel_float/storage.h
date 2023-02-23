#ifndef KERNEL_FLOAT_STORAGE_H
#define KERNEL_FLOAT_STORAGE_H

#include "core.h"

namespace kernel_float {
namespace detail {

template<typename ToIndices, typename FromIndices>
struct assign_helper;

template<size_t I, size_t J, size_t... Is, size_t... Js>
struct assign_helper<index_sequence<I, Is...>, index_sequence<J, Js...>> {
    template<typename To, typename From>
    KERNEL_FLOAT_INLINE static void call(To& to, const From& from) {
        to.set(I, from.get(J));
        assign_helper<index_sequence<Is...>, index_sequence<Js...>>::call(to, from);
    }
};

template<>
struct assign_helper<index_sequence<>, index_sequence<>> {
    template<typename To, typename From>
    KERNEL_FLOAT_INLINE static void call(To& to, const From& from) {}
};

#define KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(T, N)                                              \
    template<size_t... Is>                                                                      \
    KERNEL_FLOAT_INLINE vec_storage<T, sizeof...(Is)> get(index_sequence<Is...>) const {        \
        return {this->get(constant_index<Is> {})...};                                           \
    }                                                                                           \
    template<size_t... Is>                                                                      \
    KERNEL_FLOAT_INLINE void set(index_sequence<Is...>, vec_storage<T, sizeof...(Is)> values) { \
        assign_helper<index_sequence<Is...>, make_index_sequence<sizeof...(Is)>>::call(         \
            *this,                                                                              \
            values);                                                                            \
    }

#define KERNEL_FLOAT_STORAGE_ACCESSORS(T, N)                   \
    template<size_t I>                                         \
    KERNEL_FLOAT_INLINE T get(constant_index<I>) const {       \
        return this->get(size_t(I));                           \
    }                                                          \
    template<size_t I>                                         \
    KERNEL_FLOAT_INLINE void set(constant_index<I>, T value) { \
        this->set(size_t(I), value);                           \
    }                                                          \
    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(T, N)

#define KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(FIELD, T, N) \
    KERNEL_FLOAT_INLINE T get(size_t index) const {       \
        return FIELD[index];                              \
    }                                                     \
    KERNEL_FLOAT_INLINE void set(size_t index, T value) { \
        FIELD[index] = value;                             \
    }                                                     \
    KERNEL_FLOAT_STORAGE_ACCESSORS(T, N)

template<typename T, size_t N>
struct vec_storage;

template<typename T>
struct vec_storage<T, 1> {
    KERNEL_FLOAT_INLINE vec_storage(T value) noexcept : value_(value) {}

    KERNEL_FLOAT_INLINE operator T() const noexcept {
        return value_;
    }

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        return value_;
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        value_ = value;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS(T, 1)

  private:
    T value_;
};

template<typename T>
struct vec_storage<T, 2> {
    KERNEL_FLOAT_INLINE vec_storage(T x, T y) noexcept : values_ {x, y} {}
    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(values_, T, 2)

  private:
    T values_[2];
};

template<typename T>
struct vec_storage<T, 3> {
    KERNEL_FLOAT_INLINE vec_storage(T x, T y, T z) noexcept : values_ {x, y, z} {}
    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(values_, T, 3)

  private:
    T values_[3];
};

template<typename T>
struct vec_storage<T, 4> {
    KERNEL_FLOAT_INLINE vec_storage(T x, T y, T z, T w) noexcept : low_ {x, y}, high_ {z, w} {}
    KERNEL_FLOAT_INLINE vec_storage(vec_storage<T, 2> low, vec_storage<T, 2> high) noexcept :
        low_ {low},
        high_ {high} {}

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        if (index < 2) {
            return low_.get(index);
        } else {
            return high_.get(index - 2);
        }
    }
    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        if (index < 2) {
            low_.set(index, value);
        } else {
            high_.set(index - 2, value);
        }
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS(T, 4)

    KERNEL_FLOAT_INLINE vec_storage<T, 2> get(index_sequence<0, 1>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec_storage<T, 2> get(index_sequence<2, 3>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, vec_storage<T, 2> values) {
        low_ = values;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<2, 3>, vec_storage<T, 2> values) {
        high_ = values;
    }

  private:
    vec_storage<T, 2> high_;
    vec_storage<T, 2> low_;
};

template<typename T>
struct vec_storage<T, 5> {
    KERNEL_FLOAT_INLINE vec_storage(T v0, T v1, T v2, T v3, T v4) noexcept :
        values_ {v0, v1, v2, v3, v4} {}
    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(values_, T, 5)

  private:
    T values_[5];
};

template<typename T>
struct vec_storage<T, 6> {
    KERNEL_FLOAT_INLINE vec_storage(T v0, T v1, T v2, T v3, T v4, T v5) noexcept :
        low_ {v0, v1, v2, v3},
        high_ {v4, v5} {}
    KERNEL_FLOAT_INLINE vec_storage(vec_storage<T, 4> low, vec_storage<T, 2> high) noexcept :
        low_ {low},
        high_ {high} {}

    KERNEL_FLOAT_STORAGE_ACCESSORS(T, 6)

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        if (index < 4) {
            return low_.get(index);
        } else {
            return high_.get(index - 4);
        }
    }
    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        if (index < 4) {
            low_.set(index, value);
        } else {
            high_.set(index - 4, value);
        }
    }

    KERNEL_FLOAT_INLINE vec_storage<T, 4> get(index_sequence<0, 1, 2, 3>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec_storage<T, 2> get(index_sequence<4, 5>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1, 2, 3>, vec_storage<T, 4> values) {
        low_ = values;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<4, 5>, vec_storage<T, 2> values) {
        high_ = values;
    }

  private:
    vec_storage<T, 4> low_;
    vec_storage<T, 2> high_;
};

template<typename T>
struct vec_storage<T, 7> {
    KERNEL_FLOAT_INLINE vec_storage(T v0, T v1, T v2, T v3, T v4, T v5, T v6) noexcept :
        values_ {v0, v1, v2, v3, v4, v5, v6} {}
    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(values_, T, 7)

  private:
    T values_[7];
};

template<typename T>
struct vec_storage<T, 8> {
    KERNEL_FLOAT_INLINE vec_storage(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept :
        low_ {v0, v1, v2, v3},
        high_ {v4, v5, v6, v7} {}
    KERNEL_FLOAT_INLINE vec_storage(vec_storage<T, 4> low, vec_storage<T, 4> high) noexcept :
        low_ {low},
        high_ {high} {}
    KERNEL_FLOAT_STORAGE_ACCESSORS(T, 8)

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        if (index < 4) {
            return low_.get(index);
        } else {
            return high_.get(index - 4);
        }
    }
    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        if (index < 4) {
            low_.set(index, value);
        } else {
            high_.set(index - 4, value);
        }
    }

    KERNEL_FLOAT_INLINE vec_storage<T, 4> get(index_sequence<0, 1, 2, 3>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec_storage<T, 4> get(index_sequence<4, 5, 6, 7>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1, 2, 3>, vec_storage<T, 4> values) {
        low_ = values;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<4, 5, 6, 7>, vec_storage<T, 4> values) {
        high_ = values;
    }

  private:
    vec_storage<T, 4> low_;
    vec_storage<T, 4> high_;
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, I, FIELD)     \
    KERNEL_FLOAT_INLINE T get(constant_index<I>) const {       \
        return FIELD;                                          \
    }                                                          \
    KERNEL_FLOAT_INLINE void set(constant_index<I>, T value) { \
        FIELD = value;                                         \
    }

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T2, T3, T4)                                            \
    template<>                                                                                    \
    struct vec_storage<T, 2> {                                                                    \
        KERNEL_FLOAT_INLINE vec_storage(T x, T y) noexcept : vector_ {make_##T2(x, y)} {}         \
        KERNEL_FLOAT_INLINE vec_storage(T2 xy) noexcept : vector_ {xy} {}                         \
        KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, T, 2)                                        \
        KERNEL_FLOAT_INLINE operator T2() const noexcept {                                        \
            return vector_;                                                                       \
        }                                                                                         \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 0, vector_.x)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 1, vector_.y)                                    \
      private:                                                                                    \
        static_assert(sizeof(T) * 2 == sizeof(T2), "invalid size");                               \
        union {                                                                                   \
            T2 vector_;                                                                           \
            T array_[2];                                                                          \
        };                                                                                        \
    };                                                                                            \
    template<>                                                                                    \
    struct vec_storage<T, 3> {                                                                    \
        KERNEL_FLOAT_INLINE vec_storage(T x, T y, T z) noexcept : vector_ {make_##T3(x, y, z)} {} \
        KERNEL_FLOAT_INLINE vec_storage(T3 xyz) noexcept : vector_ {xyz} {}                       \
        KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, T, 3)                                        \
        KERNEL_FLOAT_INLINE operator T3() const noexcept {                                        \
            return vector_;                                                                       \
        }                                                                                         \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 0, vector_.x)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 1, vector_.y)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 2, vector_.z)                                    \
      private:                                                                                    \
        static_assert(sizeof(T) * 3 == sizeof(T3), "invalid size");                               \
        union {                                                                                   \
            T3 vector_;                                                                           \
            T array_[3];                                                                          \
        };                                                                                        \
    };                                                                                            \
    template<>                                                                                    \
    struct vec_storage<T, 4> {                                                                    \
        KERNEL_FLOAT_INLINE vec_storage(T x, T y, T z, T w) noexcept :                            \
            vector_ {make_##T4(x, y, z, w)} {}                                                    \
        KERNEL_FLOAT_INLINE vec_storage(T2 xy, T2 zw) noexcept :                                  \
            vec_storage {xy.x, xy.y, zw.x, zw.y} {}                                               \
        KERNEL_FLOAT_INLINE vec_storage(T4 xyzw) noexcept : vector_ {xyzw} {}                     \
        KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, T, 4)                                        \
        KERNEL_FLOAT_INLINE operator T4() const noexcept {                                        \
            return vector_;                                                                       \
        }                                                                                         \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 0, vector_.x)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 1, vector_.y)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 2, vector_.z)                                    \
        KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, 3, vector_.w)                                    \
      private:                                                                                    \
        static_assert(sizeof(T) * 4 == sizeof(T4), "invalid size");                               \
        union {                                                                                   \
            T4 vector_;                                                                           \
            T array_[4];                                                                          \
        };                                                                                        \
    };

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(char, char2, char3, char4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(short, short2, short3, short4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(int, int2, int3, int4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long, long2, long3, long4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long long, longlong2, longlong3, longlong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned char, uchar2, uchar3, uchar4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned short, ushort2, ushort3, ushort4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned int, uint2, uint3, uint4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long, ulong2, ulong3, ulong4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong2, ulonglong3, ulonglong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(float, float2, float3, float4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(double, double2, double3, double4)

template<typename T>
struct into_vec_helper;

template<typename T>
struct into_vec_helper<const T>: into_vec_helper<T> {};

template<typename T>
struct into_vec_helper<const T&>: into_vec_helper<T> {};

template<typename T>
struct into_vec_helper<T&&>: into_vec_helper<T> {};

template<typename T, size_t N>
struct into_vec_helper<vec<T, N>> {
    using value_type = T;
    static constexpr size_t size = N;

    KERNEL_FLOAT_INLINE static vec<T, N> call(vec<T, N> input) {
        return input;
    }
};

template<typename T>
struct is_vec_helper {
    static constexpr bool value = false;
};

template<typename T, size_t N>
struct is_vec_helper<vec<T, N>> {
    static constexpr bool value = true;
};
};  // namespace detail

template<typename T>
static constexpr size_t into_vec_size = detail::into_vec_helper<T>::size;

template<typename T>
using into_vec_value_t = typename detail::into_vec_helper<T>::value_type;

template<typename T>
using into_vec_t = vec<into_vec_value_t<T>, into_vec_size<T>>;

template<typename T>
static constexpr bool is_vec = detail::is_vec_helper<T>::value;

template<typename T>
KERNEL_FLOAT_INLINE into_vec_t<T> into_vec(T&& input) {
    return detail::into_vec_helper<T>::call(std::forward<T>(input));
}

};  // namespace kernel_float

#endif  //KERNEL_FLOAT_STORAGE_H
