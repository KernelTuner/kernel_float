#ifndef KERNEL_FLOAT_STORAGE_H
#define KERNEL_FLOAT_STORAGE_H

#include "meta.h"

namespace kernel_float {

template<typename V>
struct vector_traits {};

template<typename V, typename T, size_t N>
struct default_vector_traits {
    using type = V;
    using value_type = T;
    static constexpr size_t size = N;

    template<typename Input>
    KERNEL_FLOAT_INLINE static V call(Input&& input) {
        return V {std::forward<Input>(input)};
    }
};

template<typename V>
struct vector_traits<V&>: vector_traits<V> {};

template<typename V>
struct vector_traits<const V&>: vector_traits<V> {};

template<typename V>
struct vector_traits<V&&>: vector_traits<V> {};

template<typename V>
using vector_value_type = typename vector_traits<V>::value_type;

template<typename V>
static constexpr size_t vector_size = vector_traits<V>::size;

template<typename V>
using into_vector_type = typename vector_traits<V>::type;

template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_vector(V&& input) {
    return vector_traits<V>::call(std::forward<V>(input));
}

template<typename Storage>
struct vector;

namespace detail {
template<typename A, typename B = into_vector_type<A>>
struct is_vector_helper {
    static constexpr bool value = false;
};

template<typename A>
struct is_vector_helper<A, A> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename V>
static constexpr bool is_vector = detail::is_vector_helper<decay_t<V>>::value;

template<typename T, size_t N, typename = void>
struct default_vector_storage {};

template<typename T, size_t N>
using vector_storage = into_vector_type<typename default_vector_storage<T, N>::type>;

template<typename S, size_t I, typename = void>
struct vector_accessor {
    KERNEL_FLOAT_INLINE static auto get(const S& storage) {
        return storage.get(I);
    }

    template<typename Value>
    KERNEL_FLOAT_INLINE static void set(S& storage, Value&& value) {
        storage.set(I, value);
    }
};

template<typename T>
struct vector_empty {
    KERNEL_FLOAT_INLINE vector_empty(T value = {}) {}

    KERNEL_FLOAT_INLINE void get(size_t index) const noexcept {
        while (1)
            ;  // TODO: throw error
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) const noexcept {
        while (1)
            ;  // TODO: throw error
    }
};

template<typename T>
struct vector_traits<vector_empty<T>>: default_vector_traits<vector_empty<T>, T, 0> {};

template<typename T>
struct vector_scalar {
    KERNEL_FLOAT_INLINE vector_scalar(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE operator T() const {
        return value_;
    }

    KERNEL_FLOAT_INLINE T get(size_t index) const noexcept {
        return value_;
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) noexcept {
        value_ = value;
    }

  private:
    T value_;
};

template<typename T>
struct vector_traits<vector_scalar<T>>: default_vector_traits<vector_scalar<T>, T, 1> {};

template<typename T, size_t N>
struct vector_array_base {
    KERNEL_FLOAT_INLINE T get(size_t index) const noexcept {
        return items_[index];
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) const noexcept {
        items_[index] = value;
    }

    KERNEL_FLOAT_INLINE T* begin() {
        return items_;
    }

    KERNEL_FLOAT_INLINE T* end() {
        return items_ + N;
    }

    KERNEL_FLOAT_INLINE const T* begin() const {
        return items_;
    }

    KERNEL_FLOAT_INLINE const T* end() const {
        return items_ + N;
    }

    T items_[N];
};

template<typename T, size_t N>
struct vector_array {};

template<typename T>
struct vector_array<T, 1>: vector_array_base<T, 1> {
    KERNEL_FLOAT_INLINE vector_array(T value = {}) : vector_array_base<T, 1> {value} {};
};

template<typename T>
struct vector_array<T, 2>: vector_array_base<T, 2> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1) : vector_array_base<T, 2> {v0, v1} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v} {};
};

template<typename T>
struct vector_array<T, 3>: vector_array_base<T, 3> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2) : vector_array_base<T, 3> {v0, v1, v2} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v} {};
};

template<typename T>
struct vector_array<T, 4>: vector_array_base<T, 4> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2, T v3) :
        vector_array_base<T, 4> {v0, v1, v2, v3} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v, v} {};
};

template<typename T>
struct vector_array<T, 5>: vector_array_base<T, 5> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2, T v3, T v4) :
        vector_array_base<T, 5> {v0, v1, v2, v3, v4} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v, v, v} {};
};

template<typename T>
struct vector_array<T, 6>: vector_array_base<T, 6> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2, T v3, T v4, T v5) :
        vector_array_base<T, 6> {v0, v1, v2, v3, v4, v5} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v, v, v, v} {};
};

template<typename T>
struct vector_array<T, 7>: vector_array_base<T, 7> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2, T v3, T v4, T v5, T v6) :
        vector_array_base<T, 7> {v0, v1, v2, v3, v4, v5, v6} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v, v, v, v, v} {};
};

template<typename T>
struct vector_array<T, 8>: vector_array_base<T, 8> {
    KERNEL_FLOAT_INLINE vector_array(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) :
        vector_array_base<T, 8> {v0, v1, v2, v3, v4, v5, v6, v7} {};
    KERNEL_FLOAT_INLINE vector_array(T v = {}) : vector_array {v, v, v, v, v, v, v, v} {};
};

template<typename T, size_t N>
struct vector_traits<vector_array<T, N>>: default_vector_traits<vector_array<T, N>, T, N> {};

template<typename T, size_t N, size_t M>
struct vector_compound_base {
    static constexpr size_t low_size = N;
    static constexpr size_t high_size = M;

    vector_compound_base() = default;
    KERNEL_FLOAT_INLINE vector_compound_base(vector_storage<T, N> low, vector_storage<T, M> high) :
        low_(low),
        high_(high) {}

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        if (index < N) {
            return low_.get(index);
        } else {
            return high_.get(index - N);
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        if (index < N) {
            low_.set(index, value);
        } else {
            high_.set(index - N, value);
        }
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE T get(const_index<I>) const {
        return vector_accessor<vector_compound_base, I>::get(*this);
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE void set(const_index<I>, T value) {
        vector_accessor<vector_compound_base, I>::set(*this, value);
    }

    KERNEL_FLOAT_INLINE vector_storage<T, N>& low() {
        return low_;
    }

    KERNEL_FLOAT_INLINE vector_storage<T, M>& high() {
        return high_;
    }

    KERNEL_FLOAT_INLINE const vector_storage<T, N>& low() const {
        return low_;
    }

    KERNEL_FLOAT_INLINE const vector_storage<T, M>& high() const {
        return high_;
    }

  private:
    vector_storage<T, N> low_;
    vector_storage<T, M> high_;
};

template<typename T, size_t N, size_t M, size_t I>
struct vector_accessor<vector_compound_base<T, N, M>, I, enabled_t<(I < N)>> {
    KERNEL_FLOAT_INLINE static T get(const vector_compound_base<T, N, M>& storage) {
        return storage.low().get(const_index<I> {});
    }

    KERNEL_FLOAT_INLINE static void set(vector_compound_base<T, N, M>& storage, T value) {
        storage.low().set(const_index<I> {}, value);
    }
};

template<typename T, size_t N, size_t M, size_t I>
struct vector_accessor<vector_compound_base<T, N, M>, I, enabled_t<(I >= N && I < N + M)>> {
    KERNEL_FLOAT_INLINE static T get(const vector_compound_base<T, N, M>& storage) {
        return storage.high().get(const_index<I - N> {});
    }

    KERNEL_FLOAT_INLINE static void set(vector_compound_base<T, N, M>& storage, T value) {
        storage.high().set(const_index<I - N> {}, value);
    }
};

template<typename T, size_t N>
struct vector_compound;

template<typename T>
struct vector_compound<T, 2>: vector_compound_base<T, 1, 1> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 1> low, vector_storage<T, 1> high) :
        vector_compound_base<T, 1, 1>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1) : vector_compound_base<T, 1, 1>({v0}, {v1}) {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v} {}
};

template<typename T>
struct vector_compound<T, 3>: vector_compound_base<T, 2, 1> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 2> low, vector_storage<T, 1> high) :
        vector_compound_base<T, 2, 1>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2) : vector_compound {{v0, v1}, {v2}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v} {}
};

template<typename T>
struct vector_compound<T, 4>: vector_compound_base<T, 2, 2> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 2> low, vector_storage<T, 2> high) :
        vector_compound_base<T, 2, 2>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2, T v3) :
        vector_compound {{v0, v1}, {v2, v3}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v, v} {}
};

template<typename T>
struct vector_compound<T, 5>: vector_compound_base<T, 4, 1> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 4> low, vector_storage<T, 1> high) :
        vector_compound_base<T, 4, 1>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2, T v3, T v4) :
        vector_compound {{v0, v1, v2, v3}, {v4}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v, v, v} {}
};

template<typename T>
struct vector_compound<T, 6>: vector_compound_base<T, 4, 2> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 4> low, vector_storage<T, 2> high) :
        vector_compound_base<T, 4, 2>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2, T v3, T v4, T v5) :
        vector_compound {{v0, v1, v2, v3}, {v4, v5}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v, v, v, v} {}
};

template<typename T>
struct vector_compound<T, 7>: vector_compound_base<T, 4, 3> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 4> low, vector_storage<T, 3> high) :
        vector_compound_base<T, 4, 3>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2, T v3, T v4, T v5, T v6) :
        vector_compound {{v0, v1, v2, v3}, {v4, v5, v6}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v, v, v, v, v} {}
};

template<typename T>
struct vector_compound<T, 8>: vector_compound_base<T, 4, 4> {
    vector_compound() = default;
    KERNEL_FLOAT_INLINE vector_compound(vector_storage<T, 4> low, vector_storage<T, 4> high) :
        vector_compound_base<T, 4, 4>(low, high) {}
    KERNEL_FLOAT_INLINE vector_compound(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) :
        vector_compound {{v0, v1, v2, v3}, {v4, v5, v6, v7}} {}
    KERNEL_FLOAT_INLINE vector_compound(T v) : vector_compound {v, v, v, v, v, v, v, v} {}
};

template<typename T, size_t N>
struct vector_traits<vector_compound<T, N>>: default_vector_traits<vector_compound<T, N>, T, N> {};

template<typename T>
struct default_vector_storage<T, 0> {
    using type = vector_empty<T>;
};

template<typename T>
struct default_vector_storage<T, 1> {
    using type = vector_scalar<T>;
};

template<typename T, size_t N>
struct default_vector_storage<T, N> {
    using type = vector_compound<T, N>;
};

template<typename T, size_t N, typename TN>
struct vector_union_base {
    KERNEL_FLOAT_INLINE vector_union_base(TN vector) : vector_(vector) {}

    KERNEL_FLOAT_INLINE operator TN() const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE T get(size_t index) const {
        return items_[index];
    }

    KERNEL_FLOAT_INLINE void set(size_t index, T value) {
        items_[index] = value;
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE T get(const_index<I>) const {
        return vector_accessor<vector_union_base, I>::get(*this);
    }

    template<size_t I>
    KERNEL_FLOAT_INLINE void set(const_index<I>, T value) {
        vector_accessor<vector_union_base, I>::set(*this, value);
    }

  private:
    static_assert(sizeof(T) * N == sizeof(TN), "invalid size");

    union {
        T items_[N];
        TN vector_;
    };
};

template<typename T, size_t N, typename TN>
struct vector_union;

template<typename T, size_t N, typename TN>
struct vector_traits<vector_union<T, N, TN>>:
    default_vector_traits<vector_union<T, N, TN>, T, N> {};

template<typename T, typename T2>
struct vector_union<T, 2, T2>: vector_union_base<T, 2, T2> {
    KERNEL_FLOAT_INLINE vector_union(T2 vector) : vector_union_base<T, 2, T2> {vector} {}
    KERNEL_FLOAT_INLINE vector_union(T v0, T v1) : vector_union {T2 {v0, v1}} {}
    KERNEL_FLOAT_INLINE vector_union(T v = {}) : vector_union {v, v} {}
};

template<typename T, typename T3>
struct vector_union<T, 3, T3>: vector_union_base<T, 3, T3> {
    KERNEL_FLOAT_INLINE vector_union(T3 vector) : vector_union_base<T, 3, T3> {vector} {}
    KERNEL_FLOAT_INLINE vector_union(T v0, T v1, T v2) : vector_union {T3 {v0, v1, v2}} {}
    KERNEL_FLOAT_INLINE vector_union(T v = {}) : vector_union {v, v, v} {}
};

template<typename T, typename T4>
struct vector_union<T, 4, T4>: vector_union_base<T, 4, T4> {
    KERNEL_FLOAT_INLINE vector_union(T4 vector) : vector_union_base<T, 4, T4> {vector} {}
    KERNEL_FLOAT_INLINE vector_union(T v0, T v1, T v2, T v3) : vector_union {T4 {v0, v1, v2, v3}} {}
    KERNEL_FLOAT_INLINE vector_union(T v = {}) : vector_union {v, v, v, v} {}
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T2, T3, T4)                  \
    template<>                                                          \
    struct vector_traits<T>: vector_traits<vector_scalar<T>> {};        \
    template<>                                                          \
    struct vector_traits<T2>: vector_traits<vector_union<T, 2, T2>> {}; \
    template<>                                                          \
    struct vector_traits<T3>: vector_traits<vector_union<T, 3, T3>> {}; \
    template<>                                                          \
    struct vector_traits<T4>: vector_traits<vector_union<T, 4, T4>> {}; \
                                                                        \
    template<>                                                          \
    struct default_vector_storage<T, 1> {                               \
        using type = into_vector_type<T>;                               \
    };                                                                  \
    template<>                                                          \
    struct default_vector_storage<T, 2> {                               \
        using type = into_vector_type<T2>;                              \
    };                                                                  \
    template<>                                                          \
    struct default_vector_storage<T, 3> {                               \
        using type = into_vector_type<T3>;                              \
    };                                                                  \
    template<>                                                          \
    struct default_vector_storage<T, 4> {                               \
        using type = into_vector_type<T4>;                              \
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

template<>
struct vector_traits<bool>: vector_traits<vector_scalar<bool>> {};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_STORAGE_H
