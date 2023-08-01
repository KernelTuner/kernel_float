#ifndef KERNEL_FLOAT_VECTOR_H
#define KERNEL_FLOAT_VECTOR_H

#include "base.h"
#include "broadcast.h"
#include "macros.h"
#include "reduce.h"
#include "unops.h"

namespace kernel_float {

template<typename T, typename E, class S>
struct vector: S {
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
    KERNEL_FLOAT_INLINE vector(U&& input) : storage_type(convert_storage<T, E::size>(input)) {}

    template<typename U, enabled_t<!is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE explicit vector(U&& input) :
        storage_type(convert_storage<T, E::size>(input)) {}

    // List of `N` (where N >= 2), simply pass forward to the storage
    template<
        typename A,
        typename B,
        typename... Rest,
        typename = enabled_t<sizeof...(Rest) + 2 == E::size>>
    KERNEL_FLOAT_INLINE vector(const A& a, const B& b, const Rest&... rest) :
        storage_type {a, b, rest...} {}

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

    KERNEL_FLOAT_INLINE
    const T* cdata() const {
        return this->data();
    }

    KERNEL_FLOAT_INLINE
    T* begin() {
        return this->data();
    }

    KERNEL_FLOAT_INLINE
    const T* begin() const {
        return this->data();
    }

    KERNEL_FLOAT_INLINE
    const T* cbegin() const {
        return this->data();
    }

    KERNEL_FLOAT_INLINE
    T* end() {
        return this->data() + size();
    }

    KERNEL_FLOAT_INLINE
    const T* end() const {
        return this->data() + size();
    }

    KERNEL_FLOAT_INLINE
    const T* cend() const {
        return this->data() + size();
    }

    KERNEL_FLOAT_INLINE
    T& at(size_t x) {
        return *(this->data() + x);
    }

    KERNEL_FLOAT_INLINE
    const T& at(size_t x) const {
        return *(this->data() + x);
    }

    KERNEL_FLOAT_INLINE
    T get(size_t x) const {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    void set(size_t x, T value) {
        at(x) = std::move(value);
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t x) {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t x) const {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    T& operator()(size_t x) {
        return at(x);
    }

    KERNEL_FLOAT_INLINE
    const T& operator()(size_t x) const {
        return at(x);
    }

    template<typename R, RoundingMode Mode = RoundingMode::ANY>
    KERNEL_FLOAT_INLINE vector<T, E2> cast() const {
        return kernel_float::cast<R, Mode>(*this);
    }

    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> broadcast(extent<N> new_size = {}) const {
        return kernel_float::broadcast(*this, new_size);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE vector<result_t<F, T>, E> map(F fun = {}) const {
        return kernel_float::map(fun, *this);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun = {}) const {
        return kernel_float::reduce(fun, *this);
    }

  private:
    storage_type storage_;
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

template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_vector(V&& input) {
    return into_vector_traits<V>::call(std::forward<V>(input));
}

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

template<typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return vector_storage<T, sizeof...(Args)> {T {args}...};
};

}  // namespace kernel_float

#endif
