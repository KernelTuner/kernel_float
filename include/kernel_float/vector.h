#ifndef KERNEL_FLOAT_VECTOR_H
#define KERNEL_FLOAT_VECTOR_H

#include "base.h"
#include "broadcast.h"
#include "macros.h"
#include "reduce.h"
#include "unops.h"

namespace kernel_float {

template<typename Derived, typename T, size_t N>
struct vector_extension {};

template<typename T, typename E, template<typename, size_t> class S>
struct vector: vector_extension<vector<T, E, S>, T, E::value> {
    using value_type = T;
    using extent_type = E;
    using storage_type = S<T, E::value>;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return E::value;
    }

    vector(const vector&) = default;

    KERNEL_FLOAT_INLINE
    vector(storage_type storage) : storage_(storage) {}

    template<typename... Args, enabled_t<sizeof...(Args) == size() && size() >= 2, int> = 0>
    KERNEL_FLOAT_INLINE vector(Args&&... args) : storage_ {std::forward<Args>(args)...} {}

    template<
        typename U,
        typename F,
        enabled_t<
            is_implicit_convertible<U, value_type> && is_vector_broadcastable<F, extent_type>,
            int> = 0>
    KERNEL_FLOAT_INLINE vector(const vector<U, F>& input) :
        vector(convert<T>(input, extent_type {})) {}

    template<
        typename U,
        typename F,
        enabled_t<
            !is_implicit_convertible<U, value_type> && is_vector_broadcastable<F, extent_type>,
            int> = 0>
    explicit KERNEL_FLOAT_INLINE vector(const vector<U, F>& input) :
        vector(convert<T>(input, extent_type {})) {}

    KERNEL_FLOAT_INLINE vector(const value_type& input = {}) :
        vector(convert<T>(input, extent_type {})) {}

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
        return storage_.data() + size();
    }

    KERNEL_FLOAT_INLINE
    const T* end() const {
        return storage_.data() + size();
    }

    KERNEL_FLOAT_INLINE
    const T* cend() const {
        return storage_.data() + size();
    }

    KERNEL_FLOAT_INLINE
    T& at(size_t x) {
        return *(data() + x);
    }

    KERNEL_FLOAT_INLINE
    const T& at(size_t x) const {
        return *(data() + x);
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

template<typename Derived, typename T>
struct vector_extension<Derived, T, 1> {
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

template<typename T>
using scalar = vector<T, extent<1>>;

template<typename T, size_t N>
using vec = vector<T, extent<N>>;

template<typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return vector_storage<T, sizeof...(Args)> {T {args}...};
};

}  // namespace kernel_float

#endif
