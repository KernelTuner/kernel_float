#ifndef KERNEL_FLOAT_BASE_H
#define KERNEL_FLOAT_BASE_H

#include "base.h"
#include "broadcast.h"
#include "macros.h"
#include "reduce.h"
#include "unops.h"

namespace kernel_float {

template<typename T, typename E, template<typename, size_t> class S>
struct tensor {
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
    static constexpr size_t stride(size_t axis) {
        return E::stride(axis);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t linearize_index(ndindex_type index) {
        return E::ravel_index(index);
    }

    template<typename... Args, enabled_t<sizeof...(Args) == volume, int> = 0>
    KERNEL_FLOAT_INLINE tensor(Args&&... args) : storage_ {std::forward<Args>(args)...} {}

    KERNEL_FLOAT_INLINE
    tensor(T init = {}) {
        for (size_t i = 0; i < size(); i++) {
            storage_[i] = init;
        }
    }

    KERNEL_FLOAT_INLINE
    tensor(storage_type storage) : storage_(storage) {}

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

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T) \
    using k##NAME = scalar<T>;           \
    template<size_t N>                   \
    using NAME##X = vec<T, N>;           \
    using NAME##1 = vec<T, 1>;           \
    using NAME##2 = vec<T, 2>;           \
    using NAME##3 = vec<T, 3>;           \
    using NAME##4 = vec<T, 4>;           \
    using NAME##5 = vec<T, 5>;           \
    using NAME##6 = vec<T, 6>;           \
    using NAME##7 = vec<T, 7>;           \
    using NAME##8 = vec<T, 8>;

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

#endif