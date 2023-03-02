#ifndef KERNEL_FLOAT_INTERFACE_H
#define KERNEL_FLOAT_INTERFACE_H

#include "binops.h"
#include "iterate.h"
#include "unops.h"

namespace kernel_float {
template<typename Storage, typename Index>
struct vector_index {
    using value_type = vector_value_type<Storage>;

    KERNEL_FLOAT_INLINE vector_index(Storage& storage, Index index) :
        storage_(storage),
        index_(index) {}

    KERNEL_FLOAT_INLINE vector_index& operator=(value_type value) {
        storage_.set(index_, value);
        return *this;
    }

    KERNEL_FLOAT_INLINE operator value_type() const {
        return storage_.get(index_);
    }

    KERNEL_FLOAT_INLINE value_type operator()() const {
        return storage_.get(index_);
    }

  private:
    Storage& storage_;
    Index index_;
};

template<typename Storage>
struct vector: public Storage {
    using storage_type = Storage;
    using value_type = vector_value_type<Storage>;
    static constexpr size_t const_size = vector_size<Storage>;

    /**
     * Construct vector where elements are default initialized.
     */
    vector() = default;

    vector(const vector&) = default;
    vector(vector&) = default;
    vector(vector&&) = default;
    KERNEL_FLOAT_INLINE vector(const Storage& storage) : Storage(storage) {}

    vector& operator=(const vector&) = default;
    vector& operator=(vector&) = default;
    vector& operator=(vector&&) = default;
    KERNEL_FLOAT_INLINE vector& operator=(const Storage& s) {
        storage() = s;
        return *this;
    }

    /**
     * Construct vector from ``N`` argumens that can be converted to type ``T``.
     */
    template<typename... Args>
    KERNEL_FLOAT_INLINE vector(Args&&... args) : Storage(args...) {}

    /**
     * Construct vector from another vector of elements type ``U`` where ``U``
     * should be convertible to ``T``. If this is not the case, use ``cast``.
     */
    template<
        typename V,
        typename = enabled_t<
            (vector_size<V> == const_size || vector_size<V> == 1)
            && is_implicit_convertible<vector_value_type<V>, value_type>>>
    KERNEL_FLOAT_INLINE vector(V&& that) : Storage(broadcast<Storage>(std::forward<V>(that))) {}

    KERNEL_FLOAT_INLINE const Storage& storage() const noexcept {
        return *this;
    }

    KERNEL_FLOAT_INLINE Storage& storage() noexcept {
        return *this;
    }

    /**
     * Returns the number of elements.
     */
    KERNEL_FLOAT_INLINE
    size_t size() const noexcept {
        return const_size;
    }

    /**
     * Returns a reference to the ``index``-th item.
     */
    template<typename I>
    KERNEL_FLOAT_INLINE vector_index<Storage, I> operator[](I index) noexcept {
        return {*this, index};
    }

    /**
     * Returns a reference to the ``index``-th item.
     */
    template<typename I>
    KERNEL_FLOAT_INLINE value_type operator[](I index) const noexcept {
        return this->get(index);
    }

    /**
     * Cast the elements of this vector to type ``U``.
     */
    template<typename U>
    KERNEL_FLOAT_INLINE vector<cast_type<U, Storage>> cast() const noexcept {
        return ::kernel_float::cast<U>(storage());
    }

    /**
     * Apply the given function to the elements of this vector.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE vector<map_type<F, Storage>> map(F fun) const noexcept {
        return ::kernel_float::map(fun, storage());
    }
};

namespace detail {
template<typename Storage>
struct is_vector_helper<vector<Storage>> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename Storage>
struct vector_traits<vector<Storage>>: vector_traits<Storage> {};

using float32 = float;
using float64 = double;

template<typename T, size_t N>
using vec = vector<vector_storage<T, N>>;
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

template<typename T, size_t N>
using unaligned_vec = vector<vector_array<T, N>>;

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)             \
    template<size_t N>                               \
    using NAME##X = vec<T, N>;                       \
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

template<typename... Ts>
KERNEL_FLOAT_INLINE vec<common_t<Ts...>, sizeof...(Ts)> make_vec(Ts... items) {
    return vector_storage<common_t<Ts...>, sizeof...(Ts)> {items...};
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_INTERFACE_H
