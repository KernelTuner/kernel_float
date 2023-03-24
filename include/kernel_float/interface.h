#ifndef KERNEL_FLOAT_INTERFACE_H
#define KERNEL_FLOAT_INTERFACE_H

#include "storage.h"

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
