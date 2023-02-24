#ifndef KERNEL_FLOAT_ALL_H
#define KERNEL_FLOAT_ALL_H

#include <type_traits>
#include <utility>

#include "binops.h"
#include "core.h"
#include "iterate.h"
#include "macros.h"
#include "reduce.h"
#include "storage.h"
#include "unops.h"

namespace kernel_float {

template<size_t N, typename T>
KERNEL_FLOAT_INLINE vec<T, N> full(T item);

namespace detail {
#define KERNEL_FLOAT_DEFINE_SELECT(NAME, ...)                                                     \
    KERNEL_FLOAT_INLINE const vec<T, index_sequence<__VA_ARGS__>::size()> NAME() const noexcept { \
        return ((const Impl*)this)->get(index_sequence<__VA_ARGS__> {});                          \
    }

#define KERNEL_FLOAT_DEFINE_GETTER(NAME, INDEX)                    \
    KERNEL_FLOAT_INLINE T& NAME() noexcept {                       \
        return ((Impl*)this)->get(constant_index<INDEX> {});       \
    }                                                              \
    KERNEL_FLOAT_INLINE const T& NAME() const noexcept {           \
        return ((const Impl*)this)->get(constant_index<INDEX> {}); \
    }                                                              \
    KERNEL_FLOAT_INLINE T& _##INDEX() noexcept {                   \
        return ((Impl*)this)->get(constant_index<INDEX> {});       \
    }                                                              \
    KERNEL_FLOAT_INLINE const T& _##INDEX() const noexcept {       \
        return ((const Impl*)this)->get(constant_index<INDEX> {}); \
    }

template<typename T, size_t N, typename Impl>
struct swizzler: swizzler<T, N - 1, Impl> {};

template<typename T, typename Impl>
struct swizzler<T, 0, Impl> {};

template<typename T, typename Impl>
struct swizzler<T, 1, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(x, 0);
    KERNEL_FLOAT_DEFINE_SELECT(xx, 0, 0)
    KERNEL_FLOAT_DEFINE_SELECT(xxx, 0, 0, 0)
    KERNEL_FLOAT_DEFINE_SELECT(xxxx, 0, 0, 0, 0)
};

template<typename T, typename Impl>
struct swizzler<T, 2, Impl>: public swizzler<T, 1, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(y, 1);
    KERNEL_FLOAT_DEFINE_SELECT(yy, 1, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yyy, 1, 1, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yyyy, 1, 1, 1, 1)
    KERNEL_FLOAT_DEFINE_SELECT(xy, 0, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yx, 1, 0)
};

template<typename T, typename Impl>
struct swizzler<T, 3, Impl>: public swizzler<T, 2, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(z, 2);
    KERNEL_FLOAT_DEFINE_SELECT(zz, 2, 2)
    KERNEL_FLOAT_DEFINE_SELECT(zzz, 2, 2, 2)
    KERNEL_FLOAT_DEFINE_SELECT(zzzz, 2, 2, 2, 2)
    KERNEL_FLOAT_DEFINE_SELECT(xyz, 0, 1, 2)
    KERNEL_FLOAT_DEFINE_SELECT(xzy, 0, 2, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yxz, 1, 0, 2)
    KERNEL_FLOAT_DEFINE_SELECT(yzx, 1, 2, 0)
    KERNEL_FLOAT_DEFINE_SELECT(zxy, 2, 0, 1)
    KERNEL_FLOAT_DEFINE_SELECT(zyx, 2, 1, 0)
};

template<typename T, typename Impl>
struct swizzler<T, 4, Impl>: public swizzler<T, 3, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(w, 3);
    KERNEL_FLOAT_DEFINE_SELECT(ww, 3, 3)
    KERNEL_FLOAT_DEFINE_SELECT(www, 3, 3, 3)
    KERNEL_FLOAT_DEFINE_SELECT(wwww, 3, 3, 3, 3)
};
}  // namespace detail

template<typename T, size_t N, typename I>
struct index_proxy {
    KERNEL_FLOAT_INLINE
    index_proxy(vec<T, N>& inner, I index) noexcept : inner_(inner), index_(index) {}

    KERNEL_FLOAT_INLINE
    operator T() noexcept {
        return inner_.get(index_);
    }

    KERNEL_FLOAT_INLINE
    index_proxy& operator=(T value) noexcept {
        inner_.set(index_, value);
        return *this;
    }

  private:
    vec<T, N>& inner_;
    I index_;
};

template<typename T, size_t N>
struct vec: public detail::vec_storage<T, N>, public detail::swizzler<T, N, vec<T, N>> {
    using storage_type = detail::vec_storage<T, N>;
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    vec(const vec&) = default;
    vec(vec&) = default;
    vec(vec&&) noexcept = default;

    vec& operator=(const vec&) = default;
    vec& operator=(vec&) = default;
    vec& operator=(vec&&) noexcept = default;

    KERNEL_FLOAT_INLINE vec(T item) : vec(full<N>(item)) {}

    KERNEL_FLOAT_INLINE vec() : vec(T {}) {}

    KERNEL_FLOAT_INLINE vec(storage_type storage) : storage_type {storage} {}

    template<typename U, typename = enabled_t<is_implicit_convertible<U, T>>>
    KERNEL_FLOAT_INLINE vec(const vec<U, N>& that) : vec(::kernel_float::cast<T>(that)) {}

    template<typename... Args, typename = enabled_t<is_constructible<storage_type, Args...>>>
    KERNEL_FLOAT_INLINE vec(Args&&... args) : storage_type {std::forward<Args>(args)...} {}

    KERNEL_FLOAT_INLINE
    size_t size() const noexcept {
        return N;
    }

    template<typename I>
    KERNEL_FLOAT_INLINE index_proxy<T, N, I> operator[](I index) noexcept {
        return {*this, index};
    }

    template<typename I>
    KERNEL_FLOAT_INLINE T operator[](I index) const noexcept {
        return this->get(index);
    }

    template<typename U>
    KERNEL_FLOAT_INLINE vec<U, N> cast() const noexcept {
        return ::kernel_float::cast<U>(*this);
    }

    template<typename F, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE vec<R, N> map(F fun) const noexcept {
        return ::kernel_float::map(*this, fun);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) noexcept {
        return ::kernel_float::for_each(*this, fun);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const noexcept {
        return ::kernel_float::for_each(*this, fun);
    }

    template<typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun) noexcept {
        return ::kernel_float::reduce(*this, fun);
    }

    template<size_t... Is>
    KERNEL_FLOAT_INLINE vec<T, sizeof...(Is)> select(index_sequence<Is...>) noexcept {
        return {this->get(constant_index<Is> {})...};
    }
};

template<typename... Ts>
KERNEL_FLOAT_INLINE vec<common_t<Ts...>, sizeof...(Ts)> make_vec(Ts&&... args) noexcept {
    return {std::forward<Ts>(args)...};
}

template<size_t N, typename T>
KERNEL_FLOAT_INLINE vec<T, N> read(const T* ptr, size_t stride = 1) {
    return range<N>([&](auto i) { return ptr[i * stride]; });
}

template<size_t N, typename T>
KERNEL_FLOAT_INLINE void write(const vec<T, N>& data, const T* ptr, size_t stride = 1) {
    range<N>([&](auto i) {
        ptr[i * stride] = data.get(i);
        return 0;
    });
}

namespace ops {
template<typename T>
struct constant {
    KERNEL_FLOAT_INLINE constant(T item) : item_(item) {}

    template<typename... Args>
    KERNEL_FLOAT_INLINE T operator()(Args&&... args) const {
        return item_;
    }

  private:
    T item_;
};
};  // namespace ops

template<size_t N, typename T>
KERNEL_FLOAT_INLINE vec<T, N> full(T item) {
    return range<N>(ops::constant<T>(item));
}

};  // namespace kernel_float

#endif  //KERNEL_FLOAT_KERNEL_FLOAT_H
