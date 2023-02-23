#ifndef KERNEL_FLOAT_H
#define KERNEL_FLOAT_H

#include <type_traits>
#include <utility>

#ifdef __CUDACC__
    #define KERNEL_FLOAT_CUDA (1)

    #ifdef __CUDA_ARCH__
        #define KERNEL_FLOAT_INLINE      __forceinline__ __device__
        #define KERNEL_FLOAT_CUDA_DEVICE (1)
        #define KERNEL_FLOAT_CUDA_HOST   (0)
        #define KERNEL_FLOAT_CUDA_ARCH   (__CUDA_ARCH__)
    #else
        #define KERNEL_FLOAT_INLINE      __forceinline__ __host__
        #define KERNEL_FLOAT_CUDA_DEVICE (0)
        #define KERNEL_FLOAT_CUDA_HOST   (1)
        #define KERNEL_FLOAT_CUDA_ARCH   (0)
    #endif
#else
    #define KERNEL_FLOAT_INLINE      inline
    #define KERNEL_FLOAT_CUDA        (0)
    #define KERNEL_FLOAT_CUDA_HOST   (1)
    #define KERNEL_FLOAT_CUDA_DEVICE (0)
    #define KERNEL_FLOAT_CUDA_ARCH   (0)
#endif

#ifndef KERNEL_FLOAT_FP16
    #define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_BF16
    #define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif

namespace kernel_float {

template<typename T, size_t N = 1>
struct vec;

template<typename F, typename... Args>
using result_t = typename std::result_of<F(Args...)>::type;

template<bool C, typename T = void>
using enabled_t = typename std::enable_if<C, T>::type;

template<typename T, typename... Args>
static constexpr bool is_constructible = std::is_constructible<T, Args...>::value;

template<size_t I>
struct constant_index {
    using value_type = size_t;
    static constexpr size_t value = I;

    KERNEL_FLOAT_INLINE constexpr operator std::integral_constant<size_t, I>() const noexcept {
        return {};
    }

    KERNEL_FLOAT_INLINE constexpr operator size_t() const noexcept {
        return I;
    }

    KERNEL_FLOAT_INLINE constexpr size_t operator()() const noexcept {
        return I;
    }
};

template<size_t... Is>
using index_sequence = std::integer_sequence<size_t, Is...>;

template<size_t N>
using make_index_sequence = std::make_index_sequence<N>;

using I0 = constant_index<0>;
using I1 = constant_index<1>;
using I2 = constant_index<2>;
using I3 = constant_index<3>;

namespace detail {

template<typename... Ts>
struct common_type;

template<typename T>
struct common_type<T> {
    using type = T;
};

template<typename T>
struct common_type<T, T> {
    using type = T;
};

template<typename T, typename U, size_t N>
struct common_type<vec<T, N>, vec<U, N>> {
    using type = vec<typename common_type<T, U>::type, N>;
};

#define KERNEL_FLOAT_DEFINE_COMMON_TYPE(T, U) \
    template<>                                \
    struct common_type<T, U> {                \
        using type = T;                       \
    };                                        \
    template<>                                \
    struct common_type<U, T> {                \
        using type = T;                       \
    };

KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, double)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, float)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, float)
//KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, half)
//KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, half)

#define KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(T, U)       \
    KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed T, signed U) \
    KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned T, unsigned U)

KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, long)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, int)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long long, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, int)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(long, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(int, short)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(int, char)
KERNEL_FLOAT_DEFINE_COMMON_INTEGRAL(short, char)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed long long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed int, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed short, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(signed char, bool)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned long long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned long, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned int, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned short, bool)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(unsigned char, bool)

template<typename T, typename U, typename First, typename... Rest>
struct common_type<T, U, First, Rest...> {
    using type = typename common_type<typename common_type<T, U>::type, First, Rest...>::type;
};
}  // namespace detail

template<typename... Args>
using common_t = typename detail::common_type<Args...>::type;

template<typename From, typename To>
static constexpr bool is_implicit_convertible = std::is_same<common_t<From, To>, To>::value;

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
}  // namespace detail

#define KERNEL_FLOAT_STORAGE_ACCESSORS(T, N)                                                    \
    template<size_t I>                                                                          \
    KERNEL_FLOAT_INLINE T get(constant_index<I>) const {                                        \
        return this->get(size_t(I));                                                            \
    }                                                                                           \
    template<size_t I>                                                                          \
    KERNEL_FLOAT_INLINE void set(constant_index<I>, T value) {                                  \
        this->set(size_t(I), value);                                                            \
    }                                                                                           \
    template<size_t... Is>                                                                      \
    KERNEL_FLOAT_INLINE vec_storage<T, sizeof...(Is)> get(index_sequence<Is...>) const {        \
        return {this->get(constant_index<Is> {})...};                                           \
    }                                                                                           \
    template<size_t... Is>                                                                      \
    KERNEL_FLOAT_INLINE void set(index_sequence<Is...>, vec_storage<T, sizeof...(Is)> values) { \
        detail::assign_helper<index_sequence<Is...>, make_index_sequence<sizeof...(Is)>>::call( \
            *this,                                                                              \
            values);                                                                            \
    }

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

#define KERNEL_FLOAT_DEFINE_SELECT(NAME, ...)                                                     \
    KERNEL_FLOAT_INLINE const vec<T, index_sequence<__VA_ARGS__>::size()> NAME() const noexcept { \
        return ((const Impl*)this)->get(index_sequence<__VA_ARGS__> {});                          \
    }

#define KERNEL_FLOAT_DEFINE_GETTER(NAME, INDEX)                       \
    KERNEL_FLOAT_INLINE T& NAME() noexcept {                          \
        return ((Impl*)this)->get(constant_index<INDEX> {});          \
    }                                                                 \
    KERNEL_FLOAT_INLINE const T& NAME() const noexcept {              \
        return ((const Impl*)this)->get(constant_index<INDEX> {});    \
    }                                                                 \
    KERNEL_FLOAT_INLINE T& _##INDEX() noexcept {                      \
        return ((Impl*)this)->get(constant_index<INDEX> {});          \
    }                                                                 \
    KERNEL_FLOAT_INLINE const T& _##INDEX() const noexcept {          \
        return ((const Impl*)this)->get(constant_index<INDEX> {});    \
    }                                                                 \
    KERNEL_FLOAT_DEFINE_SELECT(NAME##NAME, INDEX, INDEX)              \
    KERNEL_FLOAT_DEFINE_SELECT(NAME##NAME##NAME, INDEX, INDEX, INDEX) \
    KERNEL_FLOAT_DEFINE_SELECT(NAME##NAME##NAME##NAME, INDEX, INDEX, INDEX, INDEX)

template<typename T, size_t N, typename Impl>
struct vec_swizzle: vec_swizzle<T, N - 1, Impl> {};

template<typename T, typename Impl>
struct vec_swizzle<T, 0, Impl> {};

template<typename T, typename Impl>
struct vec_swizzle<T, 1, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(x, 0);
};

template<typename T, typename Impl>
struct vec_swizzle<T, 2, Impl>: public vec_swizzle<T, 1, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(y, 1);
    KERNEL_FLOAT_DEFINE_SELECT(xy, 0, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yx, 1, 0)
};

template<typename T, typename Impl>
struct vec_swizzle<T, 3, Impl>: public vec_swizzle<T, 2, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(z, 2);
    KERNEL_FLOAT_DEFINE_SELECT(xyz, 0, 1, 2)
    KERNEL_FLOAT_DEFINE_SELECT(xzy, 0, 2, 1)
    KERNEL_FLOAT_DEFINE_SELECT(yxz, 1, 0, 2)
    KERNEL_FLOAT_DEFINE_SELECT(yzx, 1, 2, 0)
    KERNEL_FLOAT_DEFINE_SELECT(zxy, 2, 0, 1)
    KERNEL_FLOAT_DEFINE_SELECT(zyx, 2, 1, 0)
};

template<typename T, typename Impl>
struct vec_swizzle<T, 4, Impl>: public vec_swizzle<T, 3, Impl> {
    KERNEL_FLOAT_DEFINE_GETTER(w, 3);
};

template<typename F, typename T, size_t N>
struct map_helper {
    using return_type = result_t<F, T>;

    KERNEL_FLOAT_INLINE static vec<return_type, N> call(F fun, const vec<T, N>& input) noexcept {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static vec<return_type, N>
    call(F fun, const vec<T, N>& input, index_sequence<Is...>) noexcept {
        return vec_storage<return_type, N> {fun(input.get(constant_index<Is> {}))...};
    }
};

template<typename F, typename T>
struct map_helper<F, T, 4> {
    using return_type = result_t<F, T>;

    KERNEL_FLOAT_INLINE static vec<return_type, 4> call(F fun, const vec<T, 4>& input) noexcept {
        return vec_storage<return_type, 4> {
            map_helper<F, T, 2>::call(fun, input.get(index_sequence<0, 1> {})),
            map_helper<F, T, 2>::call(fun, input.get(index_sequence<2, 3> {}))};
    }
};

template<typename F, typename T>
struct map_helper<F, T, 6> {
    using return_type = result_t<F, T>;

    KERNEL_FLOAT_INLINE static vec<return_type, 6> call(F fun, const vec<T, 6>& input) noexcept {
        return vec_storage<return_type, 6> {
            map_helper<F, T, 4>::call(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map_helper<F, T, 2>::call(fun, input.get(index_sequence<4, 5> {}))};
    }
};

template<typename F, typename T>
struct map_helper<F, T, 8> {
    using return_type = result_t<F, T>;

    KERNEL_FLOAT_INLINE static vec<return_type, 8> call(F fun, const vec<T, 8>& input) noexcept {
        return vec_storage<return_type, 8> {
            map_helper<F, T, 4>::call(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map_helper<F, T, 4>::call(fun, input.get(index_sequence<4, 5, 6, 7> {}))};
    }
};

template<typename T, size_t N, typename F, typename R = result_t<F, T>>
KERNEL_FLOAT_INLINE vec<R, N> map(const vec<T, N>& input, F fun) noexcept {
    return map_helper<F, T, N>::call(fun, input);
}

namespace ops {
template<typename T, typename R>
struct cast {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        return R(input);
    }
};

template<typename T>
struct cast<T, T> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept {
        return input;
    }
};
}  // namespace ops

template<typename R, typename T, size_t N>
KERNEL_FLOAT_INLINE vec<R, N> cast(const vec<T, N>& input) noexcept {
    return map(input, ops::cast<T, R> {});
}

#define KERNEL_FLOAT_DEFINE_FUN1_OP(NAME, EXPR)                  \
    namespace ops {                                              \
    template<typename T>                                         \
    struct NAME {                                                \
        KERNEL_FLOAT_INLINE T operator()(T input) {              \
            return EXPR;                                         \
        }                                                        \
    };                                                           \
    }                                                            \
    template<typename T, size_t N>                               \
    KERNEL_FLOAT_INLINE vec<T, N> NAME(const vec<T, N>& input) { \
        return map(input, ops::NAME<T> {});                      \
    }

KERNEL_FLOAT_DEFINE_FUN1_OP(negate, -input)
KERNEL_FLOAT_DEFINE_FUN1_OP(bit_not, ~input)
KERNEL_FLOAT_DEFINE_FUN1_OP(logical_not, !input)

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator-(const vec<T, N>& input) {
    return map(input, ops::negate<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator~(const vec<T, N>& input) {
    return map(input, ops::bit_not<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator!(const vec<T, N>& input) {
    return map(input, ops::logical_not<T> {});
}

#define KERNEL_FLOAT_DEFINE_FUN1(NAME) KERNEL_FLOAT_DEFINE_FUN1_OP(NAME, ::NAME(input))

KERNEL_FLOAT_DEFINE_FUN1(acos)
KERNEL_FLOAT_DEFINE_FUN1(abs)
KERNEL_FLOAT_DEFINE_FUN1(acosh)
KERNEL_FLOAT_DEFINE_FUN1(asin)
KERNEL_FLOAT_DEFINE_FUN1(asinh)
KERNEL_FLOAT_DEFINE_FUN1(atan)
KERNEL_FLOAT_DEFINE_FUN1(atanh)
KERNEL_FLOAT_DEFINE_FUN1(cbrt)
KERNEL_FLOAT_DEFINE_FUN1(ceil)
KERNEL_FLOAT_DEFINE_FUN1(cos)
KERNEL_FLOAT_DEFINE_FUN1(cosh)
KERNEL_FLOAT_DEFINE_FUN1(cospi)
KERNEL_FLOAT_DEFINE_FUN1(erf)
KERNEL_FLOAT_DEFINE_FUN1(erfc)
KERNEL_FLOAT_DEFINE_FUN1(erfcinv)
KERNEL_FLOAT_DEFINE_FUN1(erfcx)
KERNEL_FLOAT_DEFINE_FUN1(erfinv)
KERNEL_FLOAT_DEFINE_FUN1(exp)
KERNEL_FLOAT_DEFINE_FUN1(exp10)
KERNEL_FLOAT_DEFINE_FUN1(exp2)
KERNEL_FLOAT_DEFINE_FUN1(expm1)
KERNEL_FLOAT_DEFINE_FUN1(fabs)
KERNEL_FLOAT_DEFINE_FUN1(floor)
KERNEL_FLOAT_DEFINE_FUN1(ilogb)
KERNEL_FLOAT_DEFINE_FUN1(lgamma)
KERNEL_FLOAT_DEFINE_FUN1(log)
KERNEL_FLOAT_DEFINE_FUN1(log10)
KERNEL_FLOAT_DEFINE_FUN1(logb)
KERNEL_FLOAT_DEFINE_FUN1(nearbyint)
KERNEL_FLOAT_DEFINE_FUN1(normcdf)
KERNEL_FLOAT_DEFINE_FUN1(rcbrt)
KERNEL_FLOAT_DEFINE_FUN1(sin)
KERNEL_FLOAT_DEFINE_FUN1(sinh)
KERNEL_FLOAT_DEFINE_FUN1(sqrt)
KERNEL_FLOAT_DEFINE_FUN1(tan)
KERNEL_FLOAT_DEFINE_FUN1(tanh)
KERNEL_FLOAT_DEFINE_FUN1(tgamma)
KERNEL_FLOAT_DEFINE_FUN1(trunc)
KERNEL_FLOAT_DEFINE_FUN1(y0)
KERNEL_FLOAT_DEFINE_FUN1(y1)
KERNEL_FLOAT_DEFINE_FUN1(yn)
KERNEL_FLOAT_DEFINE_FUN1(rint)
KERNEL_FLOAT_DEFINE_FUN1(rsqrt)
KERNEL_FLOAT_DEFINE_FUN1(round)
KERNEL_FLOAT_DEFINE_FUN1(signbit)

template<typename F, typename T, typename U, size_t N>
struct zip_helper {
    using return_type = result_t<F, T, U>;
    KERNEL_FLOAT_INLINE
    static vec<return_type, N> call(F fun, const vec<T, N>& lhs, const vec<U, N>& rhs) {
        return call(fun, lhs, rhs, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static vec<return_type, N>
    call(F fun, const vec<T, N>& lhs, const vec<U, N>& rhs, index_sequence<Is...>) {
        return vec_storage<return_type, N> {
            fun(lhs.get(constant_index<Is> {}), rhs.get(constant_index<Is> {}))...};
    }
};

template<typename F, typename T, typename U>
struct zip_helper<F, T, U, 4> {
    using return_type = result_t<F, T, U>;

    KERNEL_FLOAT_INLINE static vec<return_type, 4>
    call(F fun, const vec<T, 4>& lhs, const vec<U, 4>& rhs) {
        return vec_storage<return_type, 4> {
            zip_helper<F, T, U, 2>::call(
                fun,
                lhs.get(index_sequence<0, 1> {}),
                rhs.get(index_sequence<0, 1> {})),
            zip_helper<F, T, U, 2>::call(
                fun,
                lhs.get(index_sequence<2, 3> {}),
                rhs.get(index_sequence<2, 3> {}))};
    }
};

template<typename F, typename T, typename U>
struct zip_helper<F, T, U, 6> {
    using return_type = result_t<F, T, U>;

    KERNEL_FLOAT_INLINE static vec<return_type, 6>
    call(F fun, const vec<T, 6>& lhs, const vec<U, 6>& rhs) {
        return vec_storage<return_type, 6> {
            zip_helper<F, T, U, 4>::call(
                fun,
                lhs.get(index_sequence<0, 1, 2, 3> {}),
                rhs.get(index_sequence<0, 1, 2, 3> {})),
            zip_helper<F, T, U, 2>::call(
                fun,
                lhs.get(index_sequence<2, 3> {}),
                rhs.get(index_sequence<2, 3> {}))};
    }
};

template<typename F, typename T, typename U>
struct zip_helper<F, T, U, 8> {
    using return_type = result_t<F, T, U>;

    KERNEL_FLOAT_INLINE static vec<return_type, 8>
    call(F fun, const vec<T, 8>& lhs, const vec<U, 8>& rhs) {
        return vec_storage<return_type, 8> {
            zip_helper<F, T, U, 4>::call(
                fun,
                lhs.get(index_sequence<0, 1, 2, 3> {}),
                rhs.get(index_sequence<0, 1, 2, 3> {})),
            zip_helper<F, T, U, 4>::call(
                fun,
                lhs.get(index_sequence<4, 5, 6, 7> {}),
                rhs.get(index_sequence<4, 5, 6, 7> {}))};
    }
};

template<typename T, typename U, size_t N, typename F, typename R = result_t<F, T, U>>
KERNEL_FLOAT_INLINE vec<R, N> zip(F fun, const vec<T, N>& lhs, const vec<U, N>& rhs) {
    return zip_helper<F, T, U, N>::call(fun, lhs, rhs);
}

template<
    typename T,
    typename U,
    size_t N,
    typename F,
    typename C = common_t<T, U>,
    typename R = result_t<F, C, C>>
KERNEL_FLOAT_INLINE vec<R, N> zip_common(F fun, const vec<T, N>& lhs, const vec<U, N>& rhs) {
    return zip(fun, cast<C>(lhs), cast<C>(rhs));
}

#define KERNEL_FLOAT_DEFINE_FUN2_OP(NAME, EXPR)                                      \
    namespace ops {                                                                  \
    template<typename T>                                                             \
    struct NAME {                                                                    \
        KERNEL_FLOAT_INLINE auto operator()(T lhs, T rhs) -> decltype(EXPR) {        \
            return EXPR;                                                             \
        }                                                                            \
    };                                                                               \
    }                                                                                \
    template<                                                                        \
        typename T,                                                                  \
        typename U,                                                                  \
        size_t N,                                                                    \
        typename C = common_t<T, U>,                                                 \
        typename R = result_t<ops::NAME<C>, C, C>>                                   \
    KERNEL_FLOAT_INLINE vec<R, N> NAME(const vec<T, N>& lhs, const vec<U, N>& rhs) { \
        return zip(ops::NAME<C> {}, cast<C>(lhs), cast<C>(rhs));                     \
    }

#define KERNEL_FLOAT_DEFINE_BINOP(NAME, OP)                                                 \
    KERNEL_FLOAT_DEFINE_FUN2_OP(NAME, lhs OP rhs)                                           \
    template<                                                                               \
        typename T,                                                                         \
        typename U,                                                                         \
        size_t N,                                                                           \
        typename C = common_t<T, U>,                                                        \
        typename R = result_t<ops::NAME<C>, C, C>>                                          \
    KERNEL_FLOAT_INLINE vec<R, N> operator OP(const vec<T, N>& lhs, const vec<U, N>& rhs) { \
        return zip(ops::NAME<C> {}, cast<C>(lhs), cast<C>(rhs));                            \
    }                                                                                       \
    template<typename T, size_t N, typename R = result_t<ops::NAME<T>, T, T>>               \
    KERNEL_FLOAT_INLINE vec<R, N> operator OP(const vec<T, N>& lhs, const T& rhs) {         \
        return zip(ops::NAME<T> {}, lhs, vec<T, N> {rhs});                                  \
    }                                                                                       \
    template<typename T, size_t N, typename R = result_t<ops::NAME<T>, T, T>>               \
    KERNEL_FLOAT_INLINE vec<R, N> operator OP(const T& rhs, const vec<T, N>& lhs) {         \
        return zip(ops::NAME<T> {}, vec<T, N> {lhs}, rhs);                                  \
    }

KERNEL_FLOAT_DEFINE_BINOP(add, +)
KERNEL_FLOAT_DEFINE_BINOP(subtract, -)
KERNEL_FLOAT_DEFINE_BINOP(mulitply, *)
KERNEL_FLOAT_DEFINE_BINOP(divide, /)
KERNEL_FLOAT_DEFINE_BINOP(modulus, %)

KERNEL_FLOAT_DEFINE_BINOP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINOP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINOP(less, <)
KERNEL_FLOAT_DEFINE_BINOP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINOP(greater, >)
KERNEL_FLOAT_DEFINE_BINOP(greater_equal, >=)

KERNEL_FLOAT_DEFINE_BINOP(bit_and, &)
KERNEL_FLOAT_DEFINE_BINOP(bit_or, |)
KERNEL_FLOAT_DEFINE_BINOP(bit_xor, ^)

#define KERNEL_FLOAT_DEFINE_FUN2(NANE) KERNEL_FLOAT_DEFINE_FUN2_OP(NANE, ::NANE(lhs, rhs))

KERNEL_FLOAT_DEFINE_FUN2(min)
KERNEL_FLOAT_DEFINE_FUN2(max)
KERNEL_FLOAT_DEFINE_FUN2(copysign)
KERNEL_FLOAT_DEFINE_FUN2(hypot)
KERNEL_FLOAT_DEFINE_FUN2(modf)
KERNEL_FLOAT_DEFINE_FUN2(nextafter)
KERNEL_FLOAT_DEFINE_FUN2(pow)
KERNEL_FLOAT_DEFINE_FUN2(remainder)

#if KERNEL_FLOAT_CUDA_DEVICE
KERNEL_FLOAT_DEFINE_FUN2(rhypot)
#endif

template<typename F, size_t N>
struct range_helper {
    using return_type = result_t<F, size_t>;
    KERNEL_FLOAT_INLINE
    static vec<return_type, N> call(F fun) {
        return call(fun, make_index_sequence<N> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static vec<return_type, N> call(F fun, index_sequence<Is...>) {
        return vec_storage<return_type, N>(fun(constant_index<Is> {})...);
    }
};

template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vec<T, N> range(F fun) {
    return range_helper<F, N>::call(fun);
}

template<size_t N, typename T = size_t>
KERNEL_FLOAT_INLINE vec<T, N> range() {
    return range<N>(ops::cast<size_t, T> {});
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

template<typename F, typename T, size_t N>
struct iterate_helper {
    KERNEL_FLOAT_INLINE
    static void call(F fun, vec<T, N>& input) {
        call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t First, size_t... Rest>
    KERNEL_FLOAT_INLINE static void
    call(F fun, vec<T, N>& input, index_sequence<First, Rest...> = make_index_sequence<N> {}) {
        fun(input.get(constant_index<First> {}));
        call(fun, input, index_sequence<Rest...> {});
    }
    KERNEL_FLOAT_INLINE
    static void call(F fun, vec<T, N>& input, index_sequence<>) {}
};

template<typename F, typename T, size_t N>
struct iterate_helper<F, const T, N> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const vec<T, N>& input) {
        call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t First, size_t... Rest>
    KERNEL_FLOAT_INLINE static void call(
        F fun,
        const vec<T, N>& input,
        index_sequence<First, Rest...> = make_index_sequence<N> {}) {
        fun(input.get(constant_index<First> {}));
        call(fun, input, index_sequence<Rest...> {});
    }

    static void call(F fun, const vec<T, N>& input, index_sequence<>) {}
};

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE void for_each(const vec<T, N>& input, F fun) {
    return iterate_helper<F, const T, N>::call(fun, input);
}

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE void for_each(vec<T, N>& input, F fun) {
    return iterate_helper<F, T, N>::call(fun, input);
}

template<typename F, typename T, size_t N>
struct reduce_helper {
    KERNEL_FLOAT_INLINE
    static T call(F fun, const vec<T, N>& input) {
        return call(fun, input, make_index_sequence<N> {});
    }

  private:
    template<size_t... Rest>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, N>& input, index_sequence<0, Rest...>) {
        return call(input.get(constant_index<0> {}), fun, input, index_sequence<Rest...> {});
    }

    template<size_t First, size_t... Rest>
    KERNEL_FLOAT_INLINE static T
    call(T accum, F fun, const vec<T, N>& input, index_sequence<First, Rest...>) {
        accum = fun(accum, input.get(constant_index<First> {}));
        return call(accum, fun, input, index_sequence<Rest...> {});
    }
    KERNEL_FLOAT_INLINE
    static T call(T accum, F fun, const vec<T, N>& input, index_sequence<>) {
        return accum;
    }
};

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE T reduce(const vec<T, N>& input, F fun) {
    return reduce_helper<F, T, N>::call(fun, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T min(const vec<T, N>& input) {
    return reduce(input, ops::min<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T max(const vec<T, N>& input) {
    return reduce(input, ops::max<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T sum(const vec<T, N>& input) {
    return reduce(input, ops::add<T> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T product(const vec<T, N>& input) {
    return reduce(input, ops::mulitply<T> {});
}

template<size_t N>
KERNEL_FLOAT_INLINE bool all(const vec<bool, N>& input) {
    return reduce(input, ops::bit_and<bool> {});
}

template<size_t N>
KERNEL_FLOAT_INLINE bool any(const vec<bool, N>& input) {
    return reduce(input, ops::bit_or<bool> {});
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
struct vec: public vec_storage<T, N>, public vec_swizzle<T, N, vec<T, N>> {
    using storage_type = vec_storage<T, N>;
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

    KERNEL_FLOAT_INLINE explicit vec(T item) : vec(full<N>(item)) {}

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
};  // namespace kernel_float

namespace kernel_float {

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

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_CUDA_DEVICE
    #include <cuda_fp16.h>

namespace kernel_float {
using half = __half;
using half2 = __half2;

namespace detail {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(long double, half)
}  // namespace detail

template<>
struct vec_storage<half, 2> {
    KERNEL_FLOAT_INLINE vec_storage(half x, half y) noexcept : value_(__halves2half2(x, y)) {}
    KERNEL_FLOAT_INLINE vec_storage(half2 value) noexcept : value_(value) {}
    //    KERNEL_FLOAT_DEFINE_ACCESS(0, __low2half(value_))
    //    KERNEL_FLOAT_DEFINE_ACCESS(1, __high2half(value_))

    operator half2() const {
        return value_;
    }

  private:
    half2 value_;
};

    #define KERNEL_FLOAT_FP16_MONOP(NAME, FUN1, FUN2)                                             \
        namespace ops {                                                                           \
        template<>                                                                                \
        struct NAME<half> {                                                                       \
            KERNEL_FLOAT_INLINE half operator()(half input) {                                     \
                return FUN1(input);                                                               \
            }                                                                                     \
        };                                                                                        \
        }                                                                                         \
        template<>                                                                                \
        struct map_helper<ops::NAME<half>, half, 2> {                                             \
            KERNEL_FLOAT_INLINE static vec<half, 2> call(ops::NAME<half>, half2 input) noexcept { \
                return FUN2(input);                                                               \
            }                                                                                     \
        };

KERNEL_FLOAT_FP16_MONOP(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_FP16_MONOP(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_FP16_MONOP(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_FP16_MONOP(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_FP16_MONOP(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_FP16_MONOP(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_FP16_MONOP(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_FP16_MONOP(log, ::hlog, ::h2log);
KERNEL_FLOAT_FP16_MONOP(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_FP16_MONOP(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_FP16_MONOP(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_FP16_MONOP(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_FP16_MONOP(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_FP16_MONOP(trunc, ::htrunc, ::h2trunc);
    //    KERNEL_FLOAT_FP16_MONOP(rcp, hrcp, h2rcp);

    #define KERNEL_FLOAT_FP16_BINOP(NAME, FUN1, FUN2)                                      \
        namespace ops {                                                                    \
        template<>                                                                         \
        struct NAME<half> {                                                                \
            KERNEL_FLOAT_INLINE half operator()(half lhs, half rhs) {                      \
                return FUN1(lhs, rhs);                                                     \
            }                                                                              \
        };                                                                                 \
        }                                                                                  \
        template<>                                                                         \
        struct zip_helper<ops::NAME<half>, half, half, 2> {                                \
            KERNEL_FLOAT_INLINE static half2 call(ops::NAME<half>, half2 lhs, half2 rhs) { \
                return FUN2(lhs, rhs);                                                     \
            }                                                                              \
        };

KERNEL_FLOAT_FP16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_FP16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_FP16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_FP16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_FP16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_FP16_BINOP(max, __hmax, __hmax2);

    #define KERNEL_FLOAT_FP16_RELOP(NAME, FUN1, FUN2)                 \
        namespace ops {                                               \
        template<>                                                    \
        struct NAME<half> {                                           \
            KERNEL_FLOAT_INLINE bool operator()(half lhs, half rhs) { \
                return FUN1(lhs, rhs);                                \
            }                                                         \
        };                                                            \
        }                                                             \
        //        template<>                                                                                \
//        struct zip_helper<ops::NAME<half>, half, half, 2> {                                       \
//            KERNEL_FLOAT_INLINE static vec<bool, 2> call(ops::NAME<half>, half2 lhs, half2 rhs) { \
//                return FUN2(lhs, rhs);                                                            \
//            }                                                                                     \
//        };

KERNEL_FLOAT_FP16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_FP16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_FP16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_FP16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_FP16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_FP16_RELOP(less_equal, __hle, __hle2);

    #define KERNEL_FLOAT_FP16_CAST(FROM, TO, FUN1)          \
        namespace ops {                                     \
        template<>                                          \
        struct cast<FROM, TO> {                             \
            KERNEL_FLOAT_INLINE TO operator()(FROM input) { \
                return FUN1;                                \
            }                                               \
        };                                                  \
        }

KERNEL_FLOAT_FP16_CAST(half, float, __half2float(input));
KERNEL_FLOAT_FP16_CAST(float, half, __float2half(input));

KERNEL_FLOAT_FP16_CAST(half, double, double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(double, half, __double2half(input));

KERNEL_FLOAT_FP16_CAST(half, unsigned long, __half2ull_rd(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, half, __ull2half_rd(input));

//__half22float2 ( const __half2 a )
//__half2float ( const __half a )

}  // namespace kernel_float
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
    #include <cuda_bf16.h>

namespace kernel_float {}
#endif

#endif  //KERNEL_FLOAT_KERNEL_FLOAT_H
