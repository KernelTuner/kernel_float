#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

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

#endif  //KERNEL_FLOAT_MACROS_H
#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H



namespace kernel_float {
template<typename T, size_t N = 1>
struct vec;

template<typename F, typename... Args>
using result_t = typename std::result_of<F(Args...)>::type;

template<bool C, typename T = void>
using enabled_t = typename std::enable_if<C, T>::type;

using float32 = float;
using float64 = double;

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

template<typename... Args>
using common_t = typename common_type<Args...>::type;

template<typename From, typename To>
static constexpr bool is_implicit_convertible = std::is_same<common_t<From, To>, To>::value;
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_CORE_H
#ifndef KERNEL_FLOAT_STORAGE_H
#define KERNEL_FLOAT_STORAGE_H



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
#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H



namespace kernel_float {

namespace detail {
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
}  // namespace detail

template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vec<T, N> range(F fun) {
    return detail::range_helper<F, N>::call(fun);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> range() {
    return range<N>(ops::cast<size_t, T> {});
}

template<size_t N>
KERNEL_FLOAT_INLINE vec<size_t, N> range() {
    return range<size_t, N>();
}

namespace detail {
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
}  // namespace detail

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE void for_each(const vec<T, N>& input, F fun) {
    return detail::iterate_helper<F, const T, N>::call(fun, input);
}

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE void for_each(vec<T, N>& input, F fun) {
    return detail::iterate_helper<F, T, N>::call(fun, input);
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H



namespace kernel_float {

template<size_t N>
struct reduce_apply_helper;

template<typename F, typename T, size_t N>
struct reduce_helper {
    KERNEL_FLOAT_INLINE
    static T call(F fun, const vec<T, N>& input) {
        return reduce_apply_helper<N>::call(fun, input);
    }
};

template<>
struct reduce_apply_helper<1> {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, 1>& input) noexcept {
        return input.get(constant_index<0> {});
    }
};

template<>
struct reduce_apply_helper<2> {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, 2>& input) noexcept {
        return fun(input.get(constant_index<0> {}), input.get(constant_index<1> {}));
    }
};

template<size_t N>
struct reduce_apply_helper {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, N>& input) noexcept {
        T lhs = reduce_helper<F, T, N - 1>::call(fun, input.get(make_index_sequence<N - 1> {}));
        T rhs = input.get(N - 1);
        return fun(lhs, rhs);
    }
};

template<>
struct reduce_apply_helper<4> {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, 4>& input) noexcept {
        vec<T, 2> lhs = input.get(index_sequence<0, 1> {});
        vec<T, 2> rhs = input.get(index_sequence<2, 3> {});
        return reduce(fun, zip(fun, lhs, rhs));
    }
};

template<>
struct reduce_apply_helper<6> {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, 6>& input) noexcept {
        vec<T, 2> a = input.get(index_sequence<0, 1> {});
        vec<T, 2> b = input.get(index_sequence<2, 3> {});
        vec<T, 2> c = input.get(index_sequence<4, 5> {});
        return reduce(fun, zip(fun, zip(fun, a, b), c));
    }
};

template<>
struct reduce_apply_helper<8> {
    template<typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const vec<T, 8>& input) noexcept {
        vec<T, 4> lhs = input.get(index_sequence<0, 1, 2, 3> {});
        vec<T, 4> rhs = input.get(index_sequence<4, 5, 6, 7> {});
        return reduce(fun, zip(fun, lhs, rhs));
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

template<typename T, size_t N>
KERNEL_FLOAT_INLINE bool all(const vec<T, N>& input) {
    return reduce(cast<bool>(input), ops::bit_and<bool> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE bool any(const vec<T, N>& input) {
    return reduce(cast<bool>(input), ops::bit_or<bool> {});
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
#ifndef KERNEL_FLOAT_UNARY_H
#define KERNEL_FLOAT_UNARY_H




namespace kernel_float {

template<size_t N, typename Is = make_index_sequence<N>>
struct map_apply_helper;

template<typename F, typename T, size_t N>
struct map_helper {
    using return_type = result_t<F, T>;
    KERNEL_FLOAT_INLINE static vec<return_type, N> call(F fun, const vec<T, N>& input) noexcept {
        return map_apply_helper<N>::call(fun, input);
    }
};

template<size_t N, size_t... Is>
struct map_apply_helper<N, index_sequence<Is...>> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, N> call(F fun, const vec<T, N>& input) noexcept {
        return detail::vec_storage<R, N> {fun(input.get(Is))...};
    }
};

template<>
struct map_apply_helper<2> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 2> call(F fun, const vec<T, 2>& input) noexcept {
        return {fun(input.get(constant_index<0> {})), fun(input.get(constant_index<1> {}))};
    }
};

template<>
struct map_apply_helper<4> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 4> call(F fun, const vec<T, 4>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1> {})),
            map(fun, input.get(index_sequence<1, 2> {}))};
    }
};

template<>
struct map_apply_helper<6> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 6> call(F fun, const vec<T, 6>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map(fun, input.get(index_sequence<4, 5> {}))};
    }
};

template<>
struct map_apply_helper<8> {
    template<typename F, typename T, typename R = result_t<F, T>>
    KERNEL_FLOAT_INLINE static vec<R, 8> call(F fun, const vec<T, 8>& input) noexcept {
        return {
            map(fun, input.get(index_sequence<0, 1, 2, 3> {})),
            map(fun, input.get(index_sequence<4, 5, 6, 7> {}))};
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

template<typename T, size_t N>
struct map_helper<ops::cast<T, T>, T, N> {
    KERNEL_FLOAT_INLINE static vec<T, N> call(ops::cast<T, T>, const vec<T, N>& input) noexcept {
        return input;
    }
};

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
KERNEL_FLOAT_DEFINE_FUN1(isinf)
KERNEL_FLOAT_DEFINE_FUN1(isnan)

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNARY_H
#ifndef KERNEL_FLOAT_BINARY_H
#define KERNEL_FLOAT_BINARY_H




namespace kernel_float {

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
        return detail::vec_storage<return_type, N> {
            fun(lhs.get(constant_index<Is> {}), rhs.get(constant_index<Is> {}))...};
    }
};

template<typename F, typename T, typename U>
struct zip_helper<F, T, U, 4> {
    using return_type = result_t<F, T, U>;

    KERNEL_FLOAT_INLINE static vec<return_type, 4>
    call(F fun, const vec<T, 4>& lhs, const vec<U, 4>& rhs) {
        return detail::vec_storage<return_type, 4> {
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
        return detail::vec_storage<return_type, 6> {
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
        return detail::vec_storage<return_type, 8> {
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

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_BINARY_H
#ifndef KERNEL_FLOAT_ALL_H
#define KERNEL_FLOAT_ALL_H

#include <type_traits>
#include <utility>









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

#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>

namespace kernel_float {}
#endif

#endif  //KERNEL_FLOAT_KERNEL_FLOAT_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#include <cuda_bf16.h>



namespace kernel_float {
using bfloat16 = __nv_bfloat16;
using bfloat16x2 = __nv_bfloat162;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, bfloat16)

namespace detail {
template<>
struct vec_storage<bfloat16, 2> {
    static_assert(sizeof(bfloat16) * 2 == sizeof(bfloat16x2), "invalid size");
    static_assert(alignof(bfloat16) <= alignof(bfloat16x2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(bfloat16 x, bfloat16 y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(bfloat16x2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator bfloat16x2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE bfloat16 get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE bfloat16 get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, bfloat16 v) {
        *this = vec_storage(v, __high2float(vector_));
    }

    KERNEL_FLOAT_INLINE void set(I1, bfloat16 v) {
        *this = vec_storage(__low2float(vector_), v);
    }

    KERNEL_FLOAT_INLINE bfloat16 get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(bfloat16, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<0, 0>) const {
        return {vector_.x, vector_.x};
    }

    KERNEL_FLOAT_INLINE vec<bfloat16, 2> get(index_sequence<1, 1>) const {
        return __high2bfloat162(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, bfloat16x2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, bfloat16x2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    bfloat16x2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_BF16_AVAILABLE && KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_BF16_MONOP(NAME, FUN1, FUN2)                 \
    namespace ops {                                               \
    template<>                                                    \
    struct NAME<bfloat16> {                                       \
        KERNEL_FLOAT_INLINE bfloat16 operator()(bfloat16 input) { \
            return FUN1(input);                                   \
        }                                                         \
    };                                                            \
    }                                                             \
    template<>                                                    \
    struct map_helper<ops::NAME<bfloat16>, bfloat16, 2> {         \
        KERNEL_FLOAT_INLINE static vec<bfloat16, 2>               \
        call(ops::NAME<bfloat16>, bfloat16x2 input) noexcept {    \
            return FUN2(input);                                   \
        }                                                         \
    };

KERNEL_FLOAT_BF16_MONOP(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_BF16_MONOP(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_BF16_MONOP(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_BF16_MONOP(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_BF16_MONOP(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_BF16_MONOP(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_BF16_MONOP(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_BF16_MONOP(log, ::hlog, ::h2log);
KERNEL_FLOAT_BF16_MONOP(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_BF16_MONOP(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_BF16_MONOP(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_BF16_MONOP(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_BF16_MONOP(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_BF16_MONOP(trunc, ::htrunc, ::h2trunc);
//    KERNEL_FLOAT_BF16_MONOP(rcp, hrcp, h2rcp);

#define KERNEL_FLOAT_BF16_BINOP(NAME, FUN1, FUN2)                             \
    namespace ops {                                                           \
    template<>                                                                \
    struct NAME<bfloat16> {                                                   \
        KERNEL_FLOAT_INLINE bfloat16 operator()(bfloat16 lhs, bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                            \
        }                                                                     \
    };                                                                        \
    }                                                                         \
    template<>                                                                \
    struct zip_helper<ops::NAME<bfloat16>, bfloat16, bfloat16, 2> {           \
        KERNEL_FLOAT_INLINE static bfloat16x2                                 \
        call(ops::NAME<bfloat16>, bfloat16x2 lhs, bfloat16x2 rhs) {           \
            return FUN2(lhs, rhs);                                            \
        }                                                                     \
    };

KERNEL_FLOAT_BF16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_BF16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_BF16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_BF16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_BF16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_BF16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_BF16_RELOP(NAME, FUN1, FUN2)                         \
    namespace ops {                                                       \
    template<>                                                            \
    struct NAME<bfloat16> {                                               \
        KERNEL_FLOAT_INLINE bool operator()(bfloat16 lhs, bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                        \
        }                                                                 \
    };                                                                    \
    }

KERNEL_FLOAT_BF16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_BF16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_BF16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_BF16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_BF16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_BF16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)      \
    namespace ops {                                        \
    template<>                                             \
    struct cast<T, bfloat16> {                             \
        KERNEL_FLOAT_INLINE bfloat16 operator()(T input) { \
            return TO_HALF;                                \
        }                                                  \
    };                                                     \
    template<>                                             \
    struct cast<bfloat16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(bfloat16 input) { \
            return FROM_HALF;                              \
        }                                                  \
    };                                                     \
    }

KERNEL_FLOAT_BF16_CAST(float64, __double2bfloat16(input), float64(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float32, __float2bfloat16(input), __bfloat162float(input));

KERNEL_FLOAT_BF16_CAST(signed short, __bfloat162short_rz(input), __short2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(signed int, __bfloat162int_rz(input), __int2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    signed long,
    __ll2bfloat16_rn(input),
    (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __bfloat162ushort_rz(input), __ushort2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __bfloat162uint_rz(input), __uint2bfloat16_rn(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned long,
    __ull2bfloat16_rn(input),
    (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));

template<>
struct map_helper<ops::cast<bfloat16, float32>, bfloat16, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<bfloat16, float32>, const vec<bfloat16, 2>& input) noexcept {
        return __bfloat1622float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, bfloat16>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<bfloat16, 2>
    call(ops::cast<float32, bfloat16>, const vec<float32, 2>& input) noexcept {
        return __float22bfloat162_rn(input);
    }
};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_BF16_H
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#include <cuda_fp16.h>



namespace kernel_float {
using float16 = __half;
using float16x2 = __half2;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float16)

namespace detail {
template<>
struct vec_storage<float16, 2> {
    static_assert(sizeof(float16) * 2 == sizeof(float16x2), "invalid size");
    static_assert(alignof(float16) <= alignof(float16x2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(float16 x, float16 y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(float16x2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator float16x2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE float16 get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE float16 get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, float16 v) {
        *this = vec_storage(v, get(I1 {}));
    }

    KERNEL_FLOAT_INLINE void set(I1, float16 v) {
        *this = vec_storage(get(I0 {}), v);
    }

    KERNEL_FLOAT_INLINE float16 get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(float16, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<0, 0>) const {
        return __low2half2(vector_);
    }

    KERNEL_FLOAT_INLINE vec<float16, 2> get(index_sequence<1, 1>) const {
        return __high2half2(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, float16x2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, float16x2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    float16x2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_FP16_MONOP(NAME, FUN1, FUN2)               \
    namespace ops {                                             \
    template<>                                                  \
    struct NAME<float16> {                                      \
        KERNEL_FLOAT_INLINE float16 operator()(float16 input) { \
            return FUN1(input);                                 \
        }                                                       \
    };                                                          \
    }                                                           \
    template<>                                                  \
    struct map_helper<ops::NAME<float16>, float16, 2> {         \
        KERNEL_FLOAT_INLINE static vec<float16, 2>              \
        call(ops::NAME<float16>, float16x2 input) noexcept {    \
            return FUN2(input);                                 \
        }                                                       \
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

#define KERNEL_FLOAT_FP16_BINOP(NAME, FUN1, FUN2)                          \
    namespace ops {                                                        \
    template<>                                                             \
    struct NAME<float16> {                                                 \
        KERNEL_FLOAT_INLINE float16 operator()(float16 lhs, float16 rhs) { \
            return FUN1(lhs, rhs);                                         \
        }                                                                  \
    };                                                                     \
    }                                                                      \
    template<>                                                             \
    struct zip_helper<ops::NAME<float16>, float16, float16, 2> {           \
        KERNEL_FLOAT_INLINE static float16x2                               \
        call(ops::NAME<float16>, float16x2 lhs, float16x2 rhs) {           \
            return FUN2(lhs, rhs);                                         \
        }                                                                  \
    };

KERNEL_FLOAT_FP16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_FP16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_FP16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_FP16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_FP16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_FP16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_FP16_RELOP(NAME, FUN1, FUN2)                       \
    namespace ops {                                                     \
    template<>                                                          \
    struct NAME<float16> {                                              \
        KERNEL_FLOAT_INLINE bool operator()(float16 lhs, float16 rhs) { \
            return FUN1(lhs, rhs);                                      \
        }                                                               \
    };                                                                  \
    }

KERNEL_FLOAT_FP16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_FP16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_FP16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_FP16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_FP16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_FP16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)     \
    namespace ops {                                       \
    template<>                                            \
    struct cast<T, float16> {                             \
        KERNEL_FLOAT_INLINE float16 operator()(T input) { \
            return TO_HALF;                               \
        }                                                 \
    };                                                    \
    template<>                                            \
    struct cast<float16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(float16 input) { \
            return FROM_HALF;                             \
        }                                                 \
    };                                                    \
    }

KERNEL_FLOAT_FP16_CAST(float64, __double2half(input), float64(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float32, __float2half(input), __half2float(input));

KERNEL_FLOAT_FP16_CAST(signed int, __half2int_rz(input), __int2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed short, __half2short_rz(input), __short2half_rn(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned int, __half2uint_rz(input), __uint2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned short, __half2ushort_rz(input), __ushort2half_rn(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

template<>
struct map_helper<ops::cast<float16, float32>, float16, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<float16, float32>, const vec<float16, 2>& input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, float16>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<float16, 2>
    call(ops::cast<float32, float16>, const vec<float32, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_FP16_H
