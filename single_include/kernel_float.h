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

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
#define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
#define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
#define KERNEL_FLOAT_FP8_AVAILABLE (0)
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

    KERNEL_FLOAT_INLINE constexpr operator std::integral_constant<size_t, I>() const

        noexcept {
        return {};
    }

    KERNEL_FLOAT_INLINE constexpr operator size_t() const

        noexcept {
        return I;
    }

    KERNEL_FLOAT_INLINE constexpr size_t

    operator()() const

        noexcept {
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
using common_t = typename common_type<typename std::decay<Args>::type...>::type;

template<typename From, typename To>
static constexpr bool is_implicit_convertible = std::is_same<common_t<From, To>, To>::value;

namespace detail {

template<typename T>
struct into_vec_helper {};

template<typename T>
struct into_vec_helper<const T>: into_vec_helper<T> {};

template<typename T>
struct into_vec_helper<T&>: into_vec_helper<T> {};

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
}  // namespace detail

#define KERNEL_FLOAT_INTO_VEC(V, T, N)                               \
    namespace detail {                                               \
    template<>                                                       \
    struct into_vec_helper<V> {                                      \
        using value_type = T;                                        \
        static constexpr size_t size = N;                            \
        KERNEL_FLOAT_INLINE static vec_storage<T, N> call(V input) { \
            return input;                                            \
        }                                                            \
    };                                                               \
    }

namespace detail {
template<typename T>
struct is_vec_helper {
    static constexpr bool value = false;
};

template<typename T, size_t N>
struct is_vec_helper<vec<T, N>> {
    static constexpr bool value = true;
};

template<typename T>
struct is_vec_helper<const T>: is_vec_helper<T> {};

template<typename T>
struct is_vec_helper<T&>: is_vec_helper<T> {};

template<typename T>
struct is_vec_helper<const T&>: is_vec_helper<T> {};

template<typename T>
struct is_vec_helper<T&&>: is_vec_helper<T> {};

template<size_t... Is>
struct common_size {};

template<size_t N>
struct common_size<N> {
    static constexpr size_t value = N;
};

template<size_t N, size_t... Rest>
struct common_size<N, N, Rest...>: common_size<N, Rest...> {};

template<size_t N, size_t... Rest>
struct common_size<N, 1, Rest...>: common_size<N, Rest...> {};

template<size_t N, size_t... Rest>
struct common_size<1, N, Rest...>: common_size<N, Rest...> {};

template<size_t... Rest>
struct common_size<1, 1, Rest...>: common_size<1, Rest...> {};

};  // namespace detail

template<typename T>
static constexpr size_t into_vec_size = detail::into_vec_helper<T>::size;

template<typename T>
using into_vec_value_t = typename detail::into_vec_helper<T>::value_type;

template<typename T>
using into_vec_t = vec<into_vec_value_t<T>, into_vec_size<T>>;

template<typename... Ts>
static constexpr size_t common_vec_size = detail::common_size<into_vec_size<Ts>...>::value;

template<typename... Ts>
using common_vec_value_t = typename common_type<into_vec_value_t<Ts>...>::type;

template<typename T>
static constexpr bool is_vec = detail::is_vec_helper<T>::value;

template<typename T>
KERNEL_FLOAT_INLINE into_vec_t<T> into_vec(T&& input) {
    return detail::into_vec_helper<T>::call(std::forward<T>(input));
}

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

#define KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(T, N)                                      \
    template<size_t... Is>                                                              \
    KERNEL_FLOAT_INLINE vec<T, sizeof...(Is)> get(index_sequence<Is...>) const {        \
        return {this->get(constant_index<Is> {})...};                                   \
    }                                                                                   \
    template<size_t... Is>                                                              \
    KERNEL_FLOAT_INLINE void set(index_sequence<Is...>, vec<T, sizeof...(Is)> values) { \
        assign_helper<index_sequence<Is...>, make_index_sequence<sizeof...(Is)>>::call( \
            *this,                                                                      \
            values);                                                                    \
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

    KERNEL_FLOAT_INLINE vec<T, 2> get(index_sequence<0, 1>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec<T, 2> get(index_sequence<2, 3>) const {
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

    KERNEL_FLOAT_INLINE vec<T, 4> get(index_sequence<0, 1, 2, 3>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec<T, 2> get(index_sequence<4, 5>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1, 2, 3>, vec<T, 4> values) {
        low_ = values;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<4, 5>, vec<T, 2> values) {
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

    KERNEL_FLOAT_INLINE vec<T, 4> get(index_sequence<0, 1, 2, 3>) const {
        return low_;
    }

    KERNEL_FLOAT_INLINE vec<T, 4> get(index_sequence<4, 5, 6, 7>) const {
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

};  // namespace detail

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE_FIELD(T, I, FIELD)     \
    KERNEL_FLOAT_INLINE T get(constant_index<I>) const {       \
        return FIELD;                                          \
    }                                                          \
    KERNEL_FLOAT_INLINE void set(constant_index<I>, T value) { \
        FIELD = value;                                         \
    }

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T2, T3, T4)                                            \
    namespace detail {                                                                            \
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
    };                                                                                            \
    }                                                                                             \
    KERNEL_FLOAT_INTO_VEC(T, T, 1)                                                                \
    KERNEL_FLOAT_INTO_VEC(T2, T, 2)                                                               \
    KERNEL_FLOAT_INTO_VEC(T3, T, 3)                                                               \
    KERNEL_FLOAT_INTO_VEC(T4, T, 4)

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

KERNEL_FLOAT_INTO_VEC(bool, bool, 1)

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

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE T reduce(F fun, const vec<T, N>& input) {
    return reduce_helper<F, T, N>::call(fun, input);
}

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

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T min(const vec<T, N>& input) {
    return reduce(ops::min<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T max(const vec<T, N>& input) {
    return reduce(ops::max<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T sum(const vec<T, N>& input) {
    return reduce(ops::add<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE T product(const vec<T, N>& input) {
    return reduce(ops::mulitply<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE bool all(const vec<T, N>& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE bool any(const vec<T, N>& input) {
    return reduce(ops::bit_or<bool> {}, cast<bool>(input));
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE int count(const vec<T, N>& input) {
    return sum(cast<int>(cast<bool>(input)));
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

template<
    typename V,
    typename F,
    typename T = into_vec_value_t<V>,
    size_t N = into_vec_size<V>,
    typename R = result_t<F, T>>
KERNEL_FLOAT_INLINE vec<R, N> map(F fun, V&& input) noexcept {
    return map_helper<F, T, N>::call(fun, into_vec(input));
}

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
            map(fun, input.get(index_sequence<2, 3> {}))};
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

template<typename R, typename V, typename T = into_vec_value_t<V>, size_t N = into_vec_size<V>>
KERNEL_FLOAT_INLINE vec<R, N> cast(V&& input) noexcept {
    return map(ops::cast<T, R> {}, into_vec(input));
}

namespace detail {
template<typename T, size_t From, size_t To>
struct broadcast_helper;

template<typename T, size_t N>
struct broadcast_helper<T, N, N> {
    KERNEL_FLOAT_INLINE static vec<T, N> call(vec<T, N> v) {
        return v;
    }
};

template<typename T, size_t N>
struct broadcast_helper<T, 1, N> {
    KERNEL_FLOAT_INLINE static vec<T, N> call(vec<T, 1> v) {
        return vec<T, N>(v.get(I0 {}));
    }
};

template<typename T>
struct broadcast_helper<T, 1, 1> {
    KERNEL_FLOAT_INLINE static vec<T, 1> call(vec<T, 1> v) {
        return v;
    }
};
}  // namespace detail

template<typename T, size_t N, typename V>
KERNEL_FLOAT_INLINE vec<T, N> broadcast(V&& input) {
    return detail::broadcast_helper<T, into_vec_size<V>, N>::call(
        map(ops::cast<into_vec_value_t<V>, T> {}, into_vec(input)));
}

#define KERNEL_FLOAT_DEFINE_FUN1_OP(NAME, EXPR)                                         \
    namespace ops {                                                                     \
    template<typename T>                                                                \
    struct NAME {                                                                       \
        KERNEL_FLOAT_INLINE T operator()(T input) {                                     \
            return EXPR;                                                                \
        }                                                                               \
    };                                                                                  \
    }                                                                                   \
    template<typename V, typename T = into_vec_value_t<V>, size_t N = into_vec_size<V>> \
    KERNEL_FLOAT_INLINE vec<T, N> NAME(V&& input) {                                     \
        return map(ops::NAME<T> {}, into_vec(input));                                   \
    }

KERNEL_FLOAT_DEFINE_FUN1_OP(negate, -input)
KERNEL_FLOAT_DEFINE_FUN1_OP(bit_not, ~input)
KERNEL_FLOAT_DEFINE_FUN1_OP(logical_not, !input)

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator-(const vec<T, N>& input) {
    return map(ops::negate<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator~(const vec<T, N>& input) {
    return map(ops::bit_not<T> {}, input);
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vec<T, N> operator!(const vec<T, N>& input) {
    return map(ops::logical_not<T> {}, input);
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
                lhs.get(index_sequence<4, 5> {}),
                rhs.get(index_sequence<4, 5> {}))};
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

template<
    typename F,
    typename A,
    typename B,
    typename T = into_vec_value_t<A>,
    typename U = into_vec_value_t<B>,
    size_t N = common_vec_size<A, B>,
    typename R = result_t<F, T, U>>
KERNEL_FLOAT_INLINE vec<R, N> zip(F fun, A&& lhs, B&& rhs) {
    return zip_helper<F, T, U, N>::call(fun, broadcast<T, N>(lhs), broadcast<U, N>(rhs));
}

template<
    typename F,
    typename A,
    typename B,
    typename C = common_vec_value_t<A, B>,
    size_t N = common_vec_size<A, B>,
    typename R = result_t<F, C, C>>
KERNEL_FLOAT_INLINE vec<R, N> zip_common(F fun, A&& lhs, B&& rhs) {
    return zip_helper<F, C, C, N>::call(fun, broadcast<C, N>(lhs), broadcast<C, N>(rhs));
}

#define KERNEL_FLOAT_DEFINE_FUN2_OP(NAME, EXPR)                               \
    namespace ops {                                                           \
    template<typename T>                                                      \
    struct NAME {                                                             \
        KERNEL_FLOAT_INLINE auto operator()(T lhs, T rhs) -> decltype(EXPR) { \
            return EXPR;                                                      \
        }                                                                     \
    };                                                                        \
    }                                                                         \
    template<                                                                 \
        typename A,                                                           \
        typename B,                                                           \
        typename C = common_vec_value_t<A, B>,                                \
        size_t N = common_vec_size<A, B>,                                     \
        typename R = result_t<ops::NAME<C>, C, C>>                            \
    KERNEL_FLOAT_INLINE vec<R, N> NAME(A&& lhs, B&& rhs) {                    \
        return zip_common(ops::NAME<C> {}, lhs, rhs);                         \
    }

#define KERNEL_FLOAT_DEFINE_BINOP(NAME, OP)                                       \
    KERNEL_FLOAT_DEFINE_FUN2_OP(NAME, lhs OP rhs)                                 \
    template<                                                                     \
        typename A,                                                               \
        typename B,                                                               \
        typename C = enabled_t<is_vec<A> || is_vec<B>, common_vec_value_t<A, B>>, \
        size_t N = common_vec_size<A, B>,                                         \
        typename R = result_t<ops::NAME<C>, C, C>>                                \
    KERNEL_FLOAT_INLINE vec<R, N> operator OP(A&& lhs, B&& rhs) {                 \
        return zip_common(ops::NAME<C> {}, lhs, rhs);                             \
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

    KERNEL_FLOAT_INLINE vec(T item) : vec(full<N>(item)) {}

    KERNEL_FLOAT_INLINE vec() : vec(T {}) {}

    KERNEL_FLOAT_INLINE vec(storage_type storage) : storage_type {storage} {}

    template<
        typename U,
        size_t M,
        typename = enabled_t<is_implicit_convertible<U, T> && (M == 1 || M == N)>>
    KERNEL_FLOAT_INLINE vec(const vec<U, M>& that) : vec(broadcast<T, N>(that)) {}

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
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H



#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>



namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, __half)

namespace detail {
template<>
struct vec_storage<__half, 2> {
    static_assert(sizeof(__half) * 2 == sizeof(__half2), "invalid size");
    static_assert(alignof(__half) <= alignof(__half2), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(__half x, __half y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(__half2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __half2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __half get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __half get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, __half v) {
        *this = vec_storage(v, get(I1 {}));
    }

    KERNEL_FLOAT_INLINE void set(I1, __half v) {
        *this = vec_storage(get(I0 {}), v);
    }

    KERNEL_FLOAT_INLINE __half get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __half value) const {
        if (index == 0) {
            set(I0 {}, value);
        } else {
            set(I1 {}, value);
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(__half, 2)

#if KERNEL_FLOAT_CUDA_DEVICE
    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<0, 0>) const {
        return __low2half2(vector_);
    }

    KERNEL_FLOAT_INLINE vec<__half, 2> get(index_sequence<1, 1>) const {
        return __high2half2(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, __half2 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, __half2 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    __half2 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_CUDA_DEVICE
#define KERNEL_FLOAT_FP16_MONOP(NAME, FUN1, FUN2)             \
    namespace ops {                                           \
    template<>                                                \
    struct NAME<__half> {                                     \
        KERNEL_FLOAT_INLINE __half operator()(__half input) { \
            return FUN1(input);                               \
        }                                                     \
    };                                                        \
    }                                                         \
    template<>                                                \
    struct map_helper<ops::NAME<__half>, __half, 2> {         \
        KERNEL_FLOAT_INLINE static vec<__half, 2>             \
        call(ops::NAME<__half>, __half2 input) noexcept {     \
            return FUN2(input);                               \
        }                                                     \
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

#define KERNEL_FLOAT_FP16_BINOP(NAME, FUN1, FUN2)                                              \
    namespace ops {                                                                            \
    template<>                                                                                 \
    struct NAME<__half> {                                                                      \
        KERNEL_FLOAT_INLINE __half operator()(__half lhs, __half rhs) {                        \
            return FUN1(lhs, rhs);                                                             \
        }                                                                                      \
    };                                                                                         \
    }                                                                                          \
    template<>                                                                                 \
    struct zip_helper<ops::NAME<__half>, __half, __half, 2> {                                  \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, __half2 lhs, __half2 rhs) { \
            return FUN2(lhs, rhs);                                                             \
        }                                                                                      \
    };

KERNEL_FLOAT_FP16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_FP16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_FP16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_FP16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_FP16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_FP16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_FP16_RELOP(NAME, FUN1, FUN2)                     \
    namespace ops {                                                   \
    template<>                                                        \
    struct NAME<__half> {                                             \
        KERNEL_FLOAT_INLINE bool operator()(__half lhs, __half rhs) { \
            return FUN1(lhs, rhs);                                    \
        }                                                             \
    };                                                                \
    }

KERNEL_FLOAT_FP16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_FP16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_FP16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_FP16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_FP16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_FP16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)    \
    namespace ops {                                      \
    template<>                                           \
    struct cast<T, __half> {                             \
        KERNEL_FLOAT_INLINE __half operator()(T input) { \
            return TO_HALF;                              \
        }                                                \
    };                                                   \
    template<>                                           \
    struct cast<__half, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__half input) { \
            return FROM_HALF;                            \
        }                                                \
    };                                                   \
    }

KERNEL_FLOAT_FP16_CAST(float64, __double2half(input), float64(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float32, __float2half(input), __half2float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_FP16_CAST(char, __int2half_rn(input), (char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed char, __int2half_rn(input), (signed char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned char, __int2half_rn(input), (unsigned char)__half2int_rz(input));

KERNEL_FLOAT_FP16_CAST(signed short, __short2half_rn(input), __half2short_rz(input));
KERNEL_FLOAT_FP16_CAST(signed int, __int2half_rn(input), __half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned int, __uint2half_rn(input), __half2uint_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned short, __ushort2half_rn(input), __half2ushort_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));

template<>
struct map_helper<ops::cast<__half, float32>, __half, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<__half, float32>, const vec<__half, 2>& input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, __half>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<__half, 2>
    call(ops::cast<float32, __half>, const vec<float32, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

KERNEL_FLOAT_INTO_VEC(__half, __half, 1)
KERNEL_FLOAT_INTO_VEC(__half2, __half, 2)

}  // namespace kernel_float

#endif
#endif  //KERNEL_FLOAT_FP16_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H



#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>



namespace kernel_float {

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, __nv_bfloat16)

namespace detail {
template<>
struct vec_storage<__nv_bfloat16, 2> {
    static_assert(sizeof(__nv_bfloat16) * 2 == sizeof(__nv_bfloat162), "invalid size");
    static_assert(alignof(__nv_bfloat16) <= alignof(__nv_bfloat162), "invalid size");

    KERNEL_FLOAT_INLINE vec_storage(__nv_bfloat16 x, __nv_bfloat16 y) noexcept : vector_ {x, y} {}

    KERNEL_FLOAT_INLINE vec_storage(__nv_bfloat162 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __nv_bfloat162() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(I0) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(I1) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(I0, __nv_bfloat16 v) {
        *this = vec_storage(v, __high2float(vector_));
    }

    KERNEL_FLOAT_INLINE void set(I1, __nv_bfloat16 v) {
        *this = vec_storage(__low2float(vector_), v);
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(size_t index) const {
        if (index == 0) {
            return get(I0 {});
        } else {
            return get(I1 {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __nv_bfloat16 value) const {
        if (index == 0) {
            set(I0 {}, value);
        } else {
            set(I1 {}, value);
        }
    }

    KERNEL_FLOAT_STORAGE_MULTI_ACCESSORS(__nv_bfloat16, 2)

#if KERNEL_FLOAT_CUDA_DEVICE && __CUDA_ARCH__ >= 800
    KERNEL_FLOAT_INLINE vec<__nv_bfloat16, 2> get(index_sequence<0, 1>) const {
        return vector_;
    }

    KERNEL_FLOAT_INLINE vec<__nv_bfloat16, 2> get(index_sequence<1, 0>) const {
        return __lowhigh2highlow(vector_);
    }

    KERNEL_FLOAT_INLINE vec<__nv_bfloat16, 2> get(index_sequence<0, 0>) const {
        return {vector_.x, vector_.x};
    }

    KERNEL_FLOAT_INLINE vec<__nv_bfloat16, 2> get(index_sequence<1, 1>) const {
        return __high2bfloat162(vector_);
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<0, 1>, __nv_bfloat162 v) {
        vector_ = v;
    }

    KERNEL_FLOAT_INLINE void set(index_sequence<1, 0>, __nv_bfloat162 v) {
        vector_ = __lowhigh2highlow(v);
    }
#endif

  private:
    __nv_bfloat162 vector_;
};
}  // namespace detail

#if KERNEL_FLOAT_CUDA_DEVICE && __CUDA_ARCH__ >= 800
#define KERNEL_FLOAT_BF16_MONOP(NAME, FUN1, FUN2)                           \
    namespace ops {                                                         \
    template<>                                                              \
    struct NAME<__nv_bfloat16> {                                            \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) { \
            return FUN1(input);                                             \
        }                                                                   \
    };                                                                      \
    }                                                                       \
    template<>                                                              \
    struct map_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat16, 2> {         \
        KERNEL_FLOAT_INLINE static vec<__nv_bfloat16, 2>                    \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 input) noexcept {     \
            return FUN2(input);                                             \
        }                                                                   \
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

#define KERNEL_FLOAT_BF16_BINOP(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                          \
    template<>                                                                               \
    struct NAME<__nv_bfloat16> {                                                             \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 lhs, __nv_bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                                           \
        }                                                                                    \
    };                                                                                       \
    }                                                                                        \
    template<>                                                                               \
    struct zip_helper<ops::NAME<__nv_bfloat16>, __nv_bfloat16, __nv_bfloat16, 2> {           \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                            \
        call(ops::NAME<__nv_bfloat16>, __nv_bfloat162 lhs, __nv_bfloat162 rhs) {             \
            return FUN2(lhs, rhs);                                                           \
        }                                                                                    \
    };

KERNEL_FLOAT_BF16_BINOP(add, __hadd, __hadd2);
KERNEL_FLOAT_BF16_BINOP(subtract, __hsub, __hsub2);
KERNEL_FLOAT_BF16_BINOP(mulitply, __hmul, __hmul2);
KERNEL_FLOAT_BF16_BINOP(divide, __hdiv, __h2div);
KERNEL_FLOAT_BF16_BINOP(min, __hmin, __hmin2);
KERNEL_FLOAT_BF16_BINOP(max, __hmax, __hmax2);

#define KERNEL_FLOAT_BF16_RELOP(NAME, FUN1, FUN2)                                   \
    namespace ops {                                                                 \
    template<>                                                                      \
    struct NAME<__nv_bfloat16> {                                                    \
        KERNEL_FLOAT_INLINE bool operator()(__nv_bfloat16 lhs, __nv_bfloat16 rhs) { \
            return FUN1(lhs, rhs);                                                  \
        }                                                                           \
    };                                                                              \
    }

KERNEL_FLOAT_BF16_RELOP(equal_to, __heq, __heq2);
KERNEL_FLOAT_BF16_RELOP(not_equal_to, __hne, __hne2);
KERNEL_FLOAT_BF16_RELOP(greater, __hgt, __hgt2);
KERNEL_FLOAT_BF16_RELOP(greater_equal, __hge, __hge2);
KERNEL_FLOAT_BF16_RELOP(less, __hlt, __hlt2);
KERNEL_FLOAT_BF16_RELOP(less_equal, __hle, __hle2);
#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)           \
    namespace ops {                                             \
    template<>                                                  \
    struct cast<T, __nv_bfloat16> {                             \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(T input) { \
            return TO_HALF;                                     \
        }                                                       \
    };                                                          \
    template<>                                                  \
    struct cast<__nv_bfloat16, T> {                             \
        KERNEL_FLOAT_INLINE T operator()(__nv_bfloat16 input) { \
            return FROM_HALF;                                   \
        }                                                       \
    };                                                          \
    }

KERNEL_FLOAT_BF16_CAST(float64, __double2bfloat16(input), float64(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float32, __float2bfloat16(input), __bfloat162float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    signed char,
    __int2bfloat16_rn(input),
    (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(
    unsigned char,
    __int2bfloat16_rn(input),
    (unsigned char)__bfloat162int_rz(input));

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
struct map_helper<ops::cast<__nv_bfloat16, float32>, __nv_bfloat16, 2> {
    KERNEL_FLOAT_INLINE static vec<float32, 2>
    call(ops::cast<__nv_bfloat16, float32>, const vec<__nv_bfloat16, 2>& input) noexcept {
        return __bfloat1622float2(input);
    }
};

template<>
struct map_helper<ops::cast<float32, __nv_bfloat16>, float32, 2> {
    KERNEL_FLOAT_INLINE static vec<__nv_bfloat16, 2>
    call(ops::cast<float32, __nv_bfloat16>, const vec<float32, 2>& input) noexcept {
        return __float22bfloat162_rn(input);
    }
};

KERNEL_FLOAT_INTO_VEC(__nv_bfloat16, __nv_bfloat16, 1)
KERNEL_FLOAT_INTO_VEC(__nv_bfloat162, __nv_bfloat16, 2)

}  // namespace kernel_float
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE


namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE
#endif  // KERNEL_FLOAT_BF16_H
#ifndef KERNEL_FLOAT_FP8_H
#define KERNEL_FLOAT_FP8_H



#if KERNEL_FLOAT_BF8_AVAILABLE




namespace kernel_float {
using float8_e4m3 = __nv_fp8_e4m3;
using float8_e5m2 = __nv_fp8_e5m2;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float16, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(bfloat16, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float8_e4m3)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float16, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(bfloat16, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float8_e5m2)

namespace detail {
template<>
struct vec_storage<float8_e4m3, 2> {
    vec_storage(__nv_fp8_e4m3 x, float8_e4m3 y) : array {x, y} {}
    vec_storage(__nv_fp8x2_e4m3 v) : storage_(v) {}

    operator __nv_fp8x2_e4m3() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e4m3, 2)

  private:
    union {
        __nv_fp8x2_e4m3 storage_;
        __nv_fp8_e4m3 array_[2];
    }
};

template<>
struct vec_storage<float8_e4m3, 4> {
    vec_storage(__nv_fp8_e4m3 x, float8_e4m3 y, __nv_fp8_e4m3 z, float8_e4m3 w) :
        array {x, y, z, w} {}
    vec_storage(__nv_fp8x4_e4m3 v) : storage_(v) {}

    operator __nv_fp8x4_e4m3() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e4m3, 4)

    KERNEL_FLOAT_INLINE vec_storage<T, 2> get(index_sequence<4, 5>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e4m3, 2> get(index_sequence<0, 1>) const {
        __nv_fp8x2_e4m3 out;
        out.__x = storage_.__x;
        return out;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e4m3, 2> get(index_sequence<2, 3>) const {
        __nv_fp8x2_e4m3 out;
        out.__x = storage_.__x >> 16;
        return out;
    }

  private:
    union {
        __nv_fp8x4_e4m3 storage_;
        __nv_fp8_e4m3 array_[4];
    }
};

template<>
struct vec_storage<float8_e5m2, 2> {
    vec_storage(__nv_fp8_e5m2 x, float8_e5m2 y) : array {x, y} {}
    vec_storage(__nv_fp8x2_e5m2 v) : storage_(v) {}

    operator __nv_fp8x2_e5m2() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e5m2, 2)

  private:
    union {
        __nv_fp8x2_e5m2 storage_;
        __nv_fp8_e5m2 array_[2];
    }
};

template<>
struct vec_storage<float8_e5m2, 4> {
    vec_storage(__nv_fp8_e5m2 x, float8_e5m2 y, __nv_fp8_e5m2 z, float8_e5m2 w) :
        array {x, y, z, w} {}
    vec_storage(__nv_fp8x4_e5m2 v) : storage_(v) {}

    operator __nv_fp8x4_e5m2() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e5m2, 4)

    KERNEL_FLOAT_INLINE vec_storage<float8_e5m2, 2> get(index_sequence<0, 1>) const {
        __nv_fp8x2_e5m2 out;
        out.__x = (storage_.__x) & 0xffff;
        return out;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e5m2, 2> get(index_sequence<2, 3>) const {
        __nv_fp8x2_e5m2 out;
        out.__x = (storage_.__x >> 16) & 0xffff;
        return out;
    }

  private:
    union {
        __nv_fp8x4_e5m2 storage_;
        __nv_fp8_e5m2 array_[4];
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e4m3, T>, float8_e4m3, 2> {
    KERNEL_FLOAT_INLINE static vec<T, 2> call(ops::cast<float8_e4m3, T>, __nv_fp8x2_e4m3 input) {
        return cast<T>(vec<half, 2>(__half2(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<T, float8_e4m3>, T, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e4m3, 2>
    call(ops::cast<T, float8_e4m3>, vec<T, 2> input) {
        return __nv_fp8x2_e4m3(__half2(cast<half, 2>(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e5m2, T>, float8_e5m2, 2> {
    KERNEL_FLOAT_INLINEstatic vec<T, 2> call(ops::cast<float8_e5m2, T>, __nv_fp8x2_e5m2 input) {
        return cast<T>(vec<half, 2>(__half2(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<T, float8_e5m2>, T, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e5m2, 2>
    call(ops::cast<T, float8_e5m2>, vec<T, 2> input) {
        return __nv_fp8x2_e5m2(__half2(cast<half, 2>(input)));
    }
};

namespace ops {
struct cast<float8_e4m3, float8_e5m2> {
    KERNEL_FLOAT_INLINE float8_e5m2 operator()(float8_e4m3 v) const {
        return float8_e5m2(__half(v));
    }
};

struct cast<float8_e5m2, float8_e4m3> {
    KERNEL_FLOAT_INLINE float8_e4m3 operator()(float8_e5m2 v) const {
        return float8_e4m3(__half(v));
    }
};
}  // namespace ops

template<typename T>
struct map_helper<ops::cast<float8_e4m3, float8_e5m2>, float8_e4m3, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e5m2, 2>
    call(ops::cast<float8_e4m3, float8_e5m2>, __nv_fp8x2_e4m3 input) {
        return __nv_fp8x2_e5m2(__half2(input));
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e5m2, float8_e4m3>, float8_e5m2, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e4m3, 2>
    call(ops::cast<float8_e5m2, float8_e4m3>, __nv_fp8x2_e5m2 input) {
        return __nv_fp8x2_e4m3(__half2(input));
    }
};
}  // namespace detail

KERNEL_FLOAT_INTO_VEC(__nv_fp8_e5m2, __nv_fp8_e5m2, 1)
KERNEL_FLOAT_INTO_VEC(__nv_fp8x2_e5m2, __nv_fp8_e5m2, 2)
KERNEL_FLOAT_INTO_VEC(__nv_fp8x4_e5m2, __nv_fp8_e5m2, 4)

KERNEL_FLOAT_INTO_VEC(__nv_fp8_e4m3, __nv_fp8_e4m3, 1)
KERNEL_FLOAT_INTO_VEC(__nv_fp8x2_e4m3, __nv_fp8_e4m3, 2)
KERNEL_FLOAT_INTO_VEC(__nv_fp8x4_e4m3, __nv_fp8_e4m3, 4)

}  // namespace kernel_float

#endif
#endif  //KERNEL_FLOAT_FP8_H
#ifndef KERNEL_FLOAT_TYPES_H
#define KERNEL_FLOAT_TYPES_H



namespace kernel_float {

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

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T) \
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

#if KERNEL_FLOAT_FP16_AVAILABLE
using float16 = half;
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
using bfloat16 = __nv_bfloat16;
KERNEL_FLOAT_TYPE_ALIAS(bfloat16x, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bf16x, __nv_bfloat16)
#endif
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_TYPES_H
