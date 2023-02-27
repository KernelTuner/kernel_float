//================================================================================
// this file has been auto-generated, do not modify its contents!
// date: 2023-02-27 10:52:31.143841
// git hash: d60e0e6ab20801cfcd7322bd3199e2016321fa43
//================================================================================

#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __CUDACC__
#define KERNEL_FLOAT_CUDA (1)

#ifdef __CUDA_ARCH__
#define KERNEL_FLOAT_INLINE    __forceinline__ __device__
#define KERNEL_FLOAT_ON_DEVICE (1)
#define KERNEL_FLOAT_ON_HOST   (0)
#define KERNEL_FLOAT_CUDA_ARCH (__CUDA_ARCH__)
#else
#define KERNEL_FLOAT_INLINE    __forceinline__ __host__
#define KERNEL_FLOAT_ON_DEVICE (0)
#define KERNEL_FLOAT_ON_HOST   (1)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif
#else
#define KERNEL_FLOAT_INLINE    inline
#define KERNEL_FLOAT_CUDA      (0)
#define KERNEL_FLOAT_ON_HOST   (1)
#define KERNEL_FLOAT_ON_DEVICE (0)
#define KERNEL_FLOAT_CUDA_ARCH (0)
#endif

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
#define KERNEL_FLOAT_FP16_AVAILABLE (KERNEL_FLOAT_CUDA)
#endif

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
#define KERNEL_FLOAT_BF16_AVAILABLE (KERNEL_FLOAT_CUDA)
#endif

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
#define KERNEL_FLOAT_FP8_AVAILABLE (0)
#endif

#endif  //KERNEL_FLOAT_MACROS_H
#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H



namespace kernel_float {

template<size_t I>
struct const_index {
    static constexpr size_t value = I;

    KERNEL_FLOAT_INLINE constexpr operator size_t() const noexcept {
        return I;
    }
};

template<size_t... Is>
struct index_sequence {};

namespace detail {
template<size_t N, size_t... Is>
struct make_index_sequence_helper: make_index_sequence_helper<N - 1, N - 1, Is...> {};

template<size_t... Is>
struct make_index_sequence_helper<0, Is...> {
    using type = index_sequence<Is...>;
};

}  // namespace detail

template<size_t N>
using make_index_sequence = typename detail::make_index_sequence_helper<N>::type;

namespace detail {
template<typename T>
struct decay_helper {
    using type = T;
};

template<typename T>
struct decay_helper<const T> {
    using type = T;
};

template<typename T>
struct decay_helper<const T&> {
    using type = T;
};

template<typename T>
struct decay_helper<T&> {
    using type = T;
};

template<typename T>
struct decay_helper<T&&> {
    using type = T;
};
}  // namespace detail

template<typename T>
using decay_t = typename detail::decay_helper<T>::type;

template<typename T, typename U>
struct common_type;

template<typename T>
struct common_type<T, T> {
    using type = T;
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

namespace detail {
template<typename... Ts>
struct common_type_helper;

template<typename T>
struct common_type_helper<T> {
    using type = T;
};

template<typename T, typename U, typename... Rest>
struct common_type_helper<T, U, Rest...>:
    common_type_helper<typename common_type<T, U>::type, Rest...> {};
}  // namespace detail

template<typename... Ts>
using common_t = typename detail::common_type_helper<decay_t<Ts>...>::type;

namespace detail {
template<size_t...>
struct common_size_helper;

template<>
struct common_size_helper<> {
    static constexpr size_t value = 1;
};

template<size_t N>
struct common_size_helper<N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<N, N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<N, 1> {
    static constexpr size_t value = N;
};

template<size_t N>
struct common_size_helper<1, N> {
    static constexpr size_t value = N;
};

template<>
struct common_size_helper<1, 1> {
    static constexpr size_t value = 1;
};
}  // namespace detail

template<size_t... Ns>
static constexpr size_t common_size = detail::common_size_helper<Ns...>::value;

namespace detail {
template<typename From, typename To, typename Common = typename common_type<From, To>::type>
struct is_implicit_convertible_helper {
    static constexpr bool value = false;
};

template<typename From, typename To>
struct is_implicit_convertible_helper<From, To, To> {
    static constexpr bool value = true;
};
}  // namespace detail

template<typename From, typename To>
static constexpr bool is_implicit_convertible =
    detail::is_implicit_convertible_helper<decay_t<From>, decay_t<To>>::value;

template<typename F, typename... Args>
using result_t = decltype((std::declval<F>())(std::declval<Args>()...));

namespace detail {
template<bool, typename T>
struct enabled_helper {};

template<typename T>
struct enabled_helper<true, T> {
    using type = T;
};
}  // namespace detail

template<bool C, typename T = void>
using enabled_t = typename detail::enabled_helper<C, T>::type;

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_STORAGE_H
#define KERNEL_FLOAT_STORAGE_H



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

template<typename Output, typename Input, typename Indices, typename = void>
struct vector_swizzle;

template<typename Output, typename Input, size_t... Is>
struct vector_swizzle<Output, Input, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static Output call(const Input& storage) {
        return Output {storage.get(const_index<Is> {})...};
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
#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H



namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Input, typename = void>
struct map_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input) {
        return call(fun, input, make_index_sequence<vector_size<Input>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output call(F fun, const Input& input, index_sequence<Is...>) {
        return Output {fun(input.get(const_index<Is> {}))...};
    }
};

template<typename F, typename R, typename T, size_t N>
struct map_helper<F, vector_compound<R, N>, vector_compound<T, N>> {
    KERNEL_FLOAT_INLINE static vector_compound<R, N>
    call(F fun, const vector_compound<T, N>& input) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return {
            map_helper<F, vector_storage<R, low_size>, vector_storage<T, low_size>>::call(
                fun,
                input.low()),
            map_helper<F, vector_storage<R, high_size>, vector_storage<T, high_size>>::call(
                fun,
                input.high())};
    }
};
}  // namespace detail

template<typename F, typename Input>
using map_type = vector_storage<result_t<F, vector_value_type<Input>>, vector_size<Input>>;

template<typename F, typename Input, typename Output = map_type<F, Input>>
KERNEL_FLOAT_INLINE Output map(F fun, Input&& input) {
    return detail::map_helper<F, Output, into_vector_type<Input>>::call(fun, into_vector(input));
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

namespace detail {
template<
    typename Input,
    typename Output,
    typename T = vector_value_type<Input>,
    size_t N = vector_size<Input>,
    typename R = vector_value_type<Output>,
    size_t M = vector_size<Output>>
struct broadcast_helper;

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, Vector, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

template<typename Vector, typename T>
struct broadcast_helper<Vector, Vector, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Vector call(Vector input) {
        return input;
    }
};

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, Vector&, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(const Vector& input) {
        return input;
    }
};

template<typename Vector, typename T, size_t N>
struct broadcast_helper<Vector, const Vector&, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Vector call(const Vector& input) {
        return input;
    }
};

template<typename Input, typename Output, typename T, size_t N>
struct broadcast_helper<Input, Output, T, N, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        using F = ops::cast<T, T>;
        return map_helper<F, Output, Input>::call(F {}, std::forward<Input>(input));
    }
};

template<typename Output, typename Input, typename T, size_t N>
struct broadcast_helper<Input, Output, T, 1, T, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {into_vector(std::forward<Input>(input)).get(const_index<0> {})};
    }
};

template<typename Output, typename Input, typename T>
struct broadcast_helper<Input, Output, T, 1, T, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {into_vector(std::forward<Input>(input)).get(const_index<0> {})};
    }
};

template<typename Output, typename Input, typename T, typename R>
struct broadcast_helper<Input, Output, T, 1, R, 1> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {
            ops::cast<T, R> {}(into_vector(std::forward<Input>(input)).get(const_index<0> {}))};
    }
};

template<typename Output, typename Input, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, 1, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        return Output {
            ops::cast<T, R> {}(into_vector(std::forward<Input>(input)).get(const_index<0> {}))};
    }
};

template<typename Output, typename Input, typename T, typename R, size_t N>
struct broadcast_helper<Input, Output, T, N, R, N> {
    KERNEL_FLOAT_INLINE static Output call(Input&& input) {
        using F = ops::cast<T, R>;
        return map_helper<F, Output, Input>::call(F {}, std::forward<Input>(input));
    }
};
}  // namespace detail

template<typename R, size_t N, typename Input, typename Output = vector_storage<R, N>>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<size_t N, typename Input, typename Output = vector_storage<vector_value_type<Input>, N>>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<typename Output, typename Input>
KERNEL_FLOAT_INLINE Output broadcast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

template<typename R, typename Input>
using cast_type = vector_storage<R, vector_size<Input>>;

template<typename R, typename Input, typename Output = cast_type<R, Input>>
KERNEL_FLOAT_INLINE Output cast(Input&& input) noexcept {
    return detail::broadcast_helper<Input, Output>::call(std::forward<Input>(input));
}

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                   \
    namespace ops {                                                             \
    template<typename T>                                                        \
    struct NAME {                                                               \
        KERNEL_FLOAT_INLINE T operator()(T input) {                             \
            return T(EXPR);                                                     \
        }                                                                       \
    };                                                                          \
    }                                                                           \
    template<typename V>                                                        \
    KERNEL_FLOAT_INLINE into_vector_type<V> NAME(V&& input) {                   \
        return map(ops::NAME<vector_value_type<V>> {}, std::forward<V>(input)); \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                                        \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                                   \
    template<typename V>                                                                    \
    KERNEL_FLOAT_INLINE enabled_t<is_vector<V>, into_vector_type<V>> operator OP(V&& vec) { \
        return NAME(std::forward<V>(vec));                                                  \
    }

KERNEL_FLOAT_DEFINE_UNARY_OP(negate, -, -input)
KERNEL_FLOAT_DEFINE_UNARY_OP(bit_not, ~, ~input)
KERNEL_FLOAT_DEFINE_UNARY_OP(logical_not, !, !bool(input))

#define KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME) KERNEL_FLOAT_DEFINE_UNARY(NAME, ::NAME(input))

KERNEL_FLOAT_DEFINE_UNARY_FUN(acos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(abs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(acosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(asinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(atanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ceil)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cos)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cosh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(cospi)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfcx)
KERNEL_FLOAT_DEFINE_UNARY_FUN(erfinv)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(exp2)
KERNEL_FLOAT_DEFINE_UNARY_FUN(expm1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(fabs)
KERNEL_FLOAT_DEFINE_UNARY_FUN(floor)
KERNEL_FLOAT_DEFINE_UNARY_FUN(ilogb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(lgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log)
KERNEL_FLOAT_DEFINE_UNARY_FUN(log10)
KERNEL_FLOAT_DEFINE_UNARY_FUN(logb)
KERNEL_FLOAT_DEFINE_UNARY_FUN(nearbyint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(normcdf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rcbrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sin)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sinh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tan)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tanh)
KERNEL_FLOAT_DEFINE_UNARY_FUN(tgamma)
KERNEL_FLOAT_DEFINE_UNARY_FUN(trunc)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y0)
KERNEL_FLOAT_DEFINE_UNARY_FUN(y1)
KERNEL_FLOAT_DEFINE_UNARY_FUN(yn)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rint)
KERNEL_FLOAT_DEFINE_UNARY_FUN(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN(round)
KERNEL_FLOAT_DEFINE_UNARY_FUN(signbit)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isinf)
KERNEL_FLOAT_DEFINE_UNARY_FUN(isnan)

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_UNOPS_H
#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H



namespace kernel_float {
namespace detail {
template<typename F, typename Output, typename Left, typename Right, typename = void>
struct zip_helper {
    KERNEL_FLOAT_INLINE static Output call(F fun, const Left& left, const Right& right) {
        return call_with_indices(fun, left, right, make_index_sequence<vector_size<Output>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static Output
    call_with_indices(F fun, const Left& left, const Right& right, index_sequence<Is...> = {}) {
        return Output {fun(left.get(const_index<Is> {}), right.get(const_index<Is> {}))...};
    }
};

template<typename F, typename T, typename L, typename R, size_t N>
struct zip_helper<F, vector_compound<T, N>, vector_compound<L, N>, vector_compound<R, N>> {
    KERNEL_FLOAT_INLINE static vector_compound<T, N>
    call(F fun, const vector_compound<L, N>& left, const vector_compound<R, N>& right) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return {
            zip_helper<
                F,
                vector_storage<T, low_size>,
                vector_storage<L, low_size>,
                vector_storage<R, low_size>>::call(fun, left.low(), right.low()),
            zip_helper<
                F,
                vector_storage<T, high_size>,
                vector_storage<L, high_size>,
                vector_storage<R, high_size>>::call(fun, left.high(), right.high())};
    }
};
};  // namespace detail

template<typename... Ts>
using common_vector_value_type = common_t<vector_value_type<Ts>...>;

template<typename... Ts>
static constexpr size_t common_vector_size = common_size<vector_size<Ts>...>;

template<typename F, typename L, typename R>
using zip_type = vector_storage<
    result_t<F, vector_value_type<L>, vector_value_type<R>>,
    common_vector_size<L, R>>;

template<typename F, typename Left, typename Right, typename Output = zip_type<F, Left, Right>>
KERNEL_FLOAT_INLINE Output zip(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    return detail::zip_helper<F, Output, into_vector_type<Left>, into_vector_type<Right>>::call(
        fun,
        broadcast<N>(std::forward<Left>(left)),
        broadcast<N>(std::forward<Right>(right)));
}

template<typename F, typename L, typename R>
using zip_common_type = vector_storage<
    result_t<F, common_vector_value_type<L, R>, common_vector_value_type<L, R>>,
    common_vector_size<L, R>>;

template<
    typename F,
    typename Left,
    typename Right,
    typename Output = zip_common_type<F, Left, Right>>
KERNEL_FLOAT_INLINE Output zip_common(F fun, Left&& left, Right&& right) {
    static constexpr size_t N = vector_size<Output>;
    using C = common_t<vector_value_type<Left>, vector_value_type<Right>>;

    return detail::zip_helper<F, Output, vector_storage<C, N>, vector_storage<C, N>>::call(
        fun,
        broadcast<C, N>(std::forward<Left>(left)),
        broadcast<C, N>(std::forward<Right>(right)));
}

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR)                                             \
    namespace ops {                                                                        \
    template<typename T>                                                                   \
    struct NAME {                                                                          \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) {                                \
            return T(EXPR);                                                                \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    template<typename L, typename R, typename C = common_vector_value_type<L, R>>          \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {    \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right)); \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP)                                                \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right)                                            \
    template<                                                                                  \
        typename L,                                                                            \
        typename R,                                                                            \
        typename C = enabled_t<is_vector<L> || is_vector<R>, common_vector_value_type<L, R>>>  \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> operator OP(L&& left, R&& right) { \
        return zip_common(ops::NAME<C> {}, std::forward<L>(left), std::forward<R>(right));     \
    }

KERNEL_FLOAT_DEFINE_BINARY_OP(add, +)
KERNEL_FLOAT_DEFINE_BINARY_OP(subtract, -)
KERNEL_FLOAT_DEFINE_BINARY_OP(divide, /)
KERNEL_FLOAT_DEFINE_BINARY_OP(multiply, *)
KERNEL_FLOAT_DEFINE_BINARY_OP(modulo, %)

KERNEL_FLOAT_DEFINE_BINARY_OP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINARY_OP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINARY_OP(less, <)
KERNEL_FLOAT_DEFINE_BINARY_OP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater, >)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater_equal, >=)

KERNEL_FLOAT_DEFINE_BINARY_OP(bit_and, &)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_or, |)
KERNEL_FLOAT_DEFINE_BINARY_OP(bit_xor, ^)

#define KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME) KERNEL_FLOAT_DEFINE_BINARY(NAME, ::NAME(left, right))

KERNEL_FLOAT_DEFINE_BINARY_FUN(min)
KERNEL_FLOAT_DEFINE_BINARY_FUN(max)
KERNEL_FLOAT_DEFINE_BINARY_FUN(copysign)
KERNEL_FLOAT_DEFINE_BINARY_FUN(hypot)
KERNEL_FLOAT_DEFINE_BINARY_FUN(modf)
KERNEL_FLOAT_DEFINE_BINARY_FUN(nextafter)
KERNEL_FLOAT_DEFINE_BINARY_FUN(pow)
KERNEL_FLOAT_DEFINE_BINARY_FUN(remainder)

#if KERNEL_FLOAT_CUDA_DEVICE
KERNEL_FLOAT_DEFINE_BINARY_FUN(rhypot)
#endif

namespace ops {
template<>
struct add<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left || right;
    }
};

template<>
struct multiply<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) {
        return left && right;
    }
};

template<>
struct bit_and<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) && bool(right));
    }
};

template<>
struct bit_or<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) {
        return float(bool(left) ^ bool(right));
    }
};

template<>
struct bit_and<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) && bool(right));
    }
};

template<>
struct bit_or<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) || bool(right));
    }
};

template<>
struct bit_xor<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) {
        return double(bool(left) ^ bool(right));
    }
};
};  // namespace ops

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_BINOPS_H
#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H




namespace kernel_float {

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct range_helper;

template<typename F, typename V, size_t... Is>
struct range_helper<F, V, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE static V call(F fun) {
        return V {fun(const_index<Is> {})...};
    }
};
}  // namespace detail

template<typename V, typename F>
KERNEL_FLOAT_INLINE V range(F fun) {
    return detail::range_helper<F, V>::call(fun);
}

template<typename V>
KERNEL_FLOAT_INLINE V range() {
    return range<V>(ops::cast<size_t, vector_value_type<V>> {});
}

template<typename T, size_t N>
KERNEL_FLOAT_INLINE vector_storage<T, N> range() {
    return range<vector_storage<T, N>>();
}

template<size_t N>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range() {
    return range<vector_storage<size_t, N>>();
}

template<size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range(F fun) {
    return range<vector_storage<T, N>>(fun);
}

template<typename T, size_t N, typename F>
KERNEL_FLOAT_INLINE vector_storage<size_t, N> range(F fun) {
    return range<vector_storage<T, N>>(fun);
}

namespace detail {
template<typename F, typename V, typename Indices = make_index_sequence<vector_size<V>>>
struct iterate_helper;

template<typename F, typename V>
struct iterate_helper<F, V, index_sequence<>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {}
};

template<typename F, typename V, size_t I, size_t... Rest>
struct iterate_helper<F, V, index_sequence<I, Rest...>> {
    KERNEL_FLOAT_INLINE
    static void call(F fun, const V& input) {
        fun(input.get(const_index<I> {}));
        iterate_helper<F, V, index_sequence<Rest...>>::call(fun, input);
    }
};
}  // namespace detail

template<typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(V&& input, F fun) {
    detail::iterate_helper<F, into_vector_type<V>>::call(fun, into_vector(input));
}

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_ITERATE_H
#ifndef KERNEL_FLOAT_INTERFACE_H
#define KERNEL_FLOAT_INTERFACE_H





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

    template<typename... Args>
    KERNEL_FLOAT_INLINE vector(Args&&... args) : Storage(args...) {}

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

    KERNEL_FLOAT_INLINE
    size_t size() const noexcept {
        return const_size;
    }

    template<typename I>
    KERNEL_FLOAT_INLINE vector_index<Storage, I> operator[](I index) noexcept {
        return {*this, index};
    }

    template<typename I>
    KERNEL_FLOAT_INLINE value_type operator[](I index) const noexcept {
        return this->get(index);
    }

    template<typename U>
    KERNEL_FLOAT_INLINE vector<cast_type<U, Storage>> cast() const noexcept {
        return ::kernel_float::cast<U>(storage());
    }

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
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H



#if KERNEL_FLOAT_FP16_AVAILABLE
#include <cuda_fp16.h>



namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __half)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __half)

struct vector_half2 {
    static_assert(sizeof(__half) * 2 == sizeof(__half2), "invalid size");
    static_assert(alignof(__half) <= alignof(__half2), "invalid alignment");

    KERNEL_FLOAT_INLINE vector_half2(__half v = {}) noexcept : vector_ {v, v} {}
    KERNEL_FLOAT_INLINE vector_half2(__half x, __half y) noexcept : vector_ {x, y} {}
    KERNEL_FLOAT_INLINE vector_half2(__half2 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __half2() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __half get(const_index<0>) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __half get(const_index<1>) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(const_index<0>, __half v) {
        *this = vector_half2(v, get(const_index<1> {}));
    }

    KERNEL_FLOAT_INLINE void set(const_index<1>, __half v) {
        *this = vector_half2(get(const_index<0> {}), v);
    }

    KERNEL_FLOAT_INLINE __half get(size_t index) const {
        if (index == 0) {
            return get(const_index<0> {});
        } else {
            return get(const_index<1> {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __half value) const {
        if (index == 0) {
            set(const_index<0> {}, value);
        } else {
            set(const_index<1> {}, value);
        }
    }

  private:
    __half2 vector_;
};

template<>
struct vector_traits<vector_half2>: default_vector_traits<vector_half2, __half, 2> {};

template<>
struct vector_traits<__half>: vector_traits<vector_scalar<__half>> {};

template<>
struct vector_traits<__half2>: vector_traits<vector_half2> {};

template<>
struct default_vector_storage<__half, 2> {
    using type = vector_half2;
};

#if KERNEL_FLOAT_ON_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                                      \
    namespace ops {                                                                        \
    template<>                                                                             \
    struct NAME<__half> {                                                                  \
        KERNEL_FLOAT_INLINE __half operator()(__half input) {                              \
            return FUN1(input);                                                            \
        }                                                                                  \
    };                                                                                     \
    }                                                                                      \
    namespace detail {                                                                     \
    template<>                                                                             \
    struct map_helper<ops::NAME<__half>, vector_half2, vector_half2> {                     \
        KERNEL_FLOAT_INLINE static __half2 call(ops::NAME<__half>, const __half2& input) { \
            return FUN2(input);                                                            \
        }                                                                                  \
    };                                                                                     \
    }

KERNEL_FLOAT_FP16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_FP16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_FP16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_FP16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_FP16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_FP16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_FP16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_FP16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                               \
    namespace ops {                                                                  \
    template<>                                                                       \
    struct NAME<__half> {                                                            \
        KERNEL_FLOAT_INLINE __half operator()(__half left, __half right) const {     \
            return FUN1(left, right);                                                \
        }                                                                            \
    };                                                                               \
    }                                                                                \
    namespace detail {                                                               \
    template<>                                                                       \
    struct zip_helper<ops::NAME<__half>, vector_half2, vector_half2, vector_half2> { \
        KERNEL_FLOAT_INLINE static __half2                                           \
        call(ops::NAME<__half>, const __half2& left, const __half2& right) {         \
            return FUN2(left, right);                                                \
        }                                                                            \
    };                                                                               \
    }

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_FP16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater_equal, __hge, __hgt2)

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

KERNEL_FLOAT_FP16_CAST(double, __double2half(input), double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float, __float2half(input), __half2float(input));

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

namespace detail {
template<>
struct map_helper<ops::cast<__half, float>, vector_storage<float, 2>, vector_half2> {
    KERNEL_FLOAT_INLINE static vector_storage<float, 2>
    call(ops::cast<__half, float>, __half2 input) noexcept {
        return __half22float2(input);
    }
};

template<>
struct map_helper<ops::cast<float, __half>, vector_half2, vector_storage<float, 2>> {
    KERNEL_FLOAT_INLINE static vector_half2
    call(ops::cast<float, __half>, const vector_storage<float, 2>& input) noexcept {
        return __float22half2_rn(input);
    }
};

}  // namespace detail

using half = __half;
using float16 = __half;
KERNEL_FLOAT_TYPE_ALIAS(half, __half)
KERNEL_FLOAT_TYPE_ALIAS(float16x, __half)
KERNEL_FLOAT_TYPE_ALIAS(f16x, __half)

}  // namespace kernel_float

#endif

#endif  //KERNEL_FLOAT_FP16_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H



#if KERNEL_FLOAT_BF16_AVAILABLE
#include <cuda_bf16.h>



namespace kernel_float {
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float, __nv_bfloat16)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(double, __nv_bfloat16)

struct vector_bfloat16x2 {
    static_assert(sizeof(__nv_bfloat16) * 2 == sizeof(__nv_bfloat162), "invalid size");
    static_assert(alignof(__nv_bfloat16) <= alignof(__nv_bfloat162), "invalid alignment");

    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat16 v = {}) noexcept : vector_ {v, v} {}
    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat16 x, __nv_bfloat16 y) noexcept :
        vector_ {x, y} {}
    KERNEL_FLOAT_INLINE vector_bfloat16x2(__nv_bfloat162 xy) noexcept : vector_ {xy} {}

    KERNEL_FLOAT_INLINE operator __nv_bfloat162() const noexcept {
        return vector_;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(const_index<0>) const {
        return vector_.x;
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(const_index<1>) const {
        return vector_.y;
    }

    KERNEL_FLOAT_INLINE void set(const_index<0>, __nv_bfloat16 v) {
        *this = vector_bfloat16x2(v, get(const_index<1> {}));
    }

    KERNEL_FLOAT_INLINE void set(const_index<1>, __nv_bfloat16 v) {
        *this = vector_bfloat16x2(get(const_index<0> {}), v);
    }

    KERNEL_FLOAT_INLINE __nv_bfloat16 get(size_t index) const {
        if (index == 0) {
            return get(const_index<0> {});
        } else {
            return get(const_index<1> {});
        }
    }

    KERNEL_FLOAT_INLINE void set(size_t index, __nv_bfloat16 value) const {
        if (index == 0) {
            set(const_index<0> {}, value);
        } else {
            set(const_index<1> {}, value);
        }
    }

  private:
    __nv_bfloat162 vector_;
};

template<>
struct vector_traits<vector_bfloat16x2>:
    default_vector_traits<vector_bfloat16x2, __nv_bfloat16, 2> {};

template<>
struct vector_traits<__nv_bfloat16>: vector_traits<vector_scalar<__nv_bfloat16>> {};

template<>
struct vector_traits<__nv_bfloat162>: vector_traits<vector_bfloat16x2> {};

template<>
struct default_vector_storage<__nv_bfloat16, 2> {
    using type = vector_bfloat16x2;
};

#if KERNEL_FLOAT_ON_DEVICE
#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                                   \
    namespace ops {                                                                     \
    template<>                                                                          \
    struct NAME<__nv_bfloat16> {                                                        \
        KERNEL_FLOAT_INLINE __nv_bfloat16 operator()(__nv_bfloat16 input) {             \
            return FUN1(input);                                                         \
        }                                                                               \
    };                                                                                  \
    }                                                                                   \
    namespace detail {                                                                  \
    template<>                                                                          \
    struct map_helper<ops::NAME<__nv_bfloat16>, vector_bfloat16x2, vector_bfloat16x2> { \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                       \
        call(ops::NAME<__nv_bfloat16>, const __nv_bfloat162& input) {                   \
            return FUN2(input);                                                         \
        }                                                                               \
    };                                                                                  \
    }

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2);
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2);
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil);
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos);
KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp);
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10);
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor);
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log);
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2);
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint);
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin);
KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt);
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc);

#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                                            \
    namespace ops {                                                                               \
    template<>                                                                                    \
    struct NAME<__nv_bfloat16> {                                                                  \
        KERNEL_FLOAT_INLINE __nv_bfloat16                                                         \
        operator()(__nv_bfloat16 left, __nv_bfloat16 right) const {                               \
            return FUN1(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }                                                                                             \
    namespace detail {                                                                            \
    template<>                                                                                    \
    struct zip_helper<                                                                            \
        ops::NAME<__nv_bfloat16>,                                                                 \
        vector_bfloat16x2,                                                                        \
        vector_bfloat16x2,                                                                        \
        vector_bfloat16x2> {                                                                      \
        KERNEL_FLOAT_INLINE static __nv_bfloat162                                                 \
        call(ops::NAME<__nv_bfloat16>, const __nv_bfloat162& left, const __nv_bfloat162& right) { \
            return FUN2(left, right);                                                             \
        }                                                                                         \
    };                                                                                            \
    }

KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)

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

KERNEL_FLOAT_BF16_CAST(double, __double2bfloat16(input), double(__bfloat162float(input)));
KERNEL_FLOAT_BF16_CAST(float, __float2bfloat16(input), __bfloat162float(input));

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

namespace detail {
template<>
struct map_helper<ops::cast<__nv_bfloat16, float>, vector_storage<float, 2>, vector_bfloat16x2> {
    KERNEL_FLOAT_INLINE static vector_storage<float, 2>
    call(ops::cast<__nv_bfloat16, float>, __nv_bfloat162 input) noexcept {
        return __bfloat1622float2(input);
    }
};

template<>
struct map_helper<ops::cast<float, __nv_bfloat16>, vector_bfloat16x2, vector_storage<float, 2>> {
    KERNEL_FLOAT_INLINE static vector_bfloat16x2
    call(ops::cast<float, __nv_bfloat16>, const vector_storage<float, 2>& input) noexcept {
        return __float22bfloat162_rn(input);
    }
};
}  // namespace detail

using bfloat16 = __nv_bfloat16;
KERNEL_FLOAT_TYPE_ALIAS(bf16x, __nv_bfloat16)
KERNEL_FLOAT_TYPE_ALIAS(bfloat16x, __nv_bfloat16)

}  // namespace kernel_float

#endif

#if KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE


namespace kernel_float {
KERNEL_FLOAT_BF16_CAST(__half, __float2bfloat16(input), __bfloat162float(input));
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE && KERNEL_FLOAT_BF16_AVAILABLE

#endif  //KERNEL_FLOAT_BF16_H
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H



namespace kernel_float {
namespace detail {
template<typename F, typename V, typename = void>
struct reduce_helper {
    using value_type = vector_value_type<V>;

    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& input) {
        return call(fun, input, make_index_sequence<vector_size<V>> {});
    }

  private:
    template<size_t... Is>
    KERNEL_FLOAT_INLINE static value_type call(F fun, const V& vector, index_sequence<0, Is...>) {
        return call(fun, vector, vector.get(const_index<0> {}), index_sequence<Is...> {});
    }

    template<size_t I, size_t... Rest>
    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<I, Rest...>) {
        return call(
            fun,
            vector,
            fun(accum, vector.get(const_index<I> {})),
            index_sequence<Rest...> {});
    }

    KERNEL_FLOAT_INLINE static value_type
    call(F fun, const V& vector, value_type accum, index_sequence<>) {
        return accum;
    }
};

template<typename F, typename T, size_t N>
struct reduce_helper<F, vector_compound<T, N>> {
    KERNEL_FLOAT_INLINE static T call(F fun, const vector_compound<T, N>& input) {
        static constexpr size_t low_size = vector_compound<T, N>::low_size;
        static constexpr size_t high_size = vector_compound<T, N>::high_size;

        return fun(
            reduce_helper<F, vector_storage<T, low_size>>::call(fun, input.low()),
            reduce_helper<F, vector_storage<T, high_size>>::call(fun, input.high()));
    }
};
}  // namespace detail

template<typename F, typename V>
KERNEL_FLOAT_INLINE vector_value_type<V> reduce(F fun, V&& input) {
    return detail::reduce_helper<F, into_vector_type<V>>::call(
        fun,
        into_vector(std::forward<V>(input)));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T min(V&& input) {
    return reduce(ops::min<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T max(V&& input) {
    return reduce(ops::max<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T sum(V&& input) {
    return reduce(ops::add<T> {}, std::forward<V>(input));
}

template<typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T product(V&& input) {
    return reduce(ops::multiply<T> {}, std::forward<V>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE bool all(V&& input) {
    return reduce(ops::bit_and<bool> {}, cast<bool>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE bool any(V&& input) {
    return reduce(ops::bit_or<bool> {}, cast<bool>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE int count(V&& input) {
    return sum(cast<int>(cast<bool>(input)));
}
}  // namespace kernel_float

#endif  //KERNEL_FLOAT_REDUCE_H
