#ifndef KERNEL_FLOAT_BINARY_H
#define KERNEL_FLOAT_BINARY_H

#include "storage.h"
#include "unops.h"

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
