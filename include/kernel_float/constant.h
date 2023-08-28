#ifndef KERNEL_FLOAT_CONSTANT
#define KERNEL_FLOAT_CONSTANT

#include "base.h"
#include "conversion.h"

namespace kernel_float {

template<typename T = double>
struct constant {
    template<typename R>
    KERNEL_FLOAT_INLINE explicit constexpr constant(const constant<R>& that) : value_(that.get()) {}

    KERNEL_FLOAT_INLINE
    constexpr constant(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    constexpr T get() const {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    constexpr operator T() const {
        return value_;
    }

  private:
    T value_;
};

template<typename T = double>
KERNEL_FLOAT_INLINE constexpr constant<T> make_constant(T value) {
    return value;
}

template<typename L, typename R>
struct promote_type<constant<L>, constant<R>> {
    using type = constant<typename promote_type<L, R>::type>;
};

template<typename L, typename R>
struct promote_type<constant<L>, R> {
    using type = typename promote_type<L, R>::type;
};

template<typename L, typename R>
struct promote_type<L, constant<R>> {
    using type = typename promote_type<L, R>::type;
};

namespace ops {
template<typename T, typename R>
struct cast<constant<T>, R> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R> {}(input);
    }
};

template<typename T, typename R, RoundingMode m>
struct cast<constant<T>, R, m> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R, m> {}(input);
    }
};
}  // namespace ops

#define KERNEL_FLOAT_CONSTANT_DEFINE_OP(OP)                                      \
    template<typename L, typename R>                                             \
    R operator OP(const constant<L>& left, const R& right) {                     \
        using T = vector_value_type<R>;                                          \
        return operator OP(T(left.get()), right);                                \
    }                                                                            \
                                                                                 \
    template<typename L, typename R>                                             \
    L operator OP(const L& left, const constant<R>& right) {                     \
        using T = vector_value_type<L>;                                          \
        return operator OP(left, T(right.get()));                                \
    }                                                                            \
                                                                                 \
    template<typename L, typename R, typename T = promote_t<L, R>>               \
    constant<T> operator OP(const constant<L>& left, const constant<R>& right) { \
        return constant<T>(operator OP(T(left.get()), T(right.get())));          \
    }

//KERNEL_FLOAT_CONSTANT_DEFINE_OP(+)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(-)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(*)
//KERNEL_FLOAT_CONSTANT_DEFINE_OP(/)

}  // namespace kernel_float

#endif
