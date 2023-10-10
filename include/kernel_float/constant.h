#ifndef KERNEL_FLOAT_CONSTANT
#define KERNEL_FLOAT_CONSTANT

#include "base.h"
#include "conversion.h"

namespace kernel_float {

/**
 * `constant<T>` represents a constant value of type `T`.
 *
 * The object has the property that for any binary operation involving
 * a `constant<T>` and a value of type `U`, the constant is automatically
 * cast to also be of type `U`.
 *
 * For example:
 * ```
 * float a = 5;
 * constant<double> b = 3;
 *
 * auto c = a + b; // The result will be of type `float`
 * ```
 */
template<typename T = double>
struct constant {
    /**
     * Create a new constant from the given value.
     */
    KERNEL_FLOAT_INLINE
    constexpr constant(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    constexpr constant(const constant<T>& that) : value_(that.value) {}

    /**
     * Create a new constant from another constant of type `R`.
     */
    template<typename R>
    KERNEL_FLOAT_INLINE explicit constexpr constant(const constant<R>& that) {
        auto f = ops::cast<R, T>();
        value_ = f(that.get());
    }

    /**
     * Return the value of the constant
     */
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

// Deduction guide for `constant<T>`
#if defined(__cpp_deduction_guides)
template<typename T>
constant(T&&) -> constant<decay_t<T>>;
#endif

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

#define KERNEL_FLOAT_CONSTANT_DEFINE_OP(OP)                                                    \
    template<typename L, typename R>                                                           \
    KERNEL_FLOAT_INLINE auto operator OP(const constant<L>& left, const R& right) {            \
        auto f = ops::cast<L, vector_value_type<R>>();                                         \
        return f(left.get()) OP right;                                                         \
    }                                                                                          \
                                                                                               \
    template<typename L, typename R>                                                           \
    KERNEL_FLOAT_INLINE auto operator OP(const L& left, const constant<R>& right) {            \
        auto f = ops::cast<R, vector_value_type<L>>();                                         \
        return left OP f(right.get());                                                         \
    }                                                                                          \
                                                                                               \
    template<typename L, typename R, typename E>                                               \
    KERNEL_FLOAT_INLINE auto operator OP(const constant<L>& left, const vector<R, E>& right) { \
        auto f = ops::cast<L, R>();                                                            \
        return f(left.get()) OP right;                                                         \
    }                                                                                          \
                                                                                               \
    template<typename L, typename R, typename E>                                               \
    KERNEL_FLOAT_INLINE auto operator OP(const vector<L, E>& left, const constant<R>& right) { \
        auto f = ops::cast<R, L>();                                                            \
        return left OP f(right.get());                                                         \
    }                                                                                          \
                                                                                               \
    template<typename L, typename R, typename T = promote_t<L, R>>                             \
    KERNEL_FLOAT_INLINE constant<T> operator OP(                                               \
        const constant<L>& left,                                                               \
        const constant<R>& right) {                                                            \
        return constant<T>(left.get()) OP constant<T>(right.get());                            \
    }

KERNEL_FLOAT_CONSTANT_DEFINE_OP(+)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(-)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(*)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(/)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(%)

}  // namespace kernel_float

#endif
