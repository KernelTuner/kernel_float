#ifndef KERNEL_FLOAT_CONSTANT
#define KERNEL_FLOAT_CONSTANT

#include "broadcast.h"

namespace kernel_float {

template<typename T = double>
struct constant {
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
    using type = typename promote_type<L, R>::type;
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
template<typename T, typename R, RoundingMode m>
struct cast<constant<T>, R, m> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept {
        return cast<T, R, m> {}(input);
    }
};
}  // namespace ops

}  // namespace kernel_float

#endif