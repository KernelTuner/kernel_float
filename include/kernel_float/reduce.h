#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H

#include "storage.h"

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
