#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

#include "storage.h"

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
