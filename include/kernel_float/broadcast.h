#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H

#include "base.h"

namespace kernel_float {
namespace detail {

template<size_t N, size_t M>
struct unify_dimension_helper;

template<>
struct unify_dimension_helper<1, 1> {
    static constexpr size_t value = 1;
};

template<size_t N>
struct unify_dimension_helper<N, N> {
    static constexpr size_t value = N;
};

template<size_t N>
struct unify_dimension_helper<N, 1> {
    static constexpr size_t value = N;
};

template<size_t N>
struct unify_dimension_helper<1, N> {
    static constexpr size_t value = N;
};

template<typename A, typename B>
struct unify_extents_helper;

template<size_t... Ns, size_t... Ms>
struct unify_extents_helper<extents<Ns...>, extents<Ms...>> {
    using type = extents<unify_dimension_helper<Ns, Ms>::value...>;
};

template<typename E, size_t N, typename = void>
struct extents_to_rank {
    using type = E;
};

template<size_t... Ns, size_t N>
struct extents_to_rank<extents<Ns...>, N, enabled_t<(sizeof...(Ns) < N)>>:
    extents_to_rank<extents<1, Ns...>, N> {};

template<typename A, typename B>
struct broadcast_extents_helper {
    using type = typename unify_extents_helper<
        typename extents_to_rank<A, B::rank>::type,  //
        typename extents_to_rank<B, A::rank>::type  //
        >::type;
};

template<typename E>
struct broadcast_extents_helper<E, E> {
    using type = E;
};

}  // namespace detail

template<typename A, typename B>
using broadcast_extents = typename detail::broadcast_extents_helper<A, B>::type;

template<typename A, typename B>
using broadcast_tensor_extents = broadcast_extents<tensor_extents<A>, tensor_extents<B>>;

template<typename From, typename To>
static constexpr bool is_broadcastable = is_same<broadcast_extents<From, To>, To>;

template<typename V, typename To>
static constexpr bool is_tensor_broadcastable = is_broadcastable<tensor_extents<V>, To>;

namespace detail {

template<typename E, typename IS, typename OS>
struct copy_helper;

template<typename IS, typename OS>
struct copy_helper<extents<>, IS, OS> {
    template<typename T>
    static void call(T* output, const T* input) {
        ndindex<0> x;
        size_t input_index = IS::call(x);
        size_t output_index = OS::call(x);
        output[output_index] = input[input_index];
    }
};

template<size_t N, typename IS, typename OS>
struct copy_helper<extents<N>, IS, OS> {
    template<typename T>
    static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            ndindex<1> x = {i};
            size_t input_index = IS::call(x);
            size_t output_index = OS::call(x);
            output[output_index] = input[input_index];
        }
    }
};

template<size_t N, size_t M, typename IS, typename OS>
struct copy_helper<extents<N, M>, IS, OS> {
    template<typename T>
    static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                ndindex<2> x = {i, j};
                size_t input_index = IS::call(x);
                size_t output_index = OS::call(x);
                output[output_index] = input[input_index];
            }
        }
    }
};

template<size_t N, size_t M, size_t K, typename IS, typename OS>
struct copy_helper<extents<N, M, K>, IS, OS> {
    template<typename T>
    static void call(T* output, const T* input) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                for (size_t k = 0; k < K; k++) {
                    ndindex<3> x = {i, j, k};
                    size_t input_index = IS::call(x);
                    size_t output_index = OS::call(x);
                    output[output_index] = input[input_index];
                }
            }
        }
    }
};

template<typename E>
struct strides_helper;

template<>
struct strides_helper<extents<>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<0>) {
        return 0;
    }
};

template<size_t N>
struct strides_helper<extents<N>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<1> x) {
        return (N != 1 ? x[0] : 0);
    }
};

template<size_t N, size_t M>
struct strides_helper<extents<N, M>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<2> x) {
        return (N != 1 ? x[0] * M : 0) +  //
            (M != 1 ? x[1] : 0);
    }
};

template<size_t N, size_t M, size_t K>
struct strides_helper<extents<N, M, K>> {
    KERNEL_FLOAT_INLINE
    static size_t call(ndindex<3> x) {
        return (N != 1 ? x[0] * M * K : 0) +  //
            (M != 1 ? x[1] * K : 0) +  //
            (K != 1 ? x[2] : 0);
    }
};

template<typename T, typename From, typename To>
struct broadcast_helper {
    KERNEL_FLOAT_INLINE static tensor_storage<T, From::volume>
    call(tensor_storage<T, To::volume> input) {
        static_assert(is_broadcastable<From, To>, "cannot broadcast to required shape");
        using IS = strides_helper<extents_to_rank<From, To::rank>>;
        using OS = strides_helper<To>;

        tensor_storage<T, From::volume> output;
        copy_helper<To, IS, OS>::call(output.data(), input.data());
        return output;
    }
};

template<typename T, typename E>
struct broadcast_helper<T, E, E> {
    KERNEL_FLOAT_INLINE static tensor_storage<T, E::volume>
    call(tensor_storage<T, E::volume> input) {
        return input;
    }
};

}  // namespace detail

template<size_t... Ns, typename V>
tensor<tensor_value_type<V>, extents<Ns...>>
broadcast(const V& input, extents<Ns...> new_extents = {}) {
    using T = tensor_value_type<V>;
    return detail::broadcast_helper<T, tensor_extents<V>, extents<Ns...>>::call(
        into_tensor(input).storage());
}

template<size_t... Ns, typename T>
tensor<T, extents<Ns...>> fill(T value = {}, extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {value};
    return detail::broadcast_helper<T, extents<>, extents<Ns...>>::call(input);
}

template<typename T, size_t... Ns>
tensor<T, extents<Ns...>> zeros(extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {T {}};
    return detail::broadcast_helper<T, extents<>, extents<Ns...>>::call(input);
}

template<typename T, size_t... Ns>
tensor<T, extents<Ns...>> ones(extents<Ns...> = {}) {
    tensor_storage<T, 1> input = {T {1}};
    return detail::broadcast_helper<T, extents<>, extents<Ns...>>::call(input);
}

template<typename V, typename T = tensor_value_type<V>, typename E = tensor_extents<V>>
tensor<T, E> zeros_like(const V&) {
    return zeros<T>(E {});
}

template<typename V, typename T = tensor_value_type<V>, typename E = tensor_extents<V>>
tensor<T, E> ones_like(const V&) {
    return ones<T>(E {});
}

}  // namespace kernel_float

#endif