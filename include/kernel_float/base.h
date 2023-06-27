#ifndef KERNEL_FLOAT_BASE
#define KERNEL_FLOAT_BASE

#include "macros.h"
#include "meta.h"

namespace kernel_float {

template<typename T, size_t N, size_t Alignment = alignof(T)>
struct alignas(Alignment) array {
    KERNEL_FLOAT_INLINE
    T* data() {
        return items_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return items_;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        return items_[i];
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        return items_[i];
    }

    T items_[N];
};

template<typename T, size_t Alignment>
struct array<T, 1, Alignment> {
    KERNEL_FLOAT_INLINE
    array(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    operator T() const {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    T* data() {
        return &value_;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        return &value_;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t) {
        return value_;
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t) const {
        return value_;
    }

    T value_;
};

template<typename T, size_t Alignment>
struct array<T, 0, Alignment> {
    KERNEL_FLOAT_INLINE
    T* data() {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    const T* data() const {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        while (true)
            ;
    }

    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        while (true)
            ;
    }
};

template<size_t N>
using ndindex = array<size_t, N>;

KERNEL_FLOAT_INLINE
static constexpr size_t compute_max_alignment(size_t total_size, size_t min_align) {
    if (total_size % 32 == 0 || min_align >= 32) {
        return 32;
    } else if (total_size % 16 == 0 || min_align == 16) {
        return 16;
    } else if (total_size % 8 == 0 || min_align == 8) {
        return 8;
    } else if (total_size % 4 == 0 || min_align == 4) {
        return 4;
    } else if (total_size % 2 == 0 || min_align == 2) {
        return 2;
    } else {
        return 1;
    }
}

template<typename T, size_t N>
using tensor_storage = array<T, N, compute_max_alignment(sizeof(T) * N, alignof(T))>;

template<typename T, typename D, template<typename, size_t> class S = tensor_storage>
struct tensor;

template<size_t... Ns>
struct extents;

template<>
struct extents<> {
    static constexpr size_t rank = 0;
    static constexpr size_t volume = 1;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<0>) {
        return 0;
    }

    KERNEL_FLOAT_INLINE
    static ndindex<0> unravel_index(size_t i) {
        return {};
    }
};

template<size_t N>
struct extents<N> {
    static constexpr size_t rank = 1;
    static constexpr size_t volume = N;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<1> ind) {
        return ind[0];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<1> unravel_index(size_t i) {
        return {i};
    }
};

template<size_t N, size_t M>
struct extents<N, M> {
    static constexpr size_t rank = 2;
    static constexpr size_t volume = N * M;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : axis == 1 ? M : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return axis == 0 ? M : 1;
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<2> x) {
        return x[0] * M + x[1];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<2> unravel_index(size_t i) {
        return {i / M, i % M};
    }
};

template<size_t N, size_t M, size_t K>
struct extents<N, M, K> {
    static constexpr size_t rank = 3;
    static constexpr size_t volume = N * M * K;

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        return axis == 0 ? N : axis == 1 ? M : axis == 2 ? K : 1;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t stride(size_t axis) {
        return axis == 0 ? M * K  //
            : axis == 1  ? K  //
                         : 1;  //
    }

    KERNEL_FLOAT_INLINE
    static size_t ravel_index(ndindex<3> x) {
        return (x[0] * M + x[1]) * K + x[2];
    }

    KERNEL_FLOAT_INLINE
    static ndindex<3> unravel_index(size_t i) {
        return {i / (K * M), (i / K) % M, i % K};
    }
};

template<typename T>
struct into_tensor_traits {
    using type = tensor<T, extents<>>;

    KERNEL_FLOAT_INLINE
    static type call(const T& input) {
        return tensor_storage<T, 1> {input};
    }
};

template<typename V>
struct into_tensor_traits<const V> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(const V input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<V&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(V& input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<const V&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(const V& input) {
        return into_tensor_traits<V>::call(input);
    }
};

template<typename V>
struct into_tensor_traits<V&&> {
    using type = typename into_tensor_traits<V>::type;

    KERNEL_FLOAT_INLINE
    static type call(V&& input) {
        return into_tensor_traits<V>::call(std::move(input));
    }
};

template<typename T, typename D, template<typename, size_t> class S>
struct into_tensor_traits<tensor<T, D, S>> {
    using type = tensor<T, D>;

    KERNEL_FLOAT_INLINE
    static type call(const tensor<T, D, S>& input) {
        return input;
    }
};

template<typename T, size_t N, size_t A>
struct into_tensor_traits<array<T, N, A>> {
    using type = tensor<T, extents<N>>;

    KERNEL_FLOAT_INLINE
    static type call(const array<T, N, A>& input) {
        return input;
    }
};

template<typename V>
struct tensor_traits;

template<typename T, typename D, template<typename, size_t> class S>
struct tensor_traits<tensor<T, D, S>> {
    using value_type = T;
    using extents_type = D;
    using storage_type = S<T, D::volume>;
};

template<typename V>
using into_tensor_type = typename into_tensor_traits<V>::type;

template<typename V>
using tensor_extents = typename tensor_traits<into_tensor_type<V>>::extents_type;

template<typename V>
static constexpr size_t tensor_rank = tensor_extents<V>::rank;

template<typename V>
static constexpr size_t tensor_volume = tensor_extents<V>::volume;

template<typename V>
using tensor_value_type = typename tensor_traits<into_tensor_type<V>>::value_type;

template<typename V>
using tensor_storage_type = tensor_storage<tensor_value_type<V>, tensor_volume<V>>;

template<typename... Vs>
using promoted_tensor_value_type =
    promote_t<typename tensor_traits<into_tensor_type<Vs>>::value_type...>;

template<typename V>
KERNEL_FLOAT_INLINE into_tensor_type<V> into_tensor(V&& input) {
    return into_tensor_traits<V>::call(std::forward<V>(input));
}

template<typename V>
KERNEL_FLOAT_INLINE tensor_storage_type<V> into_tensor_storage(V&& input) {
    return into_tensor_traits<V>::call(std::forward<V>(input)).storage();
}

}  // namespace kernel_float

#endif