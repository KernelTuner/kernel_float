#ifndef KERNEL_FLOAT_MEMORY_H
#define KERNEL_FLOAT_MEMORY_H

#include "binops.h"
#include "conversion.h"
#include "iterate.h"

namespace kernel_float {

namespace detail {
template<typename T, size_t N, typename Is = make_index_sequence<N>>
struct load_impl;

template<typename T, size_t N, size_t... Is>
struct load_impl<T, N, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const T* input, const size_t* offsets) {
        return {input[offsets[Is]]...};
    }

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const T* input, const size_t* offsets, const bool* mask) {
        bool all_valid = true;
        for (size_t i = 0; i < N; i++) {
            all_valid &= mask[i];
        }

        if (all_valid) {
            return {input[offsets[Is]]...};
        } else {
            return {(mask[Is] ? input[offsets[Is]] : T())...};
        }
    }
};
}  // namespace detail

/**
 * Load the elements from the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * ```
 * // Load 4 elements at data[0], data[2], data[4], data[8]
 * vec<T, 4> values = load(data, make_vec(0, 2, 4, 8));
 * ```
 */
template<typename T, typename I>
KERNEL_FLOAT_INLINE vector<T, vector_extent_type<I>> load(const T* ptr, const I& indices) {
    return detail::load_impl<T, vector_extent<I>>::call(ptr, cast<size_t>(indices).data());
}

/**
 * Load the elements from the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be loaded and ``false`` indicates that the value should be skipped. This can be used
 * to prevent reading out of bounds.
 *
 * ```
 * // Load 2 elements at data[0] and data[8], skip data[2] and data[4]
 * vec<T, 4> values = = load(data, make_vec(0, 2, 4, 8), make_vec(true, false, false, true));
 * ```
 */
template<typename T, typename I, typename M, typename E = broadcast_vector_extent_type<I, M>>
KERNEL_FLOAT_INLINE vector<T, E> load(const T* ptr, const I& indices, const M& mask) {
    static constexpr E new_size = {};

    return detail::load_impl<T, E::value>::call(
        ptr,
        convert_storage<size_t>(indices, new_size).data(),
        convert_storage<bool>(mask, new_size).data());
}

/**
 * Load ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``. Optionally, an
 * ``offset`` can be given that shifts all the indices by a fixed amount.
 *
 * ```
 * // Load 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<T, 4> values = loadn<4>(data);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * vec<T, 4> values2 = loadn<4>(data, 10);
 * ```
 */
template<size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(const T* ptr, size_t offset = 0) {
    return load(ptr, offset + range<size_t, N>());
}

/**
 * Load ``N`` elements at the location ``ptr[offset+0], ptr[offset+1], ptr[offset+2], ...``.
 * Locations for which the index equals or exceeds ``max_length`` are ignored. This can be used
 * to prevent reading out of bounds.
 *
 * ```
 * // Returns {ptr[8], ptr[9], 0, 0};
 * vec<T, 4> values = loadn<4>(data, 8, 10);
 * ```
 */
template<size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(const T* ptr, size_t offset, size_t max_length) {
    auto indices = offset + range<size_t, N>();
    return load(ptr, indices, indices < max_length);
}

namespace detail {
template<typename T, size_t N, typename Is = make_index_sequence<N>>
struct store_impl;

template<typename T, size_t N, size_t... Is>
struct store_impl<T, N, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE
    static void call(T* outputs, const T* inputs, const size_t* offsets) {
        ((outputs[offsets[Is]] = inputs[Is]), ...);
    }

    KERNEL_FLOAT_INLINE
    static void call(T* outputs, const T* inputs, const size_t* offsets, const bool* mask) {
        bool all_valid = true;
        for (size_t i = 0; i < N; i++) {
            all_valid &= mask[i];
        }

        if (all_valid) {
            ((outputs[offsets[Is]] = inputs[Is]), ...);
        } else {
#pragma unroll
            for (size_t i = 0; i < N; i++) {
                if (mask[i]) {
                    outputs[offsets[i]] = inputs[i];
                }
            }
        }
    }
};
}  // namespace detail

/**
 * Load the elements from the vector `values` in the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * ```
 * // Store 4 elements at data[0], data[2], data[4], data[8]
 * auto values = make_vec(42, 13, 87, 12);
 * store(values, data, make_vec(0, 2, 4, 8));
 * ```
 */
template<
    typename T,
    typename V,
    typename I,
    typename M,
    typename E = broadcast_vector_extent_type<V, I, M>>
KERNEL_FLOAT_INLINE void store(const V& values, T* ptr, const I& indices, const M& mask) {
    return detail::store_impl<T, E::value>::call(
        ptr,
        convert_storage<T>(values, E()).data(),
        convert_storage<size_t>(indices, E()).data(),
        convert_storage<bool>(mask, E()).data());
}

/**
 * Load the elements from the vector `values` in the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be store and ``false`` indicates that the value should be skipped. This can be used
 * to prevent writing out of bounds.
 *
 * ```
 * // Store 2 elements at data[0] and data[8], skip data[2] and data[4]
 * auto values = make_vec(42, 13, 87, 12);
 * auto mask = make_vec(true, false, false, true);
 * store(values, data, make_vec(0, 2, 4, 8), mask);
 * ```
 */
template<typename T, typename V, typename I, typename E = broadcast_vector_extent_type<V, I>>
KERNEL_FLOAT_INLINE void store(const V& values, T* ptr, const I& indices) {
    return detail::store_impl<T, E::value>::call(
        ptr,
        convert_storage<T>(values, E()).data(),
        convert_storage<size_t>(indices, E()).data());
}

/**
 * Store ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``. Optionally, an
 * ``offset`` can be given that shifts all the indices by a fixed amount.
 *
 * ```
 * // Store 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * storen<4>(values, data);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * storen<4>(values, data, 10);
 * ```
 */
template<typename T, typename V, size_t N = vector_extent<V>>
KERNEL_FLOAT_INLINE void storen(const V& values, T* ptr, size_t offset = 0) {
    auto indices = offset + range<size_t, N>();
    return store(values, ptr, indices);
}

/**
 * Store ``N`` elements at the location ``ptr[offset+0], ptr[offset+1], ptr[offset+2], ...``.
 * Locations for which the index equals or exceeds ``max_length`` are ignored. This can be used
 * to prevent reading out of bounds.
 *
 * ```
 * // Store 1.0f at data[8] and 2.0f at data[9]. Ignores remaining values.
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * storen<4>(values, data, 8, 10);
 * ```
 */
template<typename T, typename V, size_t N = vector_extent<V>>
KERNEL_FLOAT_INLINE void storen(const V& values, T* ptr, size_t offset, size_t max_length) {
    auto indices = offset + range<size_t, N>();
    return store(values, ptr, indices, indices < max_length);
}

/**
 * Returns the original pointer ``ptr`` and hints to the compiler that this pointer is aligned to ``alignment`` bytes.
 * If this is not actually the case, compiler optimizations will break things and generate invalid code. Be careful!
 */
template<typename T>
KERNEL_FLOAT_INLINE T* unsafe_assume_aligned(T* ptr, size_t alignment) {
// TOOD: check if this way is support across all compilers
#if defined(__has_builtin) && __has_builtin(__builtin_assume_aligned)
    return static_cast<T*>(__builtin_assume_aligned(ptr, alignment));
#else
    return ptr;
#endif
}

/**
 * Represents a pointer of type ``T*`` that is guaranteed to be aligned to ``alignment`` bytes.
 */
template<typename T, size_t alignment = 256>
struct aligned_ptr {
    static_assert(alignment >= alignof(T), "invalid alignment");

    KERNEL_FLOAT_INLINE
    aligned_ptr(nullptr_t = nullptr) {}

    KERNEL_FLOAT_INLINE
    explicit aligned_ptr(T* ptr) : ptr_(ptr) {}

    /**
     * Return the pointer value.
     */
    KERNEL_FLOAT_INLINE
    T* get() const {
        return unsafe_assume_aligned(ptr_, alignment);
    }

    KERNEL_FLOAT_INLINE
    operator T*() const {
        return get();
    }

    template<typename I>
    KERNEL_FLOAT_INLINE T& operator[](I&& index) const {
        return get()[std::forward<I>(index)];
    }

    /**
     * See ``kernel_float::load``
     */
    template<typename I, typename M, typename E = broadcast_vector_extent_type<I, M>>
    KERNEL_FLOAT_INLINE vector<T, E> load(const I& indices, const M& mask = true) const {
        return ::kernel_float::load(get(), indices, mask);
    }

    /**
     * See ``kernel_float::loadn``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(size_t offset = 0) const {
        return ::kernel_float::loadn<N>(get(), offset);
    }

    /**
     * See ``kernel_float::loadn``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(size_t offset, size_t max_length) const {
        return ::kernel_float::loadn<N>(get(), offset, max_length);
    }

    /**
     * See ``kernel_float::store``
     */
    template<typename V, typename I, typename M, typename E = broadcast_vector_extent_type<V, I, M>>
    KERNEL_FLOAT_INLINE void store(const V& values, const I& indices, const M& mask = true) const {
        ::kernel_float::store(values, get(), indices, mask);
    }
    /**
     * See ``kernel_float::storen``
     */
    template<typename V, size_t N = vector_extent<V>>
    KERNEL_FLOAT_INLINE void storen(const V& values, size_t offset = 0) const {
        ::kernel_float::storen(values, get(), offset);
    }
    /**
     * See ``kernel_float::storen``
     */
    template<typename V, size_t N = vector_extent<V>>
    KERNEL_FLOAT_INLINE void storen(const V& values, size_t offset, size_t max_length) const {
        ::kernel_float::storen(values, get(), offset, max_length);
    }

  private:
    T* ptr_ = nullptr;
};

/**
 * Represents a pointer of type ``const T*`` that is guaranteed to be aligned to ``alignment`` bytes.
 */
template<typename T, size_t alignment>
struct aligned_ptr<const T, alignment> {
    static_assert(alignment >= alignof(T), "invalid alignment");

    KERNEL_FLOAT_INLINE
    aligned_ptr(nullptr_t = nullptr) {}

    KERNEL_FLOAT_INLINE
    explicit aligned_ptr(T* ptr) : ptr_(ptr) {}

    KERNEL_FLOAT_INLINE
    explicit aligned_ptr(const T* ptr) : ptr_(ptr) {}

    KERNEL_FLOAT_INLINE
    aligned_ptr(const aligned_ptr<T>& ptr) : ptr_(ptr.get()) {}

    KERNEL_FLOAT_INLINE
    aligned_ptr(const aligned_ptr<const T>& ptr) : ptr_(ptr.get()) {}

    /**
     * Return the pointer value.
     */
    KERNEL_FLOAT_INLINE
    const T* get() const {
        return unsafe_assume_aligned(ptr_, alignment);
    }

    KERNEL_FLOAT_INLINE
    operator const T*() const {
        return get();
    }

    template<typename I>
    KERNEL_FLOAT_INLINE const T& operator[](I&& index) const {
        return get()[std::forward<I>(index)];
    }

    /**
     * See ``kernel_float::load``
     */
    template<typename I, typename M, typename E = broadcast_vector_extent_type<I, M>>
    KERNEL_FLOAT_INLINE vector<T, E> load(const I& indices, const M& mask = true) const {
        return ::kernel_float::load(get(), indices, mask);
    }

    /**
     * See ``kernel_float::loadn``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(size_t offset = 0) const {
        return ::kernel_float::loadn<N>(get(), offset);
    }

    /**
     * See ``kernel_float::loadn``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> loadn(size_t offset, size_t max_length) const {
        return ::kernel_float::loadn<N>(get(), offset, max_length);
    }

  private:
    const T* ptr_ = nullptr;
};

template<typename T>
aligned_ptr(T*) -> aligned_ptr<T>;

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_MEMORY_H
