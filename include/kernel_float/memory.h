#ifndef KERNEL_FLOAT_MEMORY_H
#define KERNEL_FLOAT_MEMORY_H

#include "binops.h"
#include "conversion.h"
#include "iterate.h"

namespace kernel_float {
namespace detail {
template<typename T, size_t N, typename Is = make_index_sequence<N>>
struct copy_impl;

template<typename T, size_t N, size_t... Is>
struct copy_impl<T, N, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> load(const T* input, const size_t* offsets, const bool* mask) {
        return {(mask[Is] ? input[offsets[Is]] : T {})...};
    }

    KERNEL_FLOAT_INLINE
    static void store(T* outputs, const T* inputs, const size_t* offsets, const bool* mask) {
        ((mask[Is] ? outputs[offsets[Is]] = inputs[Is] : T {}), ...);
    }
};
}  // namespace detail

/**
 * Load the elements from the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be loaded and ``false`` indicates that the value should be skipped. This can be used
 * to prevent reading out of bounds.
 *
 * ```
 * // Load 2 elements at data[0] and data[8], skip data[2] and data[4]
 * vec<T, 4> values = = read(data, make_vec(0, 2, 4, 8), make_vec(true, false, false, true));
 * ```
 */
template<typename T, typename I, typename M = bool, typename E = broadcast_vector_extent_type<I, M>>
KERNEL_FLOAT_INLINE vector<T, E> read(const T* ptr, const I& indices, const M& mask = true) {
    return detail::copy_impl<T, E::value>::load(
        ptr,
        convert_storage<size_t>(indices, E()).data(),
        convert_storage<bool>(mask, E()).data());
}

/**
 * Store the elements from the vector `values` in the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be store and ``false`` indicates that the value should be skipped. This can be used
 * to prevent writing out of bounds.
 *
 * ```
 * // Store 2 elements at data[0] and data[8], skip data[2] and data[4]
 * auto values = make_vec(42, 13, 87, 12);
 * auto mask = make_vec(true, false, false, true);
 * write(data, make_vec(0, 2, 4, 8), values, mask);
 * ```
 */
template<
    typename T,
    typename V,
    typename I,
    typename M = bool,
    typename E = broadcast_vector_extent_type<V, I, M>>
KERNEL_FLOAT_INLINE void write(T* ptr, const I& indices, const V& values, const M& mask = true) {
    return detail::copy_impl<T, E::value>::store(
        ptr,
        convert_storage<T>(values, E()).data(),
        convert_storage<size_t>(indices, E()).data(),
        convert_storage<bool>(mask, E()).data());
}

/**
 * Load ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * ```
 * // Load 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<T, 4> values = read<4>(data);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * vec<T, 4> values = read<4>(values + 10, data);
 * ```
 */
template<size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> read(const T* ptr) {
    return read(ptr, range<size_t, N>());
}

/**
 * Store ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * ```
 * // Store 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * write(data, values);
 *
 * // Store 4 elements at locations data[10], data[11], data[12], data[13]
 * write(data + 10, values);
 * ```
 */
template<typename V, typename T>
KERNEL_FLOAT_INLINE void write(T* ptr, const V& values) {
    static constexpr size_t N = vector_extent<V>;
    write(ptr, range<size_t, N>(), values);
}

namespace detail {
KERNEL_FLOAT_INLINE
constexpr size_t gcd(size_t a, size_t b) {
    return b == 0 ? a : gcd(b, a % b);
}

template<typename T, size_t N, size_t alignment, typename = void>
struct copy_aligned_impl {
    static constexpr size_t K = N > 8 ? 8 : (N > 4 ? 4 : (N > 2 ? 2 : 1));
    static constexpr size_t alignment_K = gcd(alignment, sizeof(T) * K);

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        copy_aligned_impl<T, K, alignment>::load(output, input);
        copy_aligned_impl<T, N - K, alignment_K>::load(output + K, input + K);
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        copy_aligned_impl<T, K, alignment>::store(output, input);
        copy_aligned_impl<T, N - K, alignment_K>::store(output + K, input + K);
    }
};

template<typename T, size_t alignment>
struct copy_aligned_impl<T, 0, alignment> {
    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {}

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {}
};

template<typename T, size_t alignment>
struct copy_aligned_impl<T, 1, alignment> {
    using storage_type = T;

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        output[0] = input[0];
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        output[0] = input[0];
    }
};

template<typename T, size_t alignment>
struct copy_aligned_impl<T, 2, alignment, enable_if_t<(alignment > sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(alignment, 2 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0] = storage.v0;
        output[1] = storage.v1;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        *reinterpret_cast<storage_type*>(output) = storage_type {input[0], input[1]};
    }
};

template<typename T, size_t alignment>
struct copy_aligned_impl<T, 4, alignment, enable_if_t<(alignment > 2 * sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(alignment, 4 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1, v2, v3;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0] = storage.v0;
        output[1] = storage.v1;
        output[2] = storage.v2;
        output[3] = storage.v3;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        *reinterpret_cast<storage_type*>(output) = storage_type {
            input[0],  //
            input[1],
            input[2],
            input[3]};
    }
};

template<typename T, size_t alignment>
struct copy_aligned_impl<T, 8, alignment, enable_if_t<(alignment > 4 * sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(alignment, 8 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1, v2, v3, v4, v5, v6, v7;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0] = storage.v0;
        output[1] = storage.v1;
        output[2] = storage.v2;
        output[3] = storage.v3;
        output[4] = storage.v4;
        output[5] = storage.v5;
        output[6] = storage.v6;
        output[7] = storage.v7;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        *reinterpret_cast<storage_type*>(output) = storage_type {
            input[0],  //
            input[1],
            input[2],
            input[3],
            input[4],
            input[5],
            input[6],
            input[7]};
    }
};

}  // namespace detail

/**
 * Load ``N`` elements at the locations ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * It is assumed that ``ptr`` is maximum aligned such that all ``N`` elements can be loaded at once using a vector
 * operation. If the pointer is not aligned, undefined behavior will occur.
 *
 * ```
 * // Load 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<T, 4> values = read_aligned<4>(data);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * vec<T, 4> values2 = read_aligned<4>(data + 10);
 * ```
 */
template<size_t Align, size_t N = Align, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> read_aligned(const T* ptr) {
    static constexpr size_t alignment = detail::gcd(Align * sizeof(T), KERNEL_FLOAT_MAX_ALIGNMENT);
    vector_storage<T, N> result;
    detail::copy_aligned_impl<T, N, alignment>::load(
        result.data(),
        KERNEL_FLOAT_ASSUME_ALIGNED(const T, ptr, alignment));
    return result;
}

/**
 * Store ``N`` elements at the locations ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * It is assumed that ``ptr`` is maximum aligned such that all ``N`` elements can be loaded at once using a vector
 * operation. If the pointer is not aligned, undefined behavior will occur.
 *
 * ```
 * // Store 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * write_aligned(data, values);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * write_aligned(data + 10, values);
 * ```
 */
template<size_t Align, typename V, typename T>
KERNEL_FLOAT_INLINE void write_aligned(T* ptr, const V& values) {
    static constexpr size_t N = vector_extent<V>;
    static constexpr size_t alignment = detail::gcd(Align * sizeof(T), KERNEL_FLOAT_MAX_ALIGNMENT);

    return detail::copy_aligned_impl<T, N, alignment>::store(
        KERNEL_FLOAT_ASSUME_ALIGNED(T, ptr, alignment),
        convert_storage<T, N>(values).data());
}

/**
 * Represents a pointer of type ``T*`` that is guaranteed to be aligned to ``alignment`` bytes.
 */
template<typename T, size_t alignment = KERNEL_FLOAT_MAX_ALIGNMENT>
struct aligned_ptr {
    static_assert(alignment >= alignof(T), "invalid alignment");

    KERNEL_FLOAT_INLINE
    aligned_ptr(nullptr_t = nullptr) {}

    KERNEL_FLOAT_INLINE
    explicit aligned_ptr(T* ptr) : ptr_(ptr) {}

    KERNEL_FLOAT_INLINE
    aligned_ptr(const aligned_ptr<T>& ptr) : ptr_(ptr.get()) {}

    /**
     * Return the pointer value.
     */
    KERNEL_FLOAT_INLINE
    T* get() const {
        return KERNEL_FLOAT_ASSUME_ALIGNED(T, ptr_, alignment);
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
     * Returns a new pointer that is offset by ``Step * offset`` items
     *
     * Example
     * =======
     * ```
     * aligned_ptr<int> ptr = vector.data();
     *
     * // Returns a pointer to element `vector[8]` with alignment of 4
     * ptr.offset(8);
     *
     * // Returns a pointer to element `vector[4]` with alignment of 16
     * ptr.offset<4>();
     *
     * // Returns a pointer to element `vector[8]` with alignment of 16
     * ptr.offset<4>(2);
     * ```
     */
    template<size_t Step = 1>
    KERNEL_FLOAT_INLINE aligned_ptr<T, detail::gcd(sizeof(T) * Step, alignment)>
    offset(size_t n = 1) const {
        return aligned_ptr<T, detail::gcd(sizeof(T) * Step, alignment)> {get() + Step * n};
    }

    /**
     * See ``kernel_float::read``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> read() const {
        vector_storage<T, N> result;
        detail::copy_aligned_impl<T, N, alignment>::load(result.data(), get());
        return result;
    }

    /**
     * See ``kernel_float::write``
     */
    template<typename V>
    KERNEL_FLOAT_INLINE void write(const V& values) const {
        constexpr size_t N = vector_extent<V>;
        return detail::copy_aligned_impl<T, N, alignment>::store(
            get(),
            convert_storage<T, N>(values).data());
    }

    /**
     * See ``kernel_float::read``
     */
    template<typename I, typename M = bool, typename E = broadcast_vector_extent_type<I, M>>
    KERNEL_FLOAT_INLINE vector<T, E> read(const I& indices, const M& mask = true) {
        return ::kernel_float::read(get(), indices, mask);
    }

    /**
     * See ``kernel_float::write``
     */
    template<typename V, typename I, typename M = bool>
    KERNEL_FLOAT_INLINE void write(const I& indices, const V& values, const M& mask = true) {
        return ::kernel_float::write(get(), indices, values, mask);
    }

    /**
     * Offsets the pointer by `Step * offset` items and then read the subsequent `N` items.
     *
     * Example
     * =======
     * ```
     * aligned_ptr<int> ptr = vector.data();
     *
     * // Returns vector[40], vector[41], vector[42], vector[43]
     * ptr.read_at<4>(10);
     *
     * // Returns vector[20], vector[21], vector[22]
     * ptr.read_at<2, 3>(10);
     * ```
     */
    template<size_t Step = 1, size_t N = Step>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> read_at(size_t offset) const {
        return this->offset<Step>(offset).template read<N>();
    }

    /**
     * Offsets the pointer by `Step * offset` items and then writes the subsequent `N` items.
     *
     * Example
     * =======
     * ```
     * aligned_ptr<int> ptr = vector.data();
     * vec<int, 4> values = {1, 2, 3, 4};
     *
     * // Writes to vector[40], vector[41], vector[42], vector[43]
     * ptr.write_at<4>(10, values);
     *
     * // Returns vector[20], vector[21], vector[22], vector[23]
     * ptr.write_at<2>(10, values);
     * ```
     */
    template<size_t Step = 1, typename V>
    KERNEL_FLOAT_INLINE void write_at(size_t offset, const V& values) const {
        return this->offset<Step>(offset).template write(values);
    }

  private:
    T* ptr_ = nullptr;
};

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
        return KERNEL_FLOAT_ASSUME_ALIGNED(const T, ptr_, alignment);
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
     * Returns a new pointer that is offset by ``Step * offset`` items
     *
     * Example
     * =======
     * ```
     * aligned_ptr<int> ptr = vector.data();
     *
     * // Returns a pointer to element `vector[8]` with alignment of 4
     * ptr.offset(8);
     *
     * // Returns a pointer to element `vector[4]` with alignment of 16
     * ptr.offset<4>();
     *
     * // Returns a pointer to element `vector[8]` with alignment of 16
     * ptr.offset<4>(2);
     * ```
     */
    template<size_t Step = 1>
    KERNEL_FLOAT_INLINE aligned_ptr<const T, detail::gcd(sizeof(T) * Step, alignment)>
    offset(size_t n = 1) const {
        return aligned_ptr<const T, detail::gcd(sizeof(T) * Step, alignment)> {get() + Step * n};
    }

    /**
     * See ``kernel_float::read``
     */
    template<size_t N>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> read() const {
        vector_storage<T, N> result;
        detail::copy_aligned_impl<T, N, alignment>::load(result.data(), get());
        return result;
    }

    /**
     * See ``kernel_float::write``
     */
    template<typename I, typename M = bool, typename E = broadcast_vector_extent_type<I, M>>
    KERNEL_FLOAT_INLINE vector<T, E> read(const I& indices, const M& mask = true) {
        return ::kernel_float::read(get(), indices, mask);
    }

    /**
     * Offsets the pointer by `Step * offset` items and then read the subsequent `N` items.
     *
     * Example
     * =======
     * ```
     * aligned_ptr<int> ptr = vector.data();
     *
     * // Returns vector[40], vector[41], vector[42], vector[43]
     * ptr.read_at<4>(10);
     *
     * // Returns vector[20], vector[21], vector[22]
     * ptr.read_at<2, 3>(10);
     * ```
     */
    template<size_t Step = 1, size_t N = Step>
    KERNEL_FLOAT_INLINE vector<T, extent<N>> read_at(size_t offset) const {
        return this->offset<Step>(offset).template read<N>();
    }

  private:
    const T* ptr_ = nullptr;
};

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_MEMORY_H
