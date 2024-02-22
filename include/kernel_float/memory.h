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
 * @brief A reference wrapper that allows reading/writing a vector of type `T`and length `N` with optional data
 * conversion.
 *
 * @tparam T The type of the elements as seen from the user's perspective.
 * @tparam N The number of elements in the vector.
 * @tparam U The underlying storage type. Defaults to the same type as T.
 * @tparam Align  The alignment constraint for read and write operations.
 */
template<typename T, size_t N, typename U = T, size_t Align = 1>
struct vector_ref {
    using pointer_type = U*;
    using value_type = decay_t<T>;
    using vector_type = vector<value_type, extent<N>>;

    /**
     * Constructs a vector_ref to manage access to a raw data pointer.
     *
     * @param data Pointer to the raw data this vector_ref will manage.
     */
    KERNEL_FLOAT_INLINE explicit vector_ref(pointer_type data) : data_(data) {}

    /**
     * Reads data from the underlying raw pointer, converting it to type `T`.
     *
     * @return vector_type A vector of type vector_type containing the read and converted data.
     */
    KERNEL_FLOAT_INLINE vector_type read() const {
        return convert<value_type, N>(read_aligned<Align, N>(data_));
    }

    /**
     * Writes data to the underlying raw pointer, converting it from the input vector if necessary.
     *
     * @tparam V The type of the input vector, defaults to `T`.
     * @param values The values to be written.
     */
    template<typename V = vector_type>
    KERNEL_FLOAT_INLINE void write(const V& values) const {
        write_aligned<Align>(data_, convert<U, N>(values));
    }

    /**
     * Conversion operator that is shorthand for `read()`.
     */
    KERNEL_FLOAT_INLINE operator vector_type() const {
        return read();
    }

    /**
     * Assignment operator that is shorthand for `write(values)`.
     */
    template<typename V>
    KERNEL_FLOAT_INLINE vector_ref operator=(const V& values) const {
        write(values);
        return *this;
    }

    /**
     * Gets the raw data pointer managed by this vector_ref
     */
    KERNEL_FLOAT_INLINE pointer_type get() const {
        return data_;
    }

  private:
    pointer_type data_ = nullptr;
};

/**
 * Specialization for `vector_ref` if the backing storage is const.
 */
template<typename T, size_t N, typename U, size_t Align>
struct vector_ref<T, N, const U, Align> {
    using pointer_type = const U*;
    using value_type = decay_t<T>;
    using vector_type = vector<value_type, extent<N>>;

    KERNEL_FLOAT_INLINE explicit vector_ref(pointer_type data) : data_(data) {}

    KERNEL_FLOAT_INLINE vector_type read() const {
        return convert<value_type, N>(read_aligned<Align, N>(data_));
    }

    KERNEL_FLOAT_INLINE operator vector_type() const {
        return read();
    }

    KERNEL_FLOAT_INLINE pointer_type get() const {
        return data_;
    }

  private:
    pointer_type data_ = nullptr;
};

#define KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(OP, OP_ASSIGN)                 \
    template<typename T, size_t N, typename U, size_t Align, typename V> \
    KERNEL_FLOAT_INLINE vector_ref<T, N> operator OP_ASSIGN(             \
        vector_ref<T, N, U, Align> ptr,                                  \
        const V& value) {                                                \
        ptr.write(ptr.read() OP value);                                  \
        return ptr;                                                      \
    }

KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(+, +=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(-, -=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(*, *=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(/, /=)

/**
 * A wrapper for a pointer that enables vectorized access and supports type conversions..
 *
 * The `vector_ptr<T, N, U>` type is designed to function as if its a `vec<T, N>*` pointer, allowing of reading and
 * writing `vec<T, N>` elements. However, the actual type of underlying storage is a pointer of type `U*`, where
 * automatic conversion is performed between `T` and `U` when reading/writing items.
 *
 * For example, a `vector_ptr<double, N, half>`  is useful where the data is stored in low precision (here 16 bit)
 * but it should be accessed as if it was in a higher precision format (here 64 bit).
 *
 * @tparam T The type of the elements as viewed by the user.
 * @tparam N The alignment of T in number of elements.
 * @tparam U The underlying storage type, defaults to T.
 */
template<typename T, size_t N, typename U = T>
struct vector_ptr {
    using pointer_type = U*;
    using value_type = decay_t<T>;

    /**
     * Default constructor sets the pointer to `NULL`.
     */
    vector_ptr() = default;

    /**
     * Constructor from a given pointer. It is up to the user to assert that the pointer is aligned to `Align` elements.
     */
    KERNEL_FLOAT_INLINE explicit vector_ptr(pointer_type p) : data_(p) {}

    /**
     * Constructs a vector_ptr from another vector_ptr with potentially different alignment and type. This constructor
     * only allows conversion if the alignment of the source is greater than or equal to the alignment of the target.
     */
    template<typename T2, size_t N2>
    KERNEL_FLOAT_INLINE vector_ptr(vector_ptr<T2, N2, U> p, enable_if_t<(N2 % N == 0), int> = {}) :
        data_(p.get()) {}

    /**
     * Accesses a reference to a vector at a specific index with optional alignment considerations.
     *
     * @tparam N The number of elements in the vector to access, defaults to the alignment.
     * @param index The index at which to access the vector.
     */
    template<size_t K = N>
    KERNEL_FLOAT_INLINE vector_ref<T, K, U, N> at(size_t index) const {
        return vector_ref<T, K, U, N> {data_ + index * N};
    }

    /**
     * Accesses a vector at a specific index.
     *
     * @tparam K The number of elements to read, defaults to `N`.
     * @param index The index from which to read the data.
     */
    template<size_t K = N>
    KERNEL_FLOAT_INLINE vector<value_type, extent<K>> read(size_t index) const {
        return this->template at<K>(index).read();
    }

    /**
     * Shorthand for `read(index)`.
     */
    KERNEL_FLOAT_INLINE const vector<value_type, extent<N>> operator[](size_t index) const {
        return read(index);
    }

    /**
     * Shorthand for `read(0)`.
     */
    KERNEL_FLOAT_INLINE const vector<value_type, extent<N>> operator*() const {
        return read(0);
    }

    /**
     * @brief Writes data to a specific index.
     *
     * @tparam K The number of elements to write, defaults to `N`.
     * @tparam V The type of the values being written.
     * @param index The index at which to write the data.
     * @param values The vector of values to write.
     */
    template<size_t K = N, typename V>
    KERNEL_FLOAT_INLINE void write(size_t index, const V& values) const {
        this->template at<K>(index).write(values);
    }

    /**
     * Shorthand for `at(index)`. Returns a vector reference to can be used
     * to assign to this pointer, contrary to `operator[]` that does not
     * allow assignment.
     */
    KERNEL_FLOAT_INLINE vector_ref<T, N, U, N> operator()(size_t index) const {
        return at(index);
    }

    /**
     * Gets the raw data pointer managed by this `vector_ptr`.
     */
    KERNEL_FLOAT_INLINE pointer_type get() const {
        return data_;
    }

  private:
    pointer_type data_ = nullptr;
};

/**
 * Specialization for `vector_ptr` if the backing storage is const.
 */
template<typename T, size_t N, typename U>
struct vector_ptr<T, N, const U> {
    using pointer_type = const U*;
    using value_type = decay_t<T>;

    vector_ptr() = default;
    KERNEL_FLOAT_INLINE explicit vector_ptr(pointer_type p) : data_(p) {}

    template<typename T2, size_t N2>
    KERNEL_FLOAT_INLINE
    vector_ptr(vector_ptr<T2, N2, const U> p, enable_if_t<(N2 % N == 0), int> = {}) :
        data_(p.get()) {}

    template<typename T2, size_t N2>
    KERNEL_FLOAT_INLINE vector_ptr(vector_ptr<T2, N2, U> p, enable_if_t<(N2 % N == 0), int> = {}) :
        data_(p.get()) {}

    template<size_t K = N>
    KERNEL_FLOAT_INLINE vector_ref<T, K, const U, N> at(size_t index) const {
        return vector_ref<T, K, const U, N> {data_ + index * N};
    }

    template<size_t K = N>
    KERNEL_FLOAT_INLINE vector<value_type, extent<K>> read(size_t index = 0) const {
        return this->template at<K>(index).read();
    }

    KERNEL_FLOAT_INLINE const vector<value_type, extent<N>> operator[](size_t index) const {
        return read(index);
    }

    KERNEL_FLOAT_INLINE const vector<value_type, extent<N>> operator*() const {
        return read(0);
    }

    KERNEL_FLOAT_INLINE pointer_type get() const {
        return data_;
    }

  private:
    pointer_type data_ = nullptr;
};

template<typename T, size_t N, typename U>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U> operator+(vector_ptr<T, N, U> p, size_t i) {
    return vector_ptr<T, N, U> {p.get() + i * N};
}

template<typename T, size_t N, typename U>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U> operator+(size_t i, vector_ptr<T, N, U> p) {
    return p + i;
}

/**
 * Creates a `vector_ptr<T, N>` from a raw pointer `U*` by asserting a specific alignment `N`.
 *
 * @tparam T The type of the elements as viewed by the user. This type may differ from `U`.
 * @tparam N The alignment constraint for the vector_ptr. Defaults to KERNEL_FLOAT_MAX_ALIGNMENT.
 * @tparam U The type of the elements pointed to by the raw pointer.
 */
template<typename T, size_t N = KERNEL_FLOAT_MAX_ALIGNMENT, typename U>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U> assert_aligned(U* ptr) {
    return vector_ptr<T, N, U> {ptr};
}

// Doxygen cannot deal with the `assert_aligned` being defined twice, we ignore the second definition.
/// @cond IGNORE
/**
 * Creates a `vector_ptr<T, N>` from a raw pointer `T*` by asserting a specific alignment `N`.
 *
 * @tparam N The alignment constraint for the vector_ptr. Defaults to KERNEL_FLOAT_MAX_ALIGNMENT.
 * @tparam T The type of the elements pointed to by the raw pointer.
 */
template<size_t N = KERNEL_FLOAT_MAX_ALIGNMENT, typename T>
KERNEL_FLOAT_INLINE vector_ptr<T, N> assert_aligned(T* ptr) {
    return vector_ptr<T, N> {ptr};
}
/// @endcond

template<typename T, size_t N = 1, typename U = T>
using vec_ptr = vector_ptr<T, N, U>;

#if defined(__cpp_deduction_guides)
template<typename T>
vector_ptr(T*) -> vector_ptr<T, 1, T>;

template<typename T>
vector_ptr(const T*) -> vector_ptr<T, 1, const T>;

#if __cpp_deduction_guides >= 201907L
template<typename T>
vec_ptr(T*) -> vec_ptr<T, 1, T>;

template<typename T>
vec_ptr(const T*) -> vec_ptr<T, 1, const T>;
#endif
#endif

}  // namespace kernel_float

#endif  //KERNEL_FLOAT_MEMORY_H
