#ifndef KERNEL_FLOAT_MEMORY_H
#define KERNEL_FLOAT_MEMORY_H

/*
#include "binops.h"
#include "broadcast.h"
#include "iterate.h"

namespace kernel_float {

    namespace detail {
        template <typename T, size_t N, typename Is = make_index_sequence_helper<N>>
        struct load_helper;

        template <typename T, size_t N, size_t... Is>
        struct load_helper<T, N, index_sequence<Is...>> {
            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets
            ) {
                return {base[offsets.data()[Is]]...};
            }

            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<bool, N> mask
            ) {
                if (all(mask)) {
                    return call(base, offsets);
                } else {
                    return {
                            (mask.data()[Is] ? base[offsets.data()[Is]] : T())...
                    };
                }
            }
        };
    }

    template <
            typename T,
            typename I,
            typename M,
            typename E = broadcast_vector_extent_type<I, M>
    >
    KERNEL_FLOAT_INLINE
    vector<T, E> load(const T* ptr, const I& indices, const M& mask) {
        static constexpr E new_size = {};

        return detail::load_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<bool>(mask, new_size)
        );
    }

    template <typename T, typename I>
    KERNEL_FLOAT_INLINE
    vector<T, vector_extent<I>> load(const T* ptr, const I& indices) {
        return detail::load_helper<T, vector_extent<I>::value>::call(
                ptr,
                cast<ptrdiff_t>(indices)
        );
    }

    template <size_t N, typename T>
    KERNEL_FLOAT_INLINE
    vector<T, extent<N>> load(const T* ptr, ptrdiff_t length) {
        using index_type = vector_value_type<I>;
        return load_masked(ptr, range<ptrdiff_t, N>(), range<ptrdiff_t, N>() < length);
    }

    template <size_t N, typename T>
    KERNEL_FLOAT_INLINE
    vector<T, extent<N>> load(const T* ptr) {
        return load(ptr, range<ptrdiff_t, N>());
    }

    namespace detail {
        template <typename T, size_t N>
        struct store_helper {
            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<bool, N> mask,
                    vector_storage<T, N> values
            ) {
                for (size_t i = 0; i < N; i++) {
                    if (mask.data()[i]) {
                        base[offset.data()[i]] = values.data()[i];
                    }
                }
            }

            KERNEL_FLOAT_INLINE
            vector_storage<T, N> call(
                    T* base,
                    vector_storage<ptrdiff_t, N> offsets,
                    vector_storage<T, N> values
            ) {
                for (size_t i = 0; i < N; i++) {
                    base[offset.data()[i]] = values.data()[i];
                }
            }
        };
    }

    template <
            typename T,
            typename I,
            typename M,
            typename V,
            typename E = broadcast_extent<vector_extent_type<V>, broadcast_vector_extent_type<M, I>>>
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const M& mask, const V& values) {
        static constexpr E new_size = {};

        return detail::store_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<bool>(mask, new_size),
                convert_storage<T>(values, new_size)
        );
    }

    template <
            typename T,
            typename I,
            typename V,
            typename E = broadcast_vector_extent_type<V, I>
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const V& values) {
        static constexpr E new_size = {};

        return detail::store_helper<T, E::value>::call(
                ptr,
                convert_storage<ptrdiff_t>(indices, new_size),
                convert_storage<T>(values, new_size)
        );
    }


    template <
            typename T,
            typename V
    >
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const V& values) {
        using E = vector_extent<V>;
        return store(ptr, range<ptrdiff_t, E::value>(), values);
    }

    template <typename T, typename I, typename S, typename V>
    KERNEL_FLOAT_INLINE
    void store(const T* ptr, const I& indices, const S& length, const V& values) {
        using index_type = vector_value_type<I>;
        return store(ptr, indices, (indices >= I(0)) & (indices < length), values);
    }


    template <typename T, size_t alignment>
    struct aligned_ptr_base {
        static_assert(alignof(T) % alignment == 0, "invalid alignment, must be multiple of alignment of `T`");

        KERNEL_FLOAT_INLINE
        aligned_ptr_base(): ptr_(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr_base(T* ptr): ptr_(ptr) {}

        KERNEL_FLOAT_INLINE
        T* get() const {
            // TOOD: check if this way is support across all compilers
#if defined(__has_builtin) && __has_builtin(__builtin_assume_aligned)
            return __builtin_assume_aligned(ptr_, alignment);
#else
            return ptr_;
#endif
        }

        KERNEL_FLOAT_INLINE
        operator T*() const {
            return get();
        }

        KERNEL_FLOAT_INLINE
        T& operator*() const {
            return *get();
        }

        template <typename I>
        KERNEL_FLOAT_INLINE
        T& operator[](I index) const {
            return get()[index);
        }

    private:
        T* ptr_ = nullptr;
    };

    template <typename T, size_t alignment = 256>
    struct aligned_ptr;

    template <typename T, size_t alignment>
    struct aligned_ptr: aligned_ptr_base<T, alignment> {
        using base_type = aligned_ptr_base<T, alignment>;

        KERNEL_FLOAT_INLINE
        aligned_ptr(): base_type(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<T, alignment> ptr): base_type(ptr.get()) {}
    };

    template <typename T, size_t alignment>
    struct aligned_ptr<const T, alignment>: aligned_ptr_base<const T, alignment> {
        using base_type = aligned_ptr_base<const T, alignment>;

        KERNEL_FLOAT_INLINE
        aligned_ptr(): base_type(nullptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        explicit aligned_ptr(const T* ptr): base_type(ptr) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<T, alignment> ptr): base_type(ptr.get()) {}

        KERNEL_FLOAT_INLINE
        aligned_ptr(aligned_ptr<const T, alignment> ptr): base_type(ptr.get()) {}
    };


    template <typename T, size_t alignment>
    KERNEL_FLOAT_INLINE
    T* operator+(aligned_ptr<T, alignment> ptr, ptrdiff_t index) {
        return ptr.get() + index;
    }

    template <typename T, size_t alignment>
    KERNEL_FLOAT_INLINE
    T* operator+(ptrdiff_t index, aligned_ptr<T, alignment> ptr) {
        return ptr.get() + index;
    }

    template <typename T, size_t alignment, size_t alignment2>
    KERNEL_FLOAT_INLINE
    ptrdiff_t operator-(aligned_ptr<T, alignment> left, aligned_ptr<T, alignment2> right) {
        return left.get() - right.get();
    }

    template <typename T>
    using unaligned_ptr = aligned_ptr<T, alignof(T)>;

}
*/

#endif  //KERNEL_FLOAT_MEMORY_H
