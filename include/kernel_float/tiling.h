#ifndef KERNEL_FLOAT_TILING_H
#define KERNEL_FLOAT_TILING_H

#include "iterate.h"
#include "vector.h"

namespace kernel_float {

template<size_t... Ns>
struct block_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    block_size(dim3 thread_index) {
        if (rank > 0 && size(0) > 1) {
            thread_index_[0] = thread_index.x;
        }

        if (rank > 1 && size(1) > 1) {
            thread_index_[1] = thread_index.y;
        }

        if (rank > 2 && size(2) > 1) {
            thread_index_[2] = thread_index.z;
        }
    }

    KERNEL_FLOAT_INLINE
    size_t thread_index(size_t axis) const {
        return axis < rank ? thread_index_[axis] : 0;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }

  private:
    unsigned int thread_index_[rank] = {0};
};

template<size_t... Ns>
struct virtual_block_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    virtual_block_size(dim3 thread_index) {
        thread_index_ = thread_index.x;
    }

    KERNEL_FLOAT_INLINE
    size_t thread_index(size_t axis) const {
        size_t product_up_to_axis = 1;
#pragma unroll
        for (size_t i = 0; i < axis; i++) {
            product_up_to_axis *= size(i);
        }

        return (thread_index_ / product_up_to_axis) % size(axis);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }

  private:
    unsigned int thread_index_ = 0;
};

template<size_t... Ns>
struct tile_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis, size_t block_size = 0) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }
};

template<size_t... Ns>
struct tile_factor {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis, size_t block_size) {
        size_t factors[rank] = {Ns...};
        return block_size * (axis < rank ? factors[axis] : 1);
    }
};

namespace dist {
template<size_t N, size_t K>
struct blocked_impl {
    static constexpr bool is_exhaustive = N % K == 0;
    static constexpr size_t items_per_thread = (N / K) + (is_exhaustive ? 0 : 1);

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) {
        return is_exhaustive || (local_to_global(thread_index, local_index) < N);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) {
        return thread_index * items_per_thread + local_index;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) {
        return global_index % items_per_thread;
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) {
        return global_index / items_per_thread;
    }
};

struct blocked {
    template<size_t N, size_t K>
    using type = blocked_impl<N, K>;
};

template<size_t M, size_t N, size_t K>
struct cyclic_impl {
    static constexpr bool is_exhaustive = N % (K * M) == 0;
    static constexpr size_t items_per_thread = ((N / (K * M)) + (is_exhaustive ? 0 : 1)) * M;

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) {
        return is_exhaustive || (local_to_global(thread_index, local_index) < N);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) {
        return (local_index / M) * M * K + thread_index * M + (local_index % M);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) {
        return (global_index / (M * K)) * M + (global_index % M);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) {
        return (global_index / M) % K;
    }
};

struct cyclic {
    template<size_t N, size_t K>
    using type = cyclic_impl<1, N, K>;
};

template<size_t M>
struct block_cyclic {
    template<size_t N, size_t K>
    using type = cyclic_impl<M, N, K>;
};
}  // namespace dist

template<typename... Ds>
struct distributions {};

namespace detail {
template<size_t I, typename D>
struct instantiate_distribution_impl {
    template<size_t N, size_t K>
    using type = dist::cyclic::type<N, K>;
};

template<typename First, typename... Rest>
struct instantiate_distribution_impl<0, distributions<First, Rest...>> {
    template<size_t N, size_t K>
    using type = typename First::type<N, K>;
};

template<size_t I, typename First, typename... Rest>
struct instantiate_distribution_impl<I, distributions<First, Rest...>>:
    instantiate_distribution_impl<I + 1, distributions<Rest...>> {};

template<
    typename TileDim,
    typename BlockDim,
    typename Distributions,
    typename = make_index_sequence<TileDim::rank>>
struct tiling_impl;

template<typename TileDim, typename BlockDim, typename Distributions, size_t... Is>
struct tiling_impl<TileDim, BlockDim, Distributions, index_sequence<Is...>> {
    template<size_t I>
    using dist_type = typename instantiate_distribution_impl<I, Distributions>::
        type<TileDim::size(I, BlockDim::size(I)), BlockDim::size(I)>;

    static constexpr size_t rank = TileDim::rank;
    static constexpr size_t items_per_thread = (dist_type<Is>::items_per_thread * ... * 1);
    static constexpr bool is_exhaustive = (dist_type<Is>::is_exhaustive && ...);

    template<typename IndexType>
    KERNEL_FLOAT_INLINE static vector_storage<IndexType, rank>
    local_to_global(const BlockDim& block, size_t item) {
        vector_storage<IndexType, rank> result;
        ((result.data()[Is] = dist_type<Is>::local_to_global(
              block.thread_index(Is),
              item % dist_type<Is>::items_per_thread),
          item /= dist_type<Is>::items_per_thread),
         ...);
        return result;
    }

    KERNEL_FLOAT_INLINE
    static bool local_is_present(const BlockDim& block, size_t item) {
        bool is_present = true;
        ((is_present &= dist_type<Is>::local_is_present(
              block.thread_index(Is),
              item % dist_type<Is>::items_per_thread),
          item /= dist_type<Is>::items_per_thread),
         ...);
        return is_present;
    }
};
};  // namespace detail

template<typename T>
struct tiling_iterator;

/**
 * Represents a tiling where the elements given by `TileDim` are distributed over the
 * threads given by `BlockDim` according to the distributions given by `Distributions`.
 *
 * The template parameters should be the following:
 *
 * * ``TileDim``: Should be an instance of ``tile_size<...>``. For example,
 *   ``tile_size<16, 16>`` represents a 2-dimensional 16x16 tile.
 * * ``BlockDim``: Should be an instance of ``block_dim<...>``. For example,
 *    ``block_dim<16, 4>`` represents a thread block having X dimension 16
 *    and Y-dimension 4 for a total of 64 threads per block.
 * * ``Distributions``: Should be an instance of ``distributions<...>``. For example,
 *   ``distributions<dist::cyclic, dist::blocked>`` will distribute elements in
 *   cyclic fashion along the X-axis and blocked fashion along the Y-axis.
 * * ``IndexType``: The type used for index values (``int`` by default)
 */
template<
    typename TileDim,
    typename BlockDim,
    typename Distributions = distributions<>,
    typename IndexType = int>
struct tiling {
    using self_type = tiling<TileDim, BlockDim, Distributions, IndexType>;
    using impl_type = detail::tiling_impl<TileDim, BlockDim, Distributions>;
    using block_type = BlockDim;
    using tile_type = TileDim;

    static constexpr size_t rank = tile_type::rank;
    static constexpr size_t num_locals = impl_type::items_per_thread;

    using index_type = IndexType;
    using point_type = vector<index_type, extent<rank>>;

#if KERNEL_FLOAT_IS_DEVICE
    __forceinline__ __device__ tiling() : block_(threadIdx) {}
#endif

    KERNEL_FLOAT_INLINE
    tiling(BlockDim block, vec<index_type, rank> offset = {}) : block_(block), offset_(offset) {}

    /**
     * Returns the number of items per thread in the tiling.
     *
     * Note that this method is ``constexpr`` and can be called at compile-time.
     */
    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return impl_type::items_per_thread;
    }

    /**
     * Checks if the tiling is exhaustive, meaning all items are always present for all threads. If this returns
     * `true`, then ``is_present`` will always true for any given index.
     *
     * Note that this method is ``constexpr`` and can thus be called at compile-time.
     */
    KERNEL_FLOAT_INLINE
    static constexpr bool all_present() {
        return impl_type::is_exhaustive;
    }

    /**
     * Checks if a specific item is present for the current thread based on the distribution strategy. Not always
     * is the number of items stored per thread equal to the number of items _owned_ by each thread (for example,
     * if the tile size is not divisible by the block size). In this case, ``is_present`` will return `false` for
     * certain items.
     */
    KERNEL_FLOAT_INLINE
    bool is_present(size_t item) const {
        return all_present() || impl_type::local_is_present(block_, item);
    }

    /**
     * Returns the global coordinates of a specific item for the current thread.
     */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> at(size_t item) const {
        return impl_type::template local_to_global<index_type>(block_, item) + offset_;
    }

    /**
     * Returns the global coordinates of a specific item along a specified axis for the current thread.
     */
    KERNEL_FLOAT_INLINE
    index_type at(size_t item, size_t axis) const {
        return axis < rank ? at(item)[axis] : index_type {};
    }

    /**
     * Returns the global coordinates of a specific item for the current thread (alias of ``at``).
     */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> operator[](size_t item) const {
        return at(item);
    }

    /**
     * Returns a vector of global coordinates of all items present for the current thread.
     */
    KERNEL_FLOAT_INLINE
    vector<vector<index_type, extent<rank>>, extent<num_locals>> local_points() const {
        return range<num_locals>([&](size_t i) { return at(i); });
    }

    /**
     * Returns a vector of coordinate values along a specified axis for all items present for the current thread.
     */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<num_locals>> local_points(size_t axis) const {
        return range<num_locals>([&](size_t i) { return at(i, axis); });
    }

    /**
     * Returns a vector of boolean values representing the result of ``is_present`` of the items for the current thread.
     */
    KERNEL_FLOAT_INLINE
    vector<bool, extent<num_locals>> local_mask() const {
        return range<num_locals>([&](size_t i) { return is_present(i); });
    }

    /**
     * Returns the thread index (position) along a specified axis for the current thread.
     */
    KERNEL_FLOAT_INLINE
    index_type thread_index(size_t axis) const {
        return index_type(block_.thread_index(axis));
    }

    /**
     * Returns the size of the block (number of threads) along a specified axis.
     *
     * Note that this method is ``constexpr`` and can thus be called at compile-time.
     */
    KERNEL_FLOAT_INLINE
    static constexpr index_type block_size(size_t axis) {
        return index_type(block_type::size(axis));
    }

    /**
     * Returns the size of the tile along a specified axis.
     *
     * Note that this method is ``constexpr`` and can thus be called at compile-time.
     */
    KERNEL_FLOAT_INLINE
    static constexpr index_type tile_size(size_t axis) {
        return index_type(tile_type::size(axis, block_size(axis)));
    }

    /**
     * Returns the offset of the tile along a specified axis.
     */
    KERNEL_FLOAT_INLINE
    index_type tile_offset(size_t axis) const {
        return index_type(offset_[axis]);
    }

    /**
     * Returns a vector of thread indices for all axes.
     */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> thread_index() const {
        return range<rank>([&](size_t i) { return thread_index(i); });
    }

    /**
     * Returns a vector of block sizes for all axes.
     */
    KERNEL_FLOAT_INLINE
    static vector<index_type, extent<rank>> block_size() {
        return range<rank>([&](size_t i) { return block_size(i); });
    }

    /**
     * Returns a vector of tile sizes for all axes.
     */
    KERNEL_FLOAT_INLINE
    static vector<index_type, extent<rank>> tile_size() {
        return range<rank>([&](size_t i) { return tile_size(i); });
    }

    /**
     * Returns the offset of the tile for all axes.
     */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> tile_offset() const {
        return range<rank>([&](size_t i) { return tile_offset(i); });
    }

    /**
     * Returns an iterator pointing to the beginning of the tiling.
     */
    KERNEL_FLOAT_INLINE
    tiling_iterator<tiling> begin() const {
        return {*this, 0};
    }

    /**
     * Returns an iterator pointing to the end of the tiling.
     */
    KERNEL_FLOAT_INLINE
    tiling_iterator<tiling> end() const {
        return {*this, num_locals};
    }

    /**
     * Applies a provided function to each item present in the tiling for the current thread.
     * The function should take an index and a ``vector`` of global coordinates as arguments.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const {
#pragma unroll
        for (size_t i = 0; i < num_locals; i++) {
            if (is_present(i)) {
                fun(i, at(i));
            }
        }
    }

    /**
     * Adds ``offset`` to all points of this tiling and returns a new tiling.
     */
    KERNEL_FLOAT_INLINE friend tiling
    operator+(const tiling& self, const vector<index_type, extent<rank>>& offset) {
        return tiling {self.block_, self.offset_ + offset};
    }

    /**
     * Adds ``offset`` to all points of this tiling and returns a new tiling.
     */
    KERNEL_FLOAT_INLINE friend tiling
    operator+(const vector<index_type, extent<rank>>& offset, const tiling& self) {
        return self + offset;
    }

    /**
     * Adds ``offset`` to all points of this tiling.
     */
    KERNEL_FLOAT_INLINE friend tiling&
    operator+=(tiling& self, const vector<index_type, extent<rank>>& offset) {
        return self = self + offset;
    }

  private:
    BlockDim block_;
    vector<index_type, extent<rank>> offset_;
};

template<typename T>
struct tiling_iterator {
    using value_type = vector<typename T::index_type, extent<T::rank>>;

    KERNEL_FLOAT_INLINE
    tiling_iterator(const T& inner, size_t position = 0) : inner_(&inner), position_(position) {
        while (position_ < T::num_locals && !inner_->is_present(position_)) {
            position_++;
        }
    }

    KERNEL_FLOAT_INLINE
    value_type operator*() const {
        return inner_->at(position_);
    }

    KERNEL_FLOAT_INLINE
    tiling_iterator& operator++() {
        return *this = tiling_iterator(*inner_, position_ + 1);
    }

    KERNEL_FLOAT_INLINE
    tiling_iterator operator++(int) {
        tiling_iterator old = *this;
        this ++;
        return old;
    }

    KERNEL_FLOAT_INLINE
    friend bool operator==(const tiling_iterator& a, const tiling_iterator& b) {
        return a.position_ == b.position_;
    }

    KERNEL_FLOAT_INLINE
    friend bool operator!=(const tiling_iterator& a, const tiling_iterator& b) {
        return !operator==(a, b);
    }

    size_t position_ = 0;
    const T* inner_;
};

template<size_t TileDim, size_t BlockDim, typename D = dist::cyclic, typename IndexType = int>
using tiling_1d = tiling<tile_size<TileDim>, block_size<BlockDim>, distributions<D>, IndexType>;

// clang-format off
#define KERNEL_FLOAT_TILING_FOR_IMPL1(ITER_VAR, TILING, POINT_VAR, _)      \
        _Pragma("unroll")                                                         \
        for (size_t ITER_VAR = 0; ITER_VAR < (TILING).size(); ITER_VAR++)         \
            if (POINT_VAR = (TILING).at(ITER_VAR); (TILING).is_present(ITER_VAR)) \

#define KERNEL_FLOAT_TILING_FOR_IMPL2(ITER_VAR, TILING, INDEX_VAR, POINT_VAR)      \
        KERNEL_FLOAT_TILING_FOR_IMPL1(ITER_VAR, TILING, POINT_VAR, _) \
            if (INDEX_VAR = ITER_VAR; true)

#define KERNEL_FLOAT_TILING_FOR_IMPL(ITER_VAR, TILING, A, B, N, ...) \
    KERNEL_FLOAT_CALL(KERNEL_FLOAT_CONCAT(KERNEL_FLOAT_TILING_FOR_IMPL, N), ITER_VAR, TILING, A, B)

/**
 * Iterate over the points in a ``tiling<...>`` using a for loop.
 *
 * There are two ways to use this macro. Using the 1 variable form:
 * ```
 * auto t = tiling<tile_size<16, 16>, block_size<4, 4>>;
 *
 * KERNEL_FLOAT_TILING_FOR(t, auto point) {
 *  printf("%d,%d\n", point[0], point[1]);
 * }
 * ```
 *
 * Or using the 2 variables form:
 * ```
 * auto t = tiling<tile_size<16, 16>, block_size<4, 4>>;
 *
 * KERNEL_FLOAT_TILING_FOR(t, auto index, auto point) {
 *  printf("%d] %d,%d\n", index, point[0], point[1]);
 * }
 * ```
 */
#define KERNEL_FLOAT_TILING_FOR(...) \
    KERNEL_FLOAT_TILING_FOR_IMPL(KERNEL_FLOAT_CONCAT(__tiling_index_variable__, __LINE__), __VA_ARGS__, 2, 1)
// clang-format on

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_TILING_H
