#ifndef KERNEL_FLOAT_VECTOR_H
#define KERNEL_FLOAT_VECTOR_H

#include "base.h"
#include "conversion.h"
#include "iterate.h"
#include "macros.h"
#include "reduce.h"
#include "triops.h"
#include "unops.h"

namespace kernel_float {

/**
 * Container that store fixed number of elements of type ``T``.
 *
 * It is not recommended to use this class directly, instead, use the type `vec<T, N>` which is an alias for
 * `vector<T, extent<N>, vector_storage<T, E>>`.
 *
 * @tparam T The type of the values stored within the vector.
 * @tparam E The size of this vector. Should be of type `extent<N>`.
 * @tparam S The object's storage class. Should be the type `vector_storage<T, E>`
 */
template<typename T, typename E, class S>
struct vector: public S {
    using value_type = T;
    using extent_type = E;
    using storage_type = S;

    // Copy another `vector<T, E>`
    vector(const vector&) = default;

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const storage_type& storage) : storage_type(storage) {}

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const value_type& input = {}) :
        storage_type(detail::broadcast_impl<T, extent<1>, E>::call(input)) {}

    // For all other arguments, we convert it using `convert_storage` according to broadcast rules
    template<typename U, enable_if_t<is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE vector(U&& input) :
        storage_type(convert_storage<T>(input, extent_type {})) {}

    template<typename U, enable_if_t<!is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE explicit vector(U&& input) :
        storage_type(convert_storage<T>(input, extent_type {})) {}

    // List of `N` (where N >= 2), simply pass forward to the storage
    template<
        typename A,
        typename B,
        typename... Rest,
        typename = enable_if_t<sizeof...(Rest) + 2 == E::size>>
    KERNEL_FLOAT_INLINE vector(const A& a, const B& b, const Rest&... rest) :
        storage_type {T(a), T(b), T(rest)...} {}

    /**
     * Returns the number of elements in this vector.
     */
    KERNEL_FLOAT_INLINE
    static constexpr size_t size() {
        return E::size;
    }

    /**
     * Returns a reference to the underlying storage type.
     */
    KERNEL_FLOAT_INLINE
    storage_type& storage() {
        return *this;
    }

    /**
     * Returns a reference to the underlying storage type.
     */
    KERNEL_FLOAT_INLINE
    const storage_type& storage() const {
        return *this;
    }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    T* data() {
        return storage().data();
    }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    const T* data() const {
        return storage().data();
    }

    KERNEL_FLOAT_INLINE
    const T* cdata() const {
        return this->data();
    }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& at(size_t i) {
        return *(this->data() + i);
    }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& at(size_t i) const {
        return *(this->data() + i);
    }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) {
        return at(i);
    }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const {
        return at(i);
    }

    KERNEL_FLOAT_INLINE
    T& operator()(size_t i) {
        return at(i);
    }

    KERNEL_FLOAT_INLINE
    const T& operator()(size_t i) const {
        return at(i);
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    T* begin() {
        return this->data();
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* begin() const {
        return this->data();
    }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* cbegin() const {
        return this->data();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    T* end() {
        return this->data() + size();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* end() const {
        return this->data() + size();
    }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* cend() const {
        return this->data() + size();
    }

    /**
     * Copy the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T get(size_t x) const {
        return at(x);
    }

    /**
     * Set the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    void set(size_t x, T value) {
        at(x) = std::move(value);
    }

    /**
     * Selects elements from the this vector based on the specified indices.
     *
     * Example
     * =======
     * ```
     * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
     * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
     *
     * vec<int, 4> indices = {0, 4, 4, 2};
     * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
     * ```
     */
    template<typename V, typename... Is>
    KERNEL_FLOAT_INLINE select_type<V, Is...> select(const Is&... indices) {
        return kernel_float::select(*this, indices...);
    }

    /**
     * Cast the elements of this vector to type `R` and returns a new vector.
     */
    template<typename R, RoundingMode Mode = RoundingMode::ANY>
    KERNEL_FLOAT_INLINE vector<R, extent_type> cast() const {
        return kernel_float::cast<R, Mode>(*this);
    }

    /**
     * Broadcast this vector into a new size `(Ns...)`.
     */
    template<size_t... Ns>
    KERNEL_FLOAT_INLINE vector<T, extent<Ns...>> broadcast(extent<Ns...> new_size = {}) const {
        return kernel_float::broadcast(*this, new_size);
    }

    /**
     * Apply the given function `F` to each element of this vector and returns a new vector with the results.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE vector<result_t<F, T>, E> map(F fun) const {
        return kernel_float::map(fun, *this);
    }

    /**
     * Reduce the elements of the given vector input into a single value using the function `F`.
     *
     * This function should be a binary function that takes two elements and returns one element. The order in which
     * the elements are reduced is not specified and depends on the reduction function and the vector type.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun) const {
        return kernel_float::reduce(fun, *this);
    }

    /**
     * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
     */
    KERNEL_FLOAT_INLINE flatten_type<vector> flatten() const {
        return kernel_float::flatten(*this);
    }

    /**
     * Apply the given function `F` to each element of this vector.
     */
    template<typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const {
        return kernel_float::for_each(*this, std::move(fun));
    }
};

/**
 * Convert the given `input` into a vector. This function can perform one of the following actions:
 *
 * - For vectors `vec<T, N>`, it simply returns the original vector.
 * - For primitive types `T` (e.g., `int`, `float`, `double`), it returns a `vec<T, 1>`.
 * - For array-like types (e.g., `std::array<T, N>`, `T[N]`), it returns `vec<T, N>`.
 * - For vector-like types (e.g., `int2`, `dim3`), it returns `vec<T, N>`.
 */
template<typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_vec(V&& input) {
    return into_vector_impl<V>::call(std::forward<V>(input));
}

template<typename T>
using scalar = vector<T, extent<1>>;

template<typename T, size_t N>
using vec = vector<T, extent<N>>;

// clang-format off
template<typename T> using vec1 = vec<T, 1>;
template<typename T> using vec2 = vec<T, 2>;
template<typename T> using vec3 = vec<T, 3>;
template<typename T> using vec4 = vec<T, 4>;
template<typename T> using vec5 = vec<T, 5>;
template<typename T> using vec6 = vec<T, 6>;
template<typename T> using vec7 = vec<T, 7>;
template<typename T> using vec8 = vec<T, 8>;
// clang-format on

/**
 * Create a vector from a variable number of input values.
 *
 * The resulting vector type is determined by promoting the types of the input values into a common type.
 * The number of input values determines the dimension of the resulting vector.
 *
 * Example
 * =======
 * ```
 * auto v1 = make_vec(1.0f, 2.0f, 3.0f); // Creates a vec<float, 3> [1.0f, 2.0f, 3.0f]
 * auto v2 = make_vec(1, 2, 3, 4);       // Creates a vec<int, 4> [1, 2, 3, 4]
 * ```
 */
template<typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return vector_storage<T, sizeof...(Args)> {T(args)...};
};

#if defined(__cpp_deduction_guides)
// Deduction guide for `vector`
template<typename... Args>
vector(Args&&... args) -> vector<promote_t<Args...>, extent<sizeof...(Args)>>;

// Deduction guides for aliases are only supported from C++20
#if __cpp_deduction_guides >= 201907L
template<typename... Args>
vec(Args&&... args) -> vec<promote_t<Args...>, sizeof...(Args)>;
#endif
#endif

}  // namespace kernel_float

#endif
