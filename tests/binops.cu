#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct arithmetic_test;

template<typename T, size_t N, size_t... Is>
struct arithmetic_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...}, b {gen.next(Is)...}, c;

        // binary operator
        c = a + b;
        ASSERT(equals(c.get(Is), a.get(Is) + b.get(Is)) && ...);

        c = a - b;
        ASSERT(equals(c.get(Is), a.get(Is) - b.get(Is)) && ...);

        c = a * b;
        ASSERT(equals(c.get(Is), a.get(Is) * b.get(Is)) && ...);

        c = a / b;
        ASSERT(equals(c.get(Is), a.get(Is) / b.get(Is)) && ...);

        // assignment operator
        c = a;
        c += b;
        ASSERT(equals(c.get(Is), a.get(Is) + b.get(Is)) && ...);

        c = a;
        c -= b;
        ASSERT(equals(c.get(Is), a.get(Is) - b.get(Is)) && ...);

        c = a;
        c *= b;
        ASSERT(equals(c.get(Is), a.get(Is) * b.get(Is)) && ...);

        c = a;
        c /= b;
        ASSERT(equals(c.get(Is), a.get(Is) / b.get(Is)) && ...);
    }
};

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct minmax_test;

template<typename T, size_t N, size_t... Is>
struct minmax_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...}, b {gen.next(Is)...}, c;

        c = kf::min(a, b);
        ASSERT(equals(c.get(Is), a.get(Is) < b.get(Is) ? a.get(Is) : b.get(Is)) && ...);

        c = kf::max(a, b);
        ASSERT(equals(c.get(Is), a.get(Is) > b.get(Is) ? a.get(Is) : b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct minmax_test<float, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<float> gen) {
        kf::vec<float, N> a {gen.next(Is)...}, b {gen.next(Is)...}, c;

        c = kf::min(a, b);
        ASSERT(equals(c.get(Is), fminf(a.get(Is), b.get(Is))) && ...);

        c = kf::max(a, b);
        ASSERT(equals(c.get(Is), fmaxf(a.get(Is), b.get(Is))) && ...);
    }
};

template<size_t N, size_t... Is>
struct minmax_test<double, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<double> gen) {
        kf::vec<double, N> a {gen.next(Is)...}, b {gen.next(Is)...}, c;

        c = kf::min(a, b);
        ASSERT(equals(c.get(Is), fmin(a.get(Is), b.get(Is))) && ...);

        c = kf::max(a, b);
        ASSERT(equals(c.get(Is), fmax(a.get(Is), b.get(Is))) && ...);
    }
};

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct relational_test;

template<typename T, size_t N, size_t... Is>
struct relational_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...};
        kf::vec<T, N> b {gen.next(Is)...};
        kf::vec<T, N> c;

        c = a == b;
        ASSERT(equals(c.get(Is), T(a.get(Is) == b.get(Is))) && ...);

        c = a != b;
        ASSERT(equals(c.get(Is), T(a.get(Is) != b.get(Is))) && ...);

        c = a < b;
        ASSERT(equals(c.get(Is), T(a.get(Is) < b.get(Is))) && ...);

        c = a <= b;
        ASSERT(equals(c.get(Is), T(a.get(Is) <= b.get(Is))) && ...);

        c = a > b;
        ASSERT(equals(c.get(Is), T(a.get(Is) > b.get(Is))) && ...);

        c = a >= b;
        ASSERT(equals(c.get(Is), T(a.get(Is) >= b.get(Is))) && ...);
    }
};

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct bitwise_test;

template<typename T, size_t N, size_t... Is>
struct bitwise_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a = {gen.next(Is)...};
        kf::vec<T, N> b = {gen.next(Is)...};

        kf::vec<T, N> c = a | b;
        ASSERT(equals(c.get(Is), T(a.get(Is) | b.get(Is))) && ...);

        c = a & b;
        ASSERT(equals(c.get(Is), T(a.get(Is) & b.get(Is))) && ...);

        c = a ^ b;
        ASSERT(equals(c.get(Is), T(a.get(Is) ^ b.get(Is))) && ...);
    }
};

TEST_CASE("binary operators") {
    run_on_host_and_device<arithmetic_test, int, float, double>();
    run_on_device<arithmetic_test, __half, __nv_bfloat16>();

    run_on_host_and_device<minmax_test, int, float, double>();
    run_on_device<minmax_test, __half, __nv_bfloat16>();

    run_on_host_and_device<relational_test, bool, int, float, double>();
    run_on_device<relational_test, __half, __nv_bfloat16>();

    run_on_host_and_device<bitwise_test, bool, char, unsigned int, int, long, long long>();
}
