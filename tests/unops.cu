#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct int_test;

template<typename T, size_t N, size_t... Is>
struct int_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...};
        kf::vec<T, N> b;

        b = -a;
        ASSERT((b.get(Is) == -(a.get(Is))) && ...);

        b = ~a;
        ASSERT((b.get(Is) == ~(a.get(Is))) && ...);

        b = !a;
        ASSERT((b.get(Is) == !(a.get(Is))) && ...);
    }
};

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct float_test;

template<typename T, size_t N, size_t... Is>
struct float_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...};
        kf::vec<T, N> b;

        b = -a;
        ASSERT(bitwise_equal(-a.get(Is), b.get(Is)) && ...);

        // just some examples
        b = kf::cos(a);
        ASSERT(bitwise_equal(cos(a.get(Is)), b.get(Is)) && ...);

        b = kf::floor(a);
        ASSERT(bitwise_equal(floor(a.get(Is)), b.get(Is)) && ...);

        b = kf::abs(a);
        ASSERT(bitwise_equal(abs(a.get(Is)), b.get(Is)) && ...);

        b = kf::sqrt(a);
        ASSERT(bitwise_equal(sqrt(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct float_test<__half, N, std::index_sequence<Is...>> {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> a {gen.next(Is)...};
        kf::vec<T, N> b;

        b = -a;
        ASSERT(bitwise_equal(__hneg(a.get(Is)), b.get(Is)) && ...);

        // just some examples
        b = kf::cos(a);
        ASSERT(bitwise_equal(hcos(a.get(Is)), b.get(Is)) && ...);

        b = kf::floor(a);
        ASSERT(bitwise_equal(hfloor(a.get(Is)), b.get(Is)) && ...);

        b = kf::abs(a);
        ASSERT(bitwise_equal(__habs(a.get(Is)), b.get(Is)) && ...);

        b = kf::sqrt(a);
        ASSERT(bitwise_equal(hsqrt(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct float_test<__nv_bfloat16, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__nv_bfloat16> gen) {
        float_test<__half, N> {}(gen);
    }
};

TEST_CASE("unary operators") {
    run_on_host_and_device<int_test, char, short, int, unsigned, int, long, long long>();

    run_on_host_and_device<float_test, float, double>();
    run_on_device<float_test, __half, __nv_bfloat16>();
}
