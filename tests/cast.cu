#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<typename A, typename B, size_t N, typename Is = std::make_index_sequence<N>>
struct cast_test;

template<typename A, typename B, size_t N, size_t... Is>
struct cast_test<A, B, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<A> gen) {
        kf::vec<A, N> a {gen.next(Is)...};
        kf::vec<B, N> b = kf::cast<B>(a);

        ASSERT(bitwise_equal(B(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__half, long, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__half> gen) {
        kf::vec<__half, N> a {gen.next(Is)...};
        kf::vec<long, N> b = kf::cast<long>(a);
        ASSERT(bitwise_equal((long)(long long)a.get(Is), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<long, __half, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<long> gen) {
        kf::vec<long, N> a {gen.next(Is)...};
        kf::vec<__half, N> b = kf::cast<__half>(a);
        ASSERT(bitwise_equal(__half((long long)a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<unsigned long, __half, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<unsigned long> gen) {
        kf::vec<unsigned long, N> a {gen.next(Is)...};
        kf::vec<__half, N> b = kf::cast<__half>(a);
        ASSERT(bitwise_equal((__half)(unsigned long long)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__half, char, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__half> gen) {
        kf::vec<__half, N> a {gen.next(Is)...};
        kf::vec<char, N> b = kf::cast<char>(a);
        ASSERT(bitwise_equal((char)(int)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__nv_bfloat16, long, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__nv_bfloat16> gen) {
        kf::vec<__nv_bfloat16, N> a {gen.next(Is)...};
        kf::vec<long, N> b = kf::cast<long>(a);
        ASSERT(bitwise_equal((long)(long long)a.get(Is), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<long, __nv_bfloat16, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<long> gen) {
        kf::vec<long, N> a {gen.next(Is)...};
        kf::vec<__nv_bfloat16, N> b = kf::cast<__nv_bfloat16>(a);
        ASSERT(bitwise_equal(__nv_bfloat16((long long)a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<unsigned long, __nv_bfloat16, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<unsigned long> gen) {
        kf::vec<unsigned long, N> a {gen.next(Is)...};
        kf::vec<__nv_bfloat16, N> b = kf::cast<__nv_bfloat16>(a);
        ASSERT(bitwise_equal((__nv_bfloat16)(unsigned long long)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__nv_bfloat16, char, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__nv_bfloat16> gen) {
        kf::vec<__nv_bfloat16, N> a {gen.next(Is)...};
        kf::vec<char, N> b = kf::cast<char>(a);
        ASSERT(bitwise_equal((char)(int)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__nv_bfloat16, __half, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__nv_bfloat16> gen) {
        kf::vec<__nv_bfloat16, N> a {gen.next(Is)...};
        kf::vec<__half, N> b = kf::cast<__half>(a);
        ASSERT(bitwise_equal((__half)(float)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<size_t N, size_t... Is>
struct cast_test<__half, __nv_bfloat16, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<__half> gen) {
        kf::vec<__half, N> a {gen.next(Is)...};
        kf::vec<__nv_bfloat16, N> b = kf::cast<__nv_bfloat16>(a);
        ASSERT(bitwise_equal((__nv_bfloat16)(float)(a.get(Is)), b.get(Is)) && ...);
    }
};

template<typename B>
struct cast_to {
    template<typename A, size_t N>
    using type = cast_test<A, B, N>;
};

TEST_CASE("cast operators") {
    auto types = type_sequence<
        bool,
        char,
        short,
        int,
        unsigned int,
        long,
        unsigned long,
        long long,
        float,
        double,
        __half,
        __nv_bfloat16> {};

    run_on_host_and_device<cast_to<bool>::template type>(types);
    run_on_host_and_device<cast_to<char>::template type>(types);
    run_on_host_and_device<cast_to<short>::template type>(types);
    run_on_host_and_device<cast_to<int>::template type>(types);
    run_on_host_and_device<cast_to<long>::template type>(types);
    run_on_host_and_device<cast_to<long long>::template type>(types);
    run_on_host_and_device<cast_to<__half>::template type>(types);
    run_on_host_and_device<cast_to<__nv_bfloat16>::template type>(types);
    run_on_host_and_device<cast_to<float>::template type>(types);
    run_on_host_and_device<cast_to<double>::template type>(types);

    //bool, char, short, int, long long, __half, float, double
}
