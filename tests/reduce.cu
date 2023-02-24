#include "catch.hpp"
#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

__host__ __device__ bool is_close(double a, double b) {
    return (isnan(a) && isnan(b)) || (isinf(a) && isinf(b)) || fabs(a - b) < 0.0001;
}

__host__ __device__ bool is_close(__half a, __half b) {
    return is_close(double(a), double(b));
}

__host__ __device__ bool is_close(long long a, long long b) {
    return a == b;
}

__host__ __device__ bool is_close(int a, int b) {
    return a == b;
}

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct reduction_test;

template<typename T, size_t N, size_t... Is>
struct reduction_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> v {gen.next(Is)...};

        bool b = (bool(v.get(Is)) && ...);
        ASSERT(kf::all(v) == b);

        b = (bool(v.get(Is)) || ...);
        ASSERT(kf::any(v) == b);

        T sum = v.get(0);
        for (int i = 1; i < N; i++) {
            sum = sum + v.get(i);
        }
        ASSERT(is_close(kf::sum(v), sum));

        T minimum = v.get(0);
        for (int i = 1; i < N; i++) {
            minimum = kf::ops::min<T> {}(minimum, v.get(i));
        }
        ASSERT(is_close(kf::min(v), minimum));

        T maximum = v.get(0);
        for (int i = 1; i < N; i++) {
            maximum = kf::ops::max<T> {}(maximum, v.get(i));
        }
        ASSERT(is_close(kf::max(v), maximum));
    }
};

TEST_CASE("reduction operations") {
    run_on_host_and_device<reduction_test, bool, int, float, double>();
    run_on_device<reduction_test, __half>();
}