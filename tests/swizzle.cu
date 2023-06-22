#include "common.h"
#include "kernel_float.h"
/*
namespace kf = kernel_float;

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct swizzle_test;

template<typename T, size_t N, size_t... Is>
struct swizzle_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        T items[N] = {gen.next(Is)...};
        kf::vec<T, N> a = {items[Is]...};

        ASSERT(equals<T>(items[0], kf::first(a)));
        ASSERT(equals<T>(items[N - 1], kf::last(a)));

        kf::vec<T, N> b = kf::reversed(a);
        ASSERT(equals<T>(b[Is], items[N - Is - 1]) && ...);

        b = kf::rotate_left<1>(a);
        ASSERT(equals<T>(b[Is], items[(Is + 1) % N]) && ...);

        b = kf::rotate_right<1>(a);
        ASSERT(equals<T>(b[Is], items[(Is + N - 1) % N]) && ...);

        b = kf::rotate_left<2>(a);
        ASSERT(equals<T>(b[Is], items[(Is + 2) % N]) && ...);

        b = kf::rotate_right<2>(a);
        ASSERT(equals<T>(b[Is], items[(Is + N - 2) % N]) && ...);

        kf::vec<T, 2 * N + 1> c = kf::concat(a, T {}, a);
        ASSERT(equals<T>(c[Is], items[Is]) && ...);
        ASSERT(equals<T>(c[N], T {}));
        ASSERT(equals<T>(c[N + 1 + Is], items[Is]) && ...);
    }
};

TEST_CASE("swizzle") {
    run_on_host_and_device<swizzle_test, int, float>();
}
*/