#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct where_test;

template<typename T, size_t N, size_t... Is>
struct where_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, N> cond = {gen.next(Is)...};
        kf::vec<T, N> left = {gen.next(Is)...};
        kf::vec<T, N> right = {gen.next(Is)...};

        auto result = kf::where(cond, left, right);
        ASSERT(equals<T>(result[Is], cond[Is] ? left[Is] : right[Is]) && ...);

        result = kf::where(cond, left);
        ASSERT(equals<T>(result[Is], cond[Is] ? left[Is] : T {0}) && ...);

        result = kf::where<T>(cond);
        ASSERT(equals<T>(result[Is], cond[Is] ? T {1} : T {0}) && ...);
    }
};

TEST_CASE("conditional") {
    run_on_host_and_device<where_test, int, float, bool>();
}
