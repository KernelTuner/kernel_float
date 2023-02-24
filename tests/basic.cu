#include "catch.hpp"
#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<typename T, size_t N, typename Is = std::make_index_sequence<N>>
struct basic_test;

template<typename T, size_t N, size_t... Is>
struct basic_test<T, N, std::index_sequence<Is...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        T items[N] = {gen.next(Is)...};
        kf::vec<T, N> a = {items[Is]...};

        // check if getters work
        ASSERT(bitwise_equal(a.get(Is), items[Is]) && ...);
        ASSERT(bitwise_equal(a.get(kf::constant_index<Is> {}), items[Is]) && ...);
        ASSERT(bitwise_equal<T>(a[Is], items[Is]) && ...);
        ASSERT(bitwise_equal<T>(a[kf::constant_index<Is> {}], items[Is]) && ...);

        // check if setter works
        T new_items[N] = {gen.next(Is)...};
        (a.set(kf::constant_index<Is> {}, new_items[Is]), ...);
        ASSERT(bitwise_equal(a.get(Is), new_items[Is]) && ...);

        // check if setter works
        T more_new_items[N] = {gen.next(Is)...};
        ((a[Is] = more_new_items[Is]), ...);
        ASSERT(bitwise_equal(a.get(Is), more_new_items[Is]) && ...);

        // check default constructor
        kf::vec<T, N> b;
        ASSERT(bitwise_equal(b.get(Is), T {}) && ...);

        // check broadcast constructor
        T value = gen();
        kf::vec<T, N> c {value};
        ASSERT(bitwise_equal(c.get(Is), value) && ...);

        // check make_vec
        kf::vec<T, N> d = kf::make_vec(items[Is]...);
        ASSERT(bitwise_equal(d.get(Is), items[Is]) && ...);
    }
};

TEST_CASE("basic") {
    run_on_host_and_device<basic_test, bool, int, float, double>();
    run_on_device<basic_test, __half>();
}