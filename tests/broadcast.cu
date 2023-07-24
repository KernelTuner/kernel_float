#include "common.h"
#include "kernel_float.h"

namespace kf = kernel_float;

template<
    typename T,
    size_t N,
    typename = std::make_index_sequence<N>,
    typename = std::make_index_sequence<N * N>>
struct broadcast_test;

template<typename T, size_t N, size_t... Is, size_t... IIs>
struct broadcast_test<T, N, std::index_sequence<Is...>, std::index_sequence<IIs...>> {
    __host__ __device__ void operator()(generator<T> gen) {
        {
            kf::tensor<T, kf::extents<>> x = gen.next();
            T y = gen.next();
            kf::tensor<T, kf::extents<>> z = x + y;
        }

        {
            kf::tensor<T, kf::extents<N>> x = {gen.next(Is)...};
            T y = gen.next();
            kf::tensor<T, kf::extents<N>> z = x + y;
        }

        {
            kf::tensor<T, kf::extents<N, N>> x = {gen.next(IIs)...};
            T y = gen.next();
            kf::tensor<T, kf::extents<N, N>> z = x + y;
        }

        {
            kf::tensor<T, kf::extents<>> x = gen.next();
            kf::tensor<T, kf::extents<N>> y = {gen.next(Is)...};
            kf::tensor<T, kf::extents<N>> z = x + y;
        }

        {
            kf::tensor<T, kf::extents<N, 1>> x = {gen.next(Is)...};
            kf::tensor<T, kf::extents<1, N>> y = {gen.next(Is)...};
            kf::tensor<T, kf::extents<N, N>> z = x - y;
        }

        {
            kf::tensor<T, kf::extents<>> x = gen.next();
            kf::tensor<T, kf::extents<N, N>> y = {gen.next(IIs)...};
            kf::tensor<T, kf::extents<N, N>> z = x * y;
        }

        {
            kf::tensor<T, kf::extents<N, 1, 1>> x = {gen.next(Is)...};
            kf::tensor<T, kf::extents<1, N>> y = {gen.next(Is)...};
            kf::tensor<T, kf::extents<N, 1, N>> z = x / y;
        }

        {
            kf::tensor<T, kf::extents<N, 1>> x;
            kf::tensor<T, kf::extents<N, N>> y = x;
        }
    }
};

TEST_CASE("broadcast operators") {
    run_on_host_and_device<broadcast_test, int>();
}
