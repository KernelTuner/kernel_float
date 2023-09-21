#include "common.h"

struct triops_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        T x[N] = {gen.next(I)...};
        T y[N] = {gen.next(I)...};
        T z[N] = {gen.next(I)...};

        kf::vec<T, N> a = {x[I]...};
        kf::vec<T, N> b = {y[I]...};
        kf::vec<T, N> c = {z[I]...};

        kf::vec<T, N> answer = kf::where(a, b, c);
        ASSERT_EQ_ALL(answer[I], bool(x[I]) ? y[I] : z[I]);

        answer = kf::where(a, b);
        ASSERT_EQ_ALL(answer[I], bool(x[I]) ? y[I] : T());

        answer = kf::where(a);
        ASSERT_EQ_ALL(answer[I], T(bool(x[I])));

        answer = kf::fma(a, b, c);
        ASSERT_EQ_ALL(answer[I], x[I] * y[I] + z[I]);
    }
};

REGISTER_TEST_CASE("ternary operators", triops_tests, int, float, double)
REGISTER_TEST_CASE_GPU("ternary operators", triops_tests, __half, __nv_bfloat16)
