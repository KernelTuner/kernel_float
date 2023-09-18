#include "common.h"

struct unops_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        T items[N] = {gen.next(I)...};
        kf::vec<T, N> a = {items[I]...};
        kf::vec<T, N> b;

        b = -a;
        ASSERT(equals(b[I], T(-items[I])) && ...);

        b = ~a;
        ASSERT(equals(b[I], T(~items[I])) && ...);

        b = !a;
        ASSERT(equals(b[I], T(!items[I])) && ...);
    }
};

REGISTER_TEST_CASE("unary operators", unops_tests, bool, int)

struct unops_float_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        double items[N] = {gen.next(I)...};
        kf::vec<T, N> a = {T(items[I])...};
        kf::vec<T, N> b;

        b = -a;
        ASSERT(equals(b[I], T(-items[I])) && ...);

        b = !a;
        ASSERT(equals(b[I], T(!items[I])) && ...);

        // Ideally, we would test all unary operators, but that would be a lot of work and not that useful since
        // all operators are generators by the same macro. Instead, we only check a few of them
        if constexpr (is_one_of<T, __half, __nv_bfloat16>) {
            b = sqrt(a);
            ASSERT(equals(b[I], hsqrt(T(items[I]))) && ...);

            b = sin(a);
            ASSERT(equals(b[I], hsin(T(items[I]))) && ...);

            b = cos(a);
            ASSERT(equals(b[I], hcos(T(items[I]))) && ...);

            b = log(a);
            ASSERT(equals(b[I], hlog(T(items[I]))) && ...);

            b = exp(a);
            ASSERT(equals(b[I], hexp(T(items[I]))) && ...);
        } else {
            b = sqrt(a);
            ASSERT(equals(b[I], sqrt(T(items[I]))) && ...);

            b = sin(a);
            ASSERT(equals(b[I], sin(T(items[I]))) && ...);

            b = cos(a);
            ASSERT(equals(b[I], cos(T(items[I]))) && ...);

            b = log(a);
            ASSERT(equals(b[I], log(T(items[I]))) && ...);

            b = exp(a);
            ASSERT(equals(b[I], exp(T(items[I]))) && ...);
        }
    }
};

REGISTER_TEST_CASE("unary float operators", unops_float_tests, float, double)
REGISTER_TEST_CASE_GPU("unary float operators", unops_float_tests, __half, __nv_bfloat16)