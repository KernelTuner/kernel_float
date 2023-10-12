#include "common.h"

struct binops_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        T x[N] = {gen.next(I)...};
        T y[N] = {gen.next(I)...};

        kf::vec<T, N> a = {x[I]...};
        kf::vec<T, N> b = {y[I]...};
        kf::vec<T, N> c;

        // Arithmetic
        c = a + b;
        ASSERT(equals(T(x[I] + y[I]), c[I]) && ...);

        c = a - b;
        ASSERT(equals(T(x[I] - y[I]), c[I]) && ...);

        c = a * b;
        ASSERT(equals(T(x[I] * y[I]), c[I]) && ...);

        // Results in division by zero
        //        c = a / b;
        //        ASSERT(equals(T(x[I] / y[I]), c[I]) && ...);

        // Results in division by zero
        //        c = a % b;
        //        ASSERT(equals(T(x[I] % y[I]), c[I]) && ...);

        // Comparison
        auto from_bool = [](bool x) { return x ? T(1.0) : T(0.0); };

        c = a < b;
        ASSERT(equals(from_bool(x[I] < y[I]), c[I]) && ...);

        c = a > b;
        ASSERT(equals(from_bool(x[I] > y[I]), c[I]) && ...);

        c = a <= b;
        ASSERT(equals(from_bool(x[I] <= y[I]), c[I]) && ...);

        c = a >= b;
        ASSERT(equals(from_bool(x[I] >= y[I]), c[I]) && ...);

        c = a == b;
        ASSERT(equals(from_bool(x[I] == y[I]), c[I]) && ...);

        c = a != b;
        ASSERT(equals(from_bool(x[I] != y[I]), c[I]) && ...);

        // Assignment
        c = a;
        c += b;
        ASSERT(equals(T(x[I] + y[I]), c[I]) && ...);

        c = a;
        c -= b;
        ASSERT(equals(T(x[I] - y[I]), c[I]) && ...);

        c = a;
        c *= b;
        ASSERT(equals(T(x[I] * y[I]), c[I]) && ...);
    }
};

REGISTER_TEST_CASE("binary operators", binops_tests, bool, int, float, double)
REGISTER_TEST_CASE_GPU("binary operators", binops_tests, __half, __nv_bfloat16)

struct binops_float_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        T x[N] = {gen.next(I)...};
        T y[N] = {gen.next(I)...};

        kf::vec<T, N> a = {x[I]...};
        kf::vec<T, N> b = {y[I]...};
        kf::vec<T, N> c;

        c = a / b;
        ASSERT(equals(T(x[I] / y[I]), c[I]) && ...);

        // remainder is not support for fp16
        if constexpr (is_none_of<T, __half, __nv_bfloat16>) {
            //            c = a % b;
            //            ASSERT(equals(T(fmod(x[I], y[I])), c[I]) && ...);
        }
    }
};

REGISTER_TEST_CASE("binary float operators", binops_float_tests, float, double)
REGISTER_TEST_CASE_GPU("binary float operators", binops_float_tests, __half, __nv_bfloat16)

struct minmax_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        T x[N] = {gen.next(I)...};
        T y[N] = {gen.next(I)...};

        kf::vec<T, N> a = {x[I]...};
        kf::vec<T, N> b = {y[I]...};

        kf::vec<T, N> lo = min(a, b);
        kf::vec<T, N> hi = max(a, b);

        if constexpr (is_one_of<T, double>) {
            ASSERT(equals(fmin(a[I], b[I]), lo[I]) && ...);
            ASSERT(equals(fmax(a[I], b[I]), hi[I]) && ...);
        } else if constexpr (is_one_of<T, float>) {
            ASSERT(equals(fminf(a[I], b[I]), lo[I]) && ...);
            ASSERT(equals(fmaxf(a[I], b[I]), hi[I]) && ...);
        } else if constexpr (is_one_of<T, __half, __nv_bfloat16>) {
            // __hmin/__hmax are only supported in CC >= 8
#if KERNEL_FLOAT_CUDA_ARCH >= 800
            ASSERT(equals(__hmin(a[I], b[I]), lo[I]) && ...);
            ASSERT(equals(__hmax(a[I], b[I]), hi[I]) && ...);
#endif
        } else {
            ASSERT(equals(x[I] < y[I] ? x[I] : y[I], lo[I]) && ...);
            ASSERT(equals(x[I] < y[I] ? y[I] : x[I], hi[I]) && ...);
        }
    }
};

REGISTER_TEST_CASE("min/max functions", minmax_tests, bool, int, float, double)
REGISTER_TEST_CASE_GPU("min/max functions", minmax_tests, __half, __nv_bfloat16)

struct cross_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        kf::vec<T, 3> a = {T(1.0), T(2.0), T(3.0)};
        kf::vec<T, 3> b = {T(4.0), T(5.0), T(6.0)};
        kf::vec<T, 3> c = cross(a, b);

        ASSERT(c[0] == T(-3.0));
        ASSERT(c[1] == T(6.0));
        ASSERT(c[2] == T(-3.0));
    }
};

REGISTER_TEST_CASE("cross product", cross_test, float, double)
REGISTER_TEST_CASE_GPU("cross product", cross_test, __half, __nv_bfloat16)