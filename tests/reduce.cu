#include "common.h"

struct reduction_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        // TODO: these tests do not consider special numbers: NaN, -Inf, +Inf, and -0.0

        {
            kf::vec<T, 1> a;
            ASSERT_APPROX(kf::min(a), T(0));
            ASSERT_APPROX(kf::max(a), T(0));
            ASSERT_APPROX(kf::sum(a), T(0));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(1)};
            ASSERT_APPROX(kf::min(a), T(1));
            ASSERT_APPROX(kf::max(a), T(1));
            ASSERT_APPROX(kf::sum(a), T(1));
            ASSERT_APPROX(kf::product(a), T(1));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);

            a = {T(5)};
            ASSERT_APPROX(kf::min(a), T(5));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(5));
            ASSERT_APPROX(kf::product(a), T(5));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);
        }

        {
            kf::vec<T, 2> a = {T(0), T(0)};
            ASSERT_APPROX(kf::min(a), T(0));
            ASSERT_APPROX(kf::max(a), T(0));
            ASSERT_APPROX(kf::sum(a), T(0));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5), T(0)};
            ASSERT_APPROX(kf::min(a), T(0));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(5));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);

            a = {T(5), T(-3)};
            ASSERT_APPROX(kf::min(a), T(-3));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(2));
            ASSERT_APPROX(kf::product(a), T(-15));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);
        }

        {
            kf::vec<T, 3> a;
            ASSERT_APPROX(kf::min(a), T(0));
            ASSERT_APPROX(kf::max(a), T(0));
            ASSERT_APPROX(kf::sum(a), T(0));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5), T(0), T(-1)};
            ASSERT_APPROX(kf::min(a), T(-1));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(4));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);

            a = {T(5), T(-3), T(1)};
            ASSERT_APPROX(kf::min(a), T(-3));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(3));
            ASSERT_APPROX(kf::product(a), T(-15));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 3);
        }

        {
            kf::vec<T, 4> a;
            ASSERT_APPROX(kf::min(a), T(0));
            ASSERT_APPROX(kf::max(a), T(0));
            ASSERT_APPROX(kf::sum(a), T(0));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5), T(0), T(-1), T(0)};
            ASSERT_APPROX(kf::min(a), T(-1));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(4));
            ASSERT_APPROX(kf::product(a), T(0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);

            a = {T(5), T(-3), T(1), T(-2)};
            ASSERT_APPROX(kf::min(a), T(-3));
            ASSERT_APPROX(kf::max(a), T(5));
            ASSERT_APPROX(kf::sum(a), T(1));
            ASSERT_APPROX(kf::product(a), T(30));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 4);
        }
    }
};

REGISTER_TEST_CASE("reductions", reduction_tests, int, float, double)
REGISTER_TEST_CASE_GPU("reductions", reduction_tests, __half, __nv_bfloat16)

struct dot_mag_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        {
            kf::vec<T, 1> a = {-1};
            kf::vec<T, 1> b = {2};
            ASSERT_APPROX(kf::dot(a, b), T(-2));
            ASSERT_APPROX(kf::mag(a), T(1));
        }

        {
            kf::vec<T, 2> a = {3, -4};
            kf::vec<T, 2> b = {2, 1};
            ASSERT_APPROX(kf::dot(a, b), T(2));
            ASSERT_APPROX(kf::mag(a), T(5));
        }

        {
            kf::vec<T, 3> a = {2, -3, 6};
            kf::vec<T, 3> b = {2, -1, 3};
            ASSERT_APPROX(kf::dot(a, b), T(25));
            ASSERT_APPROX(kf::mag(a), T(7));
        }

        {
            kf::vec<T, 4> a = {2, -4, 5, 6};
            kf::vec<T, 4> b = {2, 1, -3, 1};
            ASSERT_APPROX(kf::dot(a, b), T(-9));
            ASSERT_APPROX(kf::mag(a), T(9));
        }

        {
            kf::vec<T, 5> a = {1, -3, 4, 5, 7};
            kf::vec<T, 5> b = {2, 0, 1, -1, 2};
            ASSERT_APPROX(kf::dot(a, b), T(15));
            ASSERT_APPROX(kf::mag(a), T(10));
        }
    }
};

REGISTER_TEST_CASE("dot product/magnitude", dot_mag_tests, float, double)
REGISTER_TEST_CASE_GPU("dot product/magnitude", dot_mag_tests, __half, __nv_bfloat16)
