#include "common.h"

struct reduction_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        // TODO: these tests do not consider special numbers: NaN, -Inf, +Inf, and -0.0

        {
            kf::vec<T, 1> a;
            ASSERT_APPROX(kf::min(a), T(0.0));
            ASSERT_APPROX(kf::max(a), T(0.0));
            ASSERT_APPROX(kf::sum(a), T(0.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(1.0)};
            ASSERT_APPROX(kf::min(a), T(1.0));
            ASSERT_APPROX(kf::max(a), T(1.0));
            ASSERT_APPROX(kf::sum(a), T(1.0));
            ASSERT_APPROX(kf::product(a), T(1.0));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);

            a = {T(5.0)};
            ASSERT_APPROX(kf::min(a), T(5.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(5.0));
            ASSERT_APPROX(kf::product(a), T(5.0));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);
        }

        {
            kf::vec<T, 2> a = {T(0.0), T(0.0)};
            ASSERT_APPROX(kf::min(a), T(0.0));
            ASSERT_APPROX(kf::max(a), T(0.0));
            ASSERT_APPROX(kf::sum(a), T(0.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5.0), T(0.0)};
            ASSERT_APPROX(kf::min(a), T(0.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(5.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 1);

            a = {T(5.0), T(-3.0)};
            ASSERT_APPROX(kf::min(a), T(-3.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(2.0));
            ASSERT_APPROX(kf::product(a), T(-15.0));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);
        }

        {
            kf::vec<T, 3> a;
            ASSERT_APPROX(kf::min(a), T(0.0));
            ASSERT_APPROX(kf::max(a), T(0.0));
            ASSERT_APPROX(kf::sum(a), T(0.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5.0), T(0.0), T(-1.0)};
            ASSERT_APPROX(kf::min(a), T(-1.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(4.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);

            a = {T(5.0), T(-3.0), T(1.0)};
            ASSERT_APPROX(kf::min(a), T(-3.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(3.0));
            ASSERT_APPROX(kf::product(a), T(-15.0));
            ASSERT_EQ(kf::all(a), true);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 3);
        }

        {
            kf::vec<T, 4> a;
            ASSERT_APPROX(kf::min(a), T(0.0));
            ASSERT_APPROX(kf::max(a), T(0.0));
            ASSERT_APPROX(kf::sum(a), T(0.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), false);
            ASSERT_EQ(kf::count(a), 0);

            a = {T(5.0), T(0.0), T(-1.0), T(0.0)};
            ASSERT_APPROX(kf::min(a), T(-1.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(4.0));
            ASSERT_APPROX(kf::product(a), T(0.0));
            ASSERT_EQ(kf::all(a), false);
            ASSERT_EQ(kf::any(a), true);
            ASSERT_EQ(kf::count(a), 2);

            a = {T(5.0), T(-3.0), T(1.0), T(-2.0)};
            ASSERT_APPROX(kf::min(a), T(-3.0));
            ASSERT_APPROX(kf::max(a), T(5.0));
            ASSERT_APPROX(kf::sum(a), T(1.0));
            ASSERT_APPROX(kf::product(a), T(30.0));
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
            kf::vec<T, 1> a = {T(-1.0)};
            kf::vec<T, 1> b = {T(2.0)};
            ASSERT_APPROX(kf::dot(a, b), T(-2.0));
            ASSERT_APPROX(kf::mag(a), T(1.0));
        }

        {
            kf::vec<T, 2> a = {T(3.0), T(-4.0)};
            kf::vec<T, 2> b = {T(2.0), T(1.0)};
            ASSERT_APPROX(kf::dot(a, b), T(2.0));
            ASSERT_APPROX(kf::mag(a), T(5.0));
        }

        {
            kf::vec<T, 3> a = {T(2.0), T(-3.0), T(6.0)};
            kf::vec<T, 3> b = {T(2.0), T(-1.0), T(3.0)};
            ASSERT_APPROX(kf::dot(a, b), T(25.0));
            ASSERT_APPROX(kf::mag(a), T(7.0));
        }

        {
            kf::vec<T, 4> a = {T(2.0), T(-4.0), T(5.0), T(6.0)};
            kf::vec<T, 4> b = {T(2.0), T(1.0), T(-3.0), T(1.0)};
            ASSERT_APPROX(kf::dot(a, b), T(-9.0));
            ASSERT_APPROX(kf::mag(a), T(9.0));
        }

        {
            kf::vec<T, 5> a = {T(1.0), T(-3.0), T(4.0), T(5.0), T(7.0)};
            kf::vec<T, 5> b = {T(2.0), T(0.0), T(1.0), T(-1.0), T(2.0)};
            ASSERT_APPROX(kf::dot(a, b), T(15.0));
            ASSERT_APPROX(kf::mag(a), T(10.0));
        }
    }
};

REGISTER_TEST_CASE("dot product/magnitude", dot_mag_tests, float, double)
REGISTER_TEST_CASE_GPU("dot product/magnitude", dot_mag_tests, __half, __nv_bfloat16)
