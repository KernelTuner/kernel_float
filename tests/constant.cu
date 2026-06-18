#include "common.h"

struct constant_ops_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        T value = gen.next();
        kf::vec<T, 2> vector = {gen.next(), gen.next()};

        auto C = [](auto x) { return kf::make_constant(x); };

        ASSERT_EQ(C(5.0) + value, T(5.0) + value);
        ASSERT_EQ(value + C(5.0), value + T(5.0));
        ASSERT_EQ(C(5.0) + vector, T(5.0) + vector);
        ASSERT_EQ(vector + C(5.0), vector + T(5.0));

        ASSERT_EQ(C(5.0) - value, T(5.0) - value);
        ASSERT_EQ(value - C(5.0), value - T(5.0));
        ASSERT_EQ(C(5.0) - vector, T(5.0) - vector);
        ASSERT_EQ(vector - C(5.0), vector - T(5.0));

        ASSERT_EQ(C(5.0) * value, T(5.0) * value);
        ASSERT_EQ(value * C(5.0), value * T(5.0));
        ASSERT_EQ(C(5.0) * vector, T(5.0) * vector);
        ASSERT_EQ(vector * C(5.0), vector * T(5.0));

        // These results in division by zero for integers
        //        ASSERT_EQ(C(5.0) / value, T(5) / value);
        //        ASSERT_EQ(value / C(5.0), value / T(5));
        //        ASSERT_EQ(C(5.0) / vector, T(5) / vector);
        //        ASSERT_EQ(vector / C(5.0), vector / T(5));
        //
        //        ASSERT_EQ(C(5.0) % value, T(5) % value);
        //        ASSERT_EQ(value % C(5.0), value % T(5));
        //        ASSERT_EQ(C(5.0) % vector, T(5) % vector);
        //        ASSERT_EQ(vector % C(5.0), vector % T(5));

        ASSERT_EQ(kf::cast<double>(C(T(5.0))), kf::make_vec(5.0));
        ASSERT_EQ(kf::cast<float>(C(T(5.0))), kf::make_vec(5.0f));
        ASSERT_EQ(kf::cast<int>(C(T(5.0))), kf::make_vec(5));

        ASSERT_EQ(kf::cast<T>(C(5.0)), kf::make_vec(T(5.0)));
        ASSERT_EQ(kf::cast<T>(C(5.0f)), kf::make_vec(T(5.0)));
        ASSERT_EQ(kf::cast<T>(C(5)), kf::make_vec(T(5.0)));

        auto expected = kf::make_vec(T(30.0));
        ASSERT_EQ(kf::fma(T(5), C(5.0), C(5.0)), expected);
        ASSERT_EQ(kf::fma(T(5), C(5.0f), C(5.0f)), expected);
        ASSERT_EQ(kf::fma(T(5), C(5), C(5)), expected);
        ASSERT_EQ(kf::fma(T(5), C(5.0f), C(5)), expected);
        ASSERT_EQ(kf::fma(T(5), C(5.0), C(5)), expected);
        ASSERT_EQ(kf::fma(T(5), C(5.0f), C(5.0)), expected);
    }
};

REGISTER_TEST_CASE("constant ops tests", constant_ops_tests, int, float, double)
REGISTER_TEST_CASE_GPU("constant ops tests", constant_ops_tests, __half, __nv_bfloat16)

struct constant_eq_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        ASSERT(kf::make_constant(T(5.0)) == double(5.0));
        ASSERT(kf::make_constant(T(5.0)) == float(5.0));
        ASSERT(kf::make_constant(T(5.0)) == int(5.0));

        ASSERT(kf::make_constant(double(5.0)) == T(5.0));
        ASSERT(kf::make_constant(float(5.0)) == T(5.0));
        ASSERT(kf::make_constant(int(5.0)) == T(5.0));

        ASSERT(kf::make_constant(T(5.0)) != double(6.0));
        ASSERT(kf::make_constant(T(5.0)) != float(6.0));
        ASSERT(kf::make_constant(T(5.0)) != int(6.0));

        ASSERT(kf::make_constant(double(5.0)) != T(6.0));
        ASSERT(kf::make_constant(float(5.0)) != T(6.0));
        ASSERT(kf::make_constant(int(5.0)) != T(6.0));
    }
};

REGISTER_TEST_CASE("constant eq tests", constant_eq_tests, int, float, double)
REGISTER_TEST_CASE_GPU("constant eq tests", constant_eq_tests, __half, __nv_bfloat16)
