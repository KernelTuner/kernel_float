#include "common.h"

struct constant_ops_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        T value = gen.next();
        kf::vec<T, 2> vector = {gen.next(), gen.next()};

        ASSERT_EQ(kf::make_constant(5.0) + value, T(5.0) + value);
        ASSERT_EQ(value + kf::make_constant(5.0), value + T(5.0));
        ASSERT_EQ(kf::make_constant(5.0) + vector, T(5.0) + vector);
        ASSERT_EQ(vector + kf::make_constant(5.0), vector + T(5.0));

        ASSERT_EQ(kf::make_constant(5.0) - value, T(5.0) - value);
        ASSERT_EQ(value - kf::make_constant(5.0), value - T(5.0));
        ASSERT_EQ(kf::make_constant(5.0) - vector, T(5.0) - vector);
        ASSERT_EQ(vector - kf::make_constant(5.0), vector - T(5.0));

        ASSERT_EQ(kf::make_constant(5.0) * value, T(5.0) * value);
        ASSERT_EQ(value * kf::make_constant(5.0), value * T(5.0));
        ASSERT_EQ(kf::make_constant(5.0) * vector, T(5.0) * vector);
        ASSERT_EQ(vector * kf::make_constant(5.0), vector * T(5.0));

        // These results in division by zero for integers
        //        ASSERT_EQ(kf::make_constant(5.0) / value, T(5) / value);
        //        ASSERT_EQ(value / kf::make_constant(5.0), value / T(5));
        //        ASSERT_EQ(kf::make_constant(5.0) / vector, T(5) / vector);
        //        ASSERT_EQ(vector / kf::make_constant(5.0), vector / T(5));
        //
        //        ASSERT_EQ(kf::make_constant(5.0) % value, T(5) % value);
        //        ASSERT_EQ(value % kf::make_constant(5.0), value % T(5));
        //        ASSERT_EQ(kf::make_constant(5.0) % vector, T(5) % vector);
        //        ASSERT_EQ(vector % kf::make_constant(5.0), vector % T(5));

        ASSERT_EQ(kf::cast<double>(kf::make_constant(T(5.0))), kf::make_vec(5.0));
        ASSERT_EQ(kf::cast<float>(kf::make_constant(T(5.0))), kf::make_vec(5.0f));
        ASSERT_EQ(kf::cast<int>(kf::make_constant(T(5.0))), kf::make_vec(5));

        ASSERT_EQ(kf::cast<T>(kf::make_constant(double(5.0))), kf::make_vec(T(5.0)));
        ASSERT_EQ(kf::cast<T>(kf::make_constant(float(5.0))), kf::make_vec(T(5.0)));
        ASSERT_EQ(kf::cast<T>(kf::make_constant(int(5.0))), kf::make_vec(T(5.0)));
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
    }
};

REGISTER_TEST_CASE("constant eq tests", constant_eq_tests, int, float, double)
REGISTER_TEST_CASE_GPU("constant eq tests", constant_eq_tests, __half, __nv_bfloat16)
