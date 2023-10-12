#include "common.h"

#define ASSERT_TYPE(A, B) ASSERT(std::is_same<decltype(A), B>::value);

struct constant_tests {
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
    }
};

REGISTER_TEST_CASE("constant tests", constant_tests, int, float, double)
REGISTER_TEST_CASE_GPU("constant tests", constant_tests, __half, __nv_bfloat16)
