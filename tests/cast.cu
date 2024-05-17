#include "common.h"

struct cast_test {
    template<typename A, typename B>
    __host__ __device__ void check() {
        kf::vec<A, 2> input = {A(0.0), A(1.0)};
        kf::vec<B, 2> output = kf::cast<B>(input);

        ASSERT_EQ(output[0], B(0.0));
        ASSERT_EQ(output[1], B(1.0));
    }

    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        check<T, bool>();

        check<T, char>();
        check<T, unsigned char>();
        check<T, signed char>();

        check<T, signed short>();
        check<T, unsigned short>();

        check<T, signed int>();
        check<T, unsigned int>();

        check<T, signed long>();
        check<T, unsigned long>();

        check<T, signed long long>();
        check<T, unsigned long long>();

        check<T, float>();
        check<T, double>();

#if KERNEL_FLOAT_IS_DEVICE
        check<T, __half>();
        check<T, __nv_bfloat16>();
#endif
    }
};

REGISTER_TEST_CASE(
    "type cast",
    cast_test,
    bool,
    char,
    unsigned char,
    signed char,
    short,
    unsigned short,
    int,
    unsigned int,
    long,
    unsigned long,
    float,
    double)

REGISTER_TEST_CASE_GPU("type cast", cast_test, __half, __nv_bfloat16)
