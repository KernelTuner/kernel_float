#include "common.h"

// Check if combining type `A` and `B` results in `C`
#define CHECK_PROMOTION(A, B, C) CHECK(std::is_same<kernel_float::promote_t<A, B>, C>::value);

TEST_CASE("type promotion") {
    CHECK_PROMOTION(int, int, int);
    CHECK_PROMOTION(int, float, float);
    CHECK_PROMOTION(int, double, double);
    //    CHECK_PROMOTION(int, unsigned int, int);
    CHECK_PROMOTION(int, bool, int);
    CHECK_PROMOTION(int, __half, __half);
    CHECK_PROMOTION(int, __nv_bfloat16, __nv_bfloat16);
    //    CHECK_PROMOTION(int, char, int);
    CHECK_PROMOTION(int, signed char, int);
    //    CHECK_PROMOTION(int, unsigned char, int);

    CHECK_PROMOTION(float, int, float);
    CHECK_PROMOTION(float, float, float);
    CHECK_PROMOTION(float, double, double);
    CHECK_PROMOTION(float, unsigned int, float);
    CHECK_PROMOTION(float, bool, float);
    CHECK_PROMOTION(float, __half, float);
    CHECK_PROMOTION(float, __nv_bfloat16, float);
    CHECK_PROMOTION(float, char, float);
    CHECK_PROMOTION(float, signed char, float);
    CHECK_PROMOTION(float, unsigned char, float);

    CHECK_PROMOTION(double, int, double);
    CHECK_PROMOTION(double, float, double);
    CHECK_PROMOTION(double, double, double);
    CHECK_PROMOTION(double, unsigned int, double);
    CHECK_PROMOTION(double, bool, double);
    CHECK_PROMOTION(double, __half, double);
    CHECK_PROMOTION(double, __nv_bfloat16, double);
    CHECK_PROMOTION(double, char, double);
    CHECK_PROMOTION(double, signed char, double);
    CHECK_PROMOTION(double, unsigned char, double);

    //    CHECK_PROMOTION(unsigned int, int, unsigned int);
    CHECK_PROMOTION(unsigned int, float, float);
    CHECK_PROMOTION(unsigned int, double, double);
    CHECK_PROMOTION(unsigned int, unsigned int, unsigned int);
    CHECK_PROMOTION(unsigned int, bool, unsigned int);
    CHECK_PROMOTION(unsigned int, __half, __half);
    CHECK_PROMOTION(unsigned int, __nv_bfloat16, __nv_bfloat16);
    //    CHECK_PROMOTION(unsigned int, char, unsigned int);
    //    CHECK_PROMOTION(unsigned int, signed char, unsigned int);
    CHECK_PROMOTION(unsigned int, unsigned char, unsigned int);

    CHECK_PROMOTION(bool, int, int);
    CHECK_PROMOTION(bool, float, float);
    CHECK_PROMOTION(bool, double, double);
    CHECK_PROMOTION(bool, unsigned int, unsigned int);
    CHECK_PROMOTION(bool, bool, bool);
    CHECK_PROMOTION(bool, __half, __half);
    CHECK_PROMOTION(bool, __nv_bfloat16, __nv_bfloat16);
    CHECK_PROMOTION(bool, char, char);
    CHECK_PROMOTION(bool, signed char, signed char);
    CHECK_PROMOTION(bool, unsigned char, unsigned char);

    CHECK_PROMOTION(__half, int, __half);
    CHECK_PROMOTION(__half, float, float);
    CHECK_PROMOTION(__half, double, double);
    CHECK_PROMOTION(__half, unsigned int, __half);
    CHECK_PROMOTION(__half, bool, __half);
    CHECK_PROMOTION(__half, __half, __half);
    CHECK_PROMOTION(__half, __nv_bfloat16, float);
    CHECK_PROMOTION(__half, char, __half);
    CHECK_PROMOTION(__half, signed char, __half);
    CHECK_PROMOTION(__half, unsigned char, __half);

    CHECK_PROMOTION(__nv_bfloat16, int, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, float, float);
    CHECK_PROMOTION(__nv_bfloat16, double, double);
    CHECK_PROMOTION(__nv_bfloat16, unsigned int, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, bool, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, __half, float);
    CHECK_PROMOTION(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, char, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, signed char, __nv_bfloat16);
    CHECK_PROMOTION(__nv_bfloat16, unsigned char, __nv_bfloat16);

    //    CHECK_PROMOTION(char, int, char);
    CHECK_PROMOTION(char, float, float);
    CHECK_PROMOTION(char, double, double);
    //    CHECK_PROMOTION(char, unsigned int, char);
    CHECK_PROMOTION(char, bool, char);
    CHECK_PROMOTION(char, __half, __half);
    CHECK_PROMOTION(char, __nv_bfloat16, __nv_bfloat16);
    CHECK_PROMOTION(char, char, char);
    //    CHECK_PROMOTION(char, signed char, char);
    //    CHECK_PROMOTION(char, unsigned char, char);

    CHECK_PROMOTION(signed char, int, int);
    CHECK_PROMOTION(signed char, float, float);
    CHECK_PROMOTION(signed char, double, double);
    //    CHECK_PROMOTION(signed char, unsigned int, signed char);
    CHECK_PROMOTION(signed char, bool, signed char);
    CHECK_PROMOTION(signed char, __half, __half);
    CHECK_PROMOTION(signed char, __nv_bfloat16, __nv_bfloat16);
    //    CHECK_PROMOTION(signed char, char, signed char);
    CHECK_PROMOTION(signed char, signed char, signed char);
    //    CHECK_PROMOTION(signed char, unsigned char, signed char);

    //    CHECK_PROMOTION(unsigned char, int, unsigned char);
    CHECK_PROMOTION(unsigned char, float, float);
    CHECK_PROMOTION(unsigned char, double, double);
    CHECK_PROMOTION(unsigned char, unsigned int, unsigned int);
    CHECK_PROMOTION(unsigned char, bool, unsigned char);
    CHECK_PROMOTION(unsigned char, __half, __half);
    CHECK_PROMOTION(unsigned char, __nv_bfloat16, __nv_bfloat16);
    //    CHECK_PROMOTION(unsigned char, char, unsigned char);
    //    CHECK_PROMOTION(unsigned char, signed char, unsigned char);
    CHECK_PROMOTION(unsigned char, unsigned char, unsigned char);
}