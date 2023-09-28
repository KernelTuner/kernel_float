#include "common.h"

struct load_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        T data[8] = {T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)};

        {
            auto expected = kf::make_vec(T(3), T(2), T(7));
            auto output = kf::load(data, kf::make_vec(3, 2, 7));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(3), T(2), T(7));
            auto output = kf::load(data, kf::make_vec(3, 2, 7), kf::make_vec(true, true, true));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(3), T(), T(7));
            auto output = kf::load(data, kf::make_vec(3, 100, 7), kf::make_vec(true, false, true));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(0), T(1), T(2));
            auto output = kf::loadn<3>(data);
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(2), T(3), T(4));
            auto output = kf::loadn<3>(data, 2);
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(6), T(7), T());
            auto output = kf::loadn<3>(data, 6, 8);
            ASSERT_EQ(expected, output);
        }
    }
};

REGISTER_TEST_CASE("load", load_test, int, float, double)
REGISTER_TEST_CASE_GPU("load", load_test, __half, __nv_bfloat16)

struct store_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            auto offsets = kf::make_vec(1, 3);
            kf::store(values, data, offsets);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(100));
            ASSERT_EQ(data[2], T(2));
            ASSERT_EQ(data[3], T(200));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            auto offsets = kf::make_vec(1, 3);
            auto mask = kf::make_vec(true, true);
            kf::store(values, data, offsets, mask);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(100));
            ASSERT_EQ(data[2], T(2));
            ASSERT_EQ(data[3], T(200));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            auto offsets = kf::make_vec(1, 3);
            auto mask = kf::make_vec(true, false);
            kf::store(values, data, offsets, mask);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(100));
            ASSERT_EQ(data[2], T(2));
            ASSERT_EQ(data[3], T(3));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            kf::storen(values, data);
            ASSERT_EQ(data[0], T(100));
            ASSERT_EQ(data[1], T(200));
            ASSERT_EQ(data[2], T(2));
            ASSERT_EQ(data[3], T(3));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            kf::storen(values, data, 1);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(100));
            ASSERT_EQ(data[2], T(200));
            ASSERT_EQ(data[3], T(3));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            kf::storen(values, data, 1, 4);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(100));
            ASSERT_EQ(data[2], T(200));
            ASSERT_EQ(data[3], T(3));
        }

        {
            T data[4] = {T(0), T(1), T(2), T(3)};
            auto values = kf::make_vec(T(100), T(200));
            kf::storen(values, data, 3, 4);
            ASSERT_EQ(data[0], T(0));
            ASSERT_EQ(data[1], T(1));
            ASSERT_EQ(data[2], T(2));
            ASSERT_EQ(data[3], T(100));
        }
    }
};

REGISTER_TEST_CASE("store", store_test, int, float, double)
REGISTER_TEST_CASE_GPU("store", store_test, __half, __nv_bfloat16)