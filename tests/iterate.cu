#include "common.h"

struct select_tests {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        T data[8] = {
            gen.next(),
            gen.next(),
            gen.next(),
            gen.next(),
            gen.next(),
            gen.next(),
            gen.next(),
            gen.next()};
        kf::vec<T, 8> x = {data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]};

        // Empty
        ASSERT_EQ(select(x), (kf::vec<int, 0>()));
        ASSERT_EQ(select(x, kf::vec<int, 0>()), (kf::vec<T, 0>()));

        // One element
        ASSERT_EQ(select(x, 0), kf::make_vec(data[0]));
        ASSERT_EQ(select(x, 5), kf::make_vec(data[5]));
        ASSERT_EQ(select(x, 7), kf::make_vec(data[4]));

        // Two elements
        ASSERT_EQ(select(x, 0, 1), kf::make_vec(data[0], data[1]));
        ASSERT_EQ(select(x, 5, 0), kf::make_vec(data[5], data[0]));
        ASSERT_EQ(select(x, 6, 7), kf::make_vec(data[6], data[7]));

        // Two elements as array
        ASSERT_EQ(select(x, kf::make_vec(0, 1)), kf::make_vec(data[0], data[1]));
        ASSERT_EQ(select(x, kf::make_vec(5, 0)), kf::make_vec(data[5], data[0]));
        ASSERT_EQ(select(x, kf::make_vec(6, 7)), kf::make_vec(data[6], data[7]));

        // Three elements
        ASSERT_EQ(select(x, kf::make_vec(0, 1), 2), kf::make_vec(data[0], data[1], data[2]));
        ASSERT_EQ(select(x, kf::make_vec(5, 0, 7)), kf::make_vec(data[5], data[0], data[7]));
        ASSERT_EQ(select(x, 6, kf::make_vec(7, 2)), kf::make_vec(data[6], data[7], data[2]));

        // Method of vector
        ASSERT_EQ(x.select(), (kf::vec<T, 0>()));
        ASSERT_EQ(x.select(4), kf::make_vec(data[4]));
        ASSERT_EQ(x.select(4, 2), kf::make_vec(data[4], data[2]));
        ASSERT_EQ(x.select(4, 2, 7), kf::make_vec(data[4], data[2], data[7]));
    }
};