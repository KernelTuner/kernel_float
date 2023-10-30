#include "common.h"

struct load_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        T data[8] = {T(0.0), T(1.0), T(2.0), T(3.0), T(4.0), T(5.0), T(6.0), T(7.0)};

        {
            auto expected = kf::make_vec(T(3.0), T(2.0), T(7.0));
            auto output = kf::read(data, kf::make_vec(3, 2, 7));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(3.0), T(2.0), T(7.0));
            auto output = kf::read(data, kf::make_vec(3, 2, 7), kf::make_vec(true, true, true));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(3.0), T(), T(7.0));
            auto output = kf::read(data, kf::make_vec(3, 100, 7), kf::make_vec(true, false, true));
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(0.0), T(1.0), T(2.0));
            auto output = kf::read<3>(data);
            ASSERT_EQ(expected, output);
        }

        {
            auto expected = kf::make_vec(T(2.0), T(3.0), T(4.0));
            auto output = kf::read<3>(data + 2);
            ASSERT_EQ(expected, output);
        }
    }
};

REGISTER_TEST_CASE("load", load_test, int, float, double, __half, __nv_bfloat16)

struct store_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        {
            T data[4] = {T(0.0), T(1.0), T(2.0), T(3.0)};
            auto values = kf::make_vec(T(100.0), T(200.0));
            auto offsets = kf::make_vec(1, 3);
            kf::write(data, offsets, values);
            ASSERT_EQ(data[0], T(0.0));
            ASSERT_EQ(data[1], T(100.0));
            ASSERT_EQ(data[2], T(2.0));
            ASSERT_EQ(data[3], T(200.0));
        }

        {
            T data[4] = {T(0.0), T(1.0), T(2.0), T(3.0)};
            auto values = kf::make_vec(T(100.0), T(200.0));
            auto offsets = kf::make_vec(1, 3);
            auto mask = kf::make_vec(true, true);
            kf::write(data, offsets, values, mask);
            ASSERT_EQ(data[0], T(0.0));
            ASSERT_EQ(data[1], T(100.0));
            ASSERT_EQ(data[2], T(2.0));
            ASSERT_EQ(data[3], T(200.0));
        }

        {
            T data[4] = {T(0.0), T(1.0), T(2.0), T(3.0)};
            auto values = kf::make_vec(T(100.0), T(200.0));
            auto offsets = kf::make_vec(1, 3);
            auto mask = kf::make_vec(true, false);
            kf::write(data, offsets, values, mask);
            ASSERT_EQ(data[0], T(0.0));
            ASSERT_EQ(data[1], T(100.0));
            ASSERT_EQ(data[2], T(2.0));
            ASSERT_EQ(data[3], T(3.0));
        }

        {
            T data[4] = {T(0.0), T(1.0), T(2.0), T(3.0)};
            auto values = kf::make_vec(T(100.0), T(200.0));
            kf::write(data, values);
            ASSERT_EQ(data[0], T(100.0));
            ASSERT_EQ(data[1], T(200.0));
            ASSERT_EQ(data[2], T(2.0));
            ASSERT_EQ(data[3], T(3.0));
        }

        {
            T data[4] = {T(0.0), T(1.0), T(2.0), T(3.0)};
            auto values = kf::make_vec(T(100.0), T(200.0));
            kf::write(data + 1, values);
            ASSERT_EQ(data[0], T(0.0));
            ASSERT_EQ(data[1], T(100.0));
            ASSERT_EQ(data[2], T(200.0));
            ASSERT_EQ(data[3], T(3.0));
        }
    }
};

REGISTER_TEST_CASE("store", store_test, int, float, double, __half, __nv_bfloat16)

struct assign_conversion_test {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        kf::vec<T, N> x = {gen.next(I)...};
        kf::vec<float, N> y;

        kf::cast_to(y) = x;

        ASSERT_EQ_ALL(float(x[I]), y[I]);
    }
};

REGISTER_TEST_CASE(
    "assign conversion",
    assign_conversion_test,
    int,
    float,
    double,
    __half,
    __nv_bfloat16)

struct aligned_ptr_test {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T>, std::index_sequence<I...>) {
        struct alignas(32) storage_type {
            T data[N];
        };

        storage_type input = {T(double(I))...};
        auto v = kf::read_aligned<N>(input.data);
        ASSERT_EQ_ALL(v[I], T(double(I)));

        storage_type output = {T(double(I * 0))...};
        kf::write_aligned<N>(output.data, v);
        ASSERT_EQ_ALL(output.data[I], T(double(I)));
    }
};

REGISTER_TEST_CASE("aligned pointer", aligned_ptr_test, int, float, double, __half, __nv_bfloat16)
