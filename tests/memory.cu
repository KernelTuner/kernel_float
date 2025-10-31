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

struct aligned_access_test {
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

REGISTER_TEST_CASE("aligned access", aligned_access_test, int, float, double, __half, __nv_bfloat16)

struct vector_ptr_test {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T>, std::index_sequence<I...>) {
        using U = double;
        struct alignas(32) storage_type {
            U data[3 * N];
        };

        storage_type storage = {
            U(I)...,
            U(N + I)...,
            U(2 * N + I)...,
        };

        {
            kf::vector_ptr<const U, N> storage_ptr = kf::make_vec_ptr(storage.data);
            kf::vector_ptr<T, N, const U> ptr = storage_ptr;
            ASSERT_EQ(ptr.get(), static_cast<const U*>(storage.data));

            T expected[N] = {T(double(N + I))...};

            auto a = ptr.read(1);
            ASSERT_EQ_ALL(a[I], expected[I]);

            auto b = ptr[1];
            ASSERT_EQ_ALL(b[I], expected[I]);

            kf::vec<T, N> c = ptr.at(1);
            ASSERT_EQ_ALL(c[I], expected[I]);

            ASSERT_EQ(ptr.at(1).get(), static_cast<const U*>(&storage.data[N]));
        }

        {
            kf::vector_ptr<U, N> storage_ptr = kf::make_vec_ptr(storage.data);
            kf::vector_ptr<T, N, U> ptr = storage_ptr;
            ASSERT_EQ(ptr.get(), static_cast<U*>(storage.data));

            T expected[N] = {T(double(N + I))...};

            auto a = ptr.read(1);
            ASSERT_EQ_ALL(a[I], expected[I]);

            kf::vec<T, N> b = ptr[1];
            ASSERT_EQ_ALL(b[I], expected[I]);

            kf::vec<T, N> c = ptr.at(1);
            ASSERT_EQ_ALL(c[I], expected[I]);

            kf::vec<T, N> overwrite = {T(double(100 + I))...};
            ptr.at(1) = overwrite;

            kf::vec<T, N> e = ptr[1];
            ASSERT_EQ_ALL(e[I], overwrite[I]);

            ptr.write(1, T(1337.0));
            kf::vec<T, N> f = ptr[1];
            ASSERT_EQ_ALL(f[I], T(1337.0));

            ptr.at(1) += T(1.0);
            kf::vec<T, N> g = ptr[1];
            ASSERT_EQ_ALL(g[I], T(1338.0));

            kf::cast_to(ptr[1]) = double(3.14);
            kf::vec<T, N> h = ptr[1];
            ASSERT_EQ_ALL(h[I], T(3.14));
        }

        {
            // This does *not* require an explicit constructor (N == 1)
            kf::vector_ptr<T, 1, U> a1_ptr = storage.data;
            kf::vector_ptr<const T, 1, U> a2_ptr = storage.data;
            kf::vector_ptr<T, 1, const U> a3_ptr = storage.data;
            kf::vector_ptr<const T, 1, const U> a4_ptr = storage.data;

            ASSERT_EQ(a1_ptr.get(), static_cast<U*>(storage.data));
            ASSERT_EQ(a2_ptr.get(), static_cast<U*>(storage.data));
            ASSERT_EQ(a3_ptr.get(), static_cast<const U*>(storage.data));
            ASSERT_EQ(a4_ptr.get(), static_cast<const U*>(storage.data));

            // This *does* require an explicit constructor (N > 1)
            kf::vector_ptr<T, 2, U> b1_ptr = kf::vector_ptr<T, 2, U>(storage.data);
            kf::vector_ptr<const T, 2, U> b2_ptr = kf::vector_ptr<const T, 2, U>(storage.data);
            kf::vector_ptr<T, 2, const U> b3_ptr = kf::vector_ptr<T, 2, const U>(storage.data);
            kf::vector_ptr<const T, 2, const U> b4_ptr =
                kf::vector_ptr<const T, 2, const U>(storage.data);

            ASSERT_EQ(b1_ptr.get(), static_cast<U*>(storage.data));
            ASSERT_EQ(b2_ptr.get(), static_cast<U*>(storage.data));
            ASSERT_EQ(b3_ptr.get(), static_cast<const U*>(storage.data));
            ASSERT_EQ(b4_ptr.get(), static_cast<const U*>(storage.data));
        }

        {
            U* ptr = nullptr;

            auto a1 = kf::make_vec_ptr<T>(ptr);
            ASSERT(std::is_same<decltype(a1), kf::vector_ptr<T, 1, U>>::value);

            auto a2 = kf::make_vec_ptr<T, 2>(ptr);
            ASSERT(std::is_same<decltype(a2), kf::vector_ptr<T, 2, U>>::value);

            auto a3 = kf::make_vec_ptr(ptr);
            ASSERT(
                std::is_same<decltype(a3), kf::vector_ptr<U, 1, U, KERNEL_FLOAT_MAX_ALIGNMENT>>::
                    value);

            auto a4 = kf::make_vec_ptr<2>(ptr);
            ASSERT(std::is_same<decltype(a4), kf::vector_ptr<U, 2>>::value);
        }
    }
};

REGISTER_TEST_CASE_CPU("vectorized pointer", vector_ptr_test, int, float, double)
REGISTER_TEST_CASE_GPU(
    "vectorized pointer",
    vector_ptr_test,
    int,
    float,
    double,
    __half,
    __nv_bfloat16)