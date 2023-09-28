#include "common.h"

struct basics_tests {
    template<typename T, size_t... I, size_t N = sizeof...(I)>
    __host__ __device__ void operator()(generator<T> gen, std::index_sequence<I...>) {
        // default constructor
        {
            kf::vec<T, N> x;
            ASSERT(equals(x[I], T()) && ...);
        }

        // filled with one
        {
            kf::vec<T, N> x = {T((gen.next(I), 1))...};
            ASSERT(equals(x[I], T(1)) && ...);
        }

        // filled with steps
        {
            kf::vec<T, N> x = {T(I)...};
            ASSERT(equals(x[I], T(I)) && ...);
        }

        // broadcast constructor
        {
            T init = gen.next();
            kf::vec<T, N> x {init};
            ASSERT(equals(x[I], init) && ...);
        }

        // Getters
        T items[N] = {gen.next(I)...};
        kf::vec<T, N> a = {items[I]...};

        ASSERT(equals(a[I], items[I]) && ...);
        ASSERT(equals(a.get(I), items[I]) && ...);
        ASSERT(equals(a.at(I), items[I]) && ...);
        ASSERT(equals(a(I), items[I]) && ...);

        // Data, begin, end
        ASSERT(a.size() == N);
        ASSERT(&a[0] == a.data());
        ASSERT(&a[0] == a.begin());
        ASSERT(&a[0] + N == a.end());
        ASSERT(&a[0] == a.cdata());
        ASSERT(&a[0] == a.cbegin());
        ASSERT(&a[0] + N == a.cend());

        // setters
        T new_items[N] = {gen.next(I)...};
        (a.set(I, new_items[I]), ...);
        ASSERT(equals(a[I], new_items[I]) && ...);
    }
};

REGISTER_TEST_CASE("basics", basics_tests, int, float)

struct creation_tests {
    __host__ __device__ void operator()(generator<int> gen) {
        using kernel_float::into_vec;
        using kernel_float::make_vec;

        // into_vec on scalar
        {
            kf::vec<int, 1> a = into_vec(int(5));
            ASSERT(a[0] == 5);
        }

        // into_vec on CUDA vector types
        {
            kf::vec<int, 1> a = into_vec(make_int1(5));
            kf::vec<int, 2> b = into_vec(make_int2(5, 4));
            kf::vec<int, 3> c = into_vec(make_int3(5, 4, -1));
            kf::vec<int, 4> d = into_vec(make_int4(5, 4, -1, 0));

            ASSERT(a[0] == 5);
            ASSERT(b[0] == 5 && b[1] == 4);
            ASSERT(c[0] == 5 && c[1] == 4 && c[2] == -1);
            ASSERT(d[0] == 5 && d[1] == 4 && d[2] == -1 && d[3] == 0);
        }

        // into_vec on C-style array
        {
            int items[3] = {1, 2, 3};
            kf::vec<int, 3> a = into_vec(items);
            ASSERT(a[0] == 1 && a[1] == 2 && a[2] == 3);
        }

        // into_vec on kf array
        {
            kf::vec<int, 3> items = {1, 2, 3};
            kf::vec<int, 3> a = into_vec(items);
            ASSERT(a[0] == 1 && a[1] == 2 && a[2] == 3);
        }

        // make_vec
        {
            kf::vec<int, 3> a = make_vec(true, short(2), int(3));
            ASSERT(a[0] == 1 && a[1] == 2 && a[2] == 3);
        }
    }

    __host__ __device__ void operator()(generator<float> gen) {
        using kernel_float::into_vec;
        using kernel_float::make_vec;

        // into_vec on scalar
        {
            kf::vec<float, 1> a = into_vec(float(5.0f));
            ASSERT(a[0] == 5.0f);
        }

        // into_vec on CUDA vector types
        {
            kf::vec<float, 1> a = into_vec(make_float1(5.0f));
            kf::vec<float, 2> b = into_vec(make_float2(5.0f, 4.0f));
            kf::vec<float, 3> c = into_vec(make_float3(5.0f, 4.0f, -1.0f));
            kf::vec<float, 4> d = into_vec(make_float4(5.0f, 4.0f, -1.0f, 0.0f));

            ASSERT(a[0] == 5.0f);
            ASSERT(b[0] == 5.0f && b[1] == 4.0f);
            ASSERT(c[0] == 5.0f && c[1] == 4.0f && c[2] == -1.0f);
            ASSERT(d[0] == 5.0f && d[1] == 4.0f && d[2] == -1.0f && d[3] == 0.0f);
        }

        // into_vec on C-style array
        {
            float items[3] = {1.0f, 2.0f, 3.0f};
            kf::vec<float, 3> a = into_vec(items);
            ASSERT(a[0] == 1.0f && a[1] == 2.0f && a[2] == 3.0f);
        }

        // into_vec on kf array
        {
            kf::vec<float, 3> items = {1.0f, 2.0f, 3.0f};
            kf::vec<float, 3> a = into_vec(items);
            ASSERT(a[0] == 1.0f && a[1] == 2.0f && a[2] == 3.0f);
        }

        // make_vec
        {
            kf::vec<float, 3> a = make_vec(true, int(2), 3.0f);
            ASSERT(a[0] == 1.0f && a[1] == 2.0f && a[2] == 3.0f);
        }
    }
};

REGISTER_TEST_CASE("into_vec and make_vec", creation_tests, int, float)
