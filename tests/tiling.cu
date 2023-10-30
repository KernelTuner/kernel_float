#include "common.h"
#include "kernel_float/tiling.h"

struct basic_tiling_test {
    template<typename T>
    __host__ __device__ void operator()(generator<T> gen) {
        using TestTiling = kf::tiling<
            kf::tile_size<8, 8>,
            kf::block_size<2, 4>,
            kf::distributions<kf::dist::cyclic, kf::dist::blocked>  //
            >;
        auto tiling = TestTiling(dim3(1, 2, 0));

        ASSERT_EQ(TestTiling::size(), size_t(8));

        ASSERT_EQ(
            tiling.local_points(),
            kf::make_vec(
                kf::make_vec(1, 2),
                kf::make_vec(3, 2),
                kf::make_vec(5, 2),
                kf::make_vec(7, 2),
                kf::make_vec(1, 6),
                kf::make_vec(3, 6),
                kf::make_vec(5, 6),
                kf::make_vec(7, 6)));

        ASSERT_EQ(tiling.local_points(0), kf::make_vec(1, 3, 5, 7, 1, 3, 5, 7));
        ASSERT_EQ(tiling.local_points(1), kf::make_vec(2, 2, 2, 2, 6, 6, 6, 6));

        ASSERT_EQ(tiling.at(0), kf::make_vec(1, 2));
        ASSERT_EQ(tiling.at(1), kf::make_vec(3, 2));
        ASSERT_EQ(tiling.at(2), kf::make_vec(5, 2));

        ASSERT_EQ(tiling.at(0, 0), 1);
        ASSERT_EQ(tiling.at(0, 1), 2);
        ASSERT_EQ(tiling.at(1, 0), 3);
        ASSERT_EQ(tiling.at(1, 1), 2);

        ASSERT_EQ(tiling[0], kf::make_vec(1, 2));
        ASSERT_EQ(tiling[1], kf::make_vec(3, 2));
        ASSERT_EQ(tiling[2], kf::make_vec(5, 2));
        ASSERT_EQ(tiling[3], kf::make_vec(7, 2));

        ASSERT_EQ(
            tiling.local_mask(),
            kf::make_vec(true, true, true, true, true, true, true, true));
        ASSERT_EQ(TestTiling::all_present(), true);
        ASSERT_EQ(tiling.is_present(0), true);
        ASSERT_EQ(tiling.is_present(1), true);
        ASSERT_EQ(tiling.is_present(2), true);
        ASSERT_EQ(tiling.is_present(3), true);

        ASSERT_EQ(tiling.thread_index(0), 1);
        ASSERT_EQ(tiling.thread_index(1), 2);
        ASSERT_EQ(tiling.thread_index(2), 0);
        ASSERT_EQ(tiling.thread_index(), kf::make_vec(1, 2));

        ASSERT_EQ(TestTiling::block_size(0), 2);
        ASSERT_EQ(TestTiling::block_size(1), 4);
        ASSERT_EQ(TestTiling::block_size(2), 1);
        ASSERT_EQ(TestTiling::block_size(), kf::make_vec(2, 4));

        ASSERT_EQ(TestTiling::tile_size(0), 8);
        ASSERT_EQ(TestTiling::tile_size(1), 8);
        ASSERT_EQ(TestTiling::tile_size(2), 1);
        ASSERT_EQ(TestTiling::tile_size(), kf::make_vec(8, 8));

        const int points[8][2] = {
            {1, 2},
            {3, 2},
            {5, 2},
            {7, 2},
            {1, 6},
            {3, 6},
            {5, 6},
            {7, 6},
        };

        size_t counter = 0;
        KERNEL_FLOAT_TILING_FOR(tiling, auto point) {
            ASSERT(counter < 8);
            ASSERT_EQ(point[0], points[counter][0]);
            ASSERT_EQ(point[1], points[counter][1]);
            counter++;
        }

        ASSERT(counter == 8);

        counter = 0;
        KERNEL_FLOAT_TILING_FOR(tiling, int i, auto point) {
            ASSERT(counter < 8);
            ASSERT_EQ(counter, size_t(i));
            ASSERT_EQ(point[0], points[i][0]);
            ASSERT_EQ(point[1], points[i][1]);
            counter++;
        }

        ASSERT(counter == 8);
    }
};

REGISTER_TEST_CASE("basic tiling tests", basic_tiling_test, int)
