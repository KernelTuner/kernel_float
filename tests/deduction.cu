#include <type_traits>

#include "common.h"
#include "kernel_float/prelude.h"

// Class template argument deduction. These tests run on host only: a bug in clang (HIP)
// treats deduction guides as host functions, so CTAD cannot be used in device code.
// See https://github.com/llvm/llvm-project/issues/146646 (fixed in LLVM 22 by
// https://github.com/llvm/llvm-project/pull/170481).

namespace deduction_tests {
using namespace kernel_float::prelude;

struct packed3 {
    float x, y, z;  // sizeof(packed3) != alignof(packed3)
};

TEST_CASE("deduction guides") {
    float* fp = nullptr;
    const packed3* cp = nullptr;

    kf::vector v(1, 2.0f);
    STATIC_REQUIRE(std::is_same_v<decltype(v), kf::vec<float, 2>>);

    kf::vector_ptr p(fp);
    STATIC_REQUIRE(std::is_same_v<decltype(p), kf::vector_ptr<float, 1, kf::access_policy<float>>>);

    kf::vector_ptr q(cp);
    STATIC_REQUIRE(
        std::is_same_v<decltype(q), kf::vector_ptr<packed3, 1, kf::access_policy<const packed3>>>);

    kf::constant c(2.0);
    STATIC_REQUIRE(std::is_same_v<decltype(c), kf::constant<double>>);
    CHECK(c.get() == 2.0);
}

// From C++20, the compiler derives deduction guides for alias templates from those of the class
// template they name.
#if __cplusplus >= 202002L
TEST_CASE("deduction guides for alias templates") {
    kf::scalar s(1.5);
    STATIC_REQUIRE(std::is_same_v<decltype(s), kf::scalar<double>>);

    kscalar ks(1.5f);
    STATIC_REQUIRE(std::is_same_v<decltype(ks), kscalar<float>>);

    kconstant kc(2.0);
    STATIC_REQUIRE(std::is_same_v<decltype(kc), kconstant<double>>);
}
#endif
}  // namespace deduction_tests
