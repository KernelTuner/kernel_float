#include <type_traits>

#include "common.h"
#include "kernel_float/prelude.h"

// Class template argument deduction. These checks are compile-time only: if this file compiles,
// they pass. The file is valid as both C++17 and C++20.

namespace deduction_tests {
using namespace kernel_float::prelude;

struct packed3 {
    float x, y, z;  // sizeof(packed3) != alignof(packed3)
};

__host__ __device__ void class_templates(float* fp, const packed3* cp) {
    kf::vector v(1, 2.0f);
    static_assert(std::is_same<decltype(v), kf::vec<float, 2>>::value, "");

    kf::vector_ptr p(fp);
    static_assert(
        std::is_same<decltype(p), kf::vector_ptr<float, 1, kf::access_policy<float>>>::value,
        "");

    kf::vector_ptr q(cp);
    using expected_q = kf::vector_ptr<packed3, 1, kf::access_policy<const packed3>>;
    static_assert(std::is_same<decltype(q), expected_q>::value, "");

    kf::constant c(2.0);
    static_assert(std::is_same<decltype(c), kf::constant<double>>::value, "");
}

// From C++20, the compiler derives deduction guides for alias templates from those of the class
// template they name.
#if __cplusplus >= 202002L
__host__ __device__ void alias_templates() {
    kf::scalar s(1.5);
    static_assert(std::is_same<decltype(s), kf::scalar<double>>::value, "");

    kscalar ks(1.5f);
    static_assert(std::is_same<decltype(ks), kscalar<float>>::value, "");

    kconstant kc(2.0);
    static_assert(std::is_same<decltype(kc), kconstant<double>>::value, "");
}
#endif
}  // namespace deduction_tests
