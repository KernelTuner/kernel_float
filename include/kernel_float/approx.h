#pragma once

#include "apply.h"
#include "bf16.h"
#include "fp16.h"
#include "macros.h"

namespace kernel_float {

namespace approx {

static_assert(sizeof(unsigned int) * 8 == 32, "invalid size of unsigned int");
static_assert(sizeof(unsigned short) * 8 == 16, "invalid size of unsigned short");
using uint32_t = unsigned int;
using uint16_t = unsigned short;

template<typename T, typename U>
KERNEL_FLOAT_DEVICE T transmute(const U& input) {
    static_assert(sizeof(T) == sizeof(U), "types must have equal size");
    T result {};
    ::memcpy(&result, &input, sizeof(T));
    return result;
}

KERNEL_FLOAT_DEVICE uint32_t
bitwise_if_else(uint32_t condition, uint32_t if_true, uint32_t if_false) {
    uint32_t result;

#if KERNEL_FLOAT_IS_CUDA
    // equivalent to (condition & if_true) | ((~condition) & if_false)
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;"
        : "=r"(result)
        : "r"(condition), "r"(if_true), "r"(if_false));
#else
    result = (condition & if_true) | ((~condition) & if_false);
#endif
    return result;
}

template<typename T, typename T2>
KERNEL_FLOAT_DEVICE T2 eval_poly_recur(T2 y, T2 x) {
    return y;
}

template<typename T, typename T2, typename... TRest>
KERNEL_FLOAT_DEVICE T2 eval_poly_recur(T2 y, T2 x, T coef, TRest... coefs) {
    y = __hfma2(x, y, T2 {coef, coef});
    return eval_poly_recur<T>(y, x, coefs...);
}

template<typename T, typename T2, typename... TRest>
KERNEL_FLOAT_DEVICE T2 eval_poly(T2 x, T coef, TRest... coefs) {
    return eval_poly_recur<T>(T2 {coef, coef}, x, coefs...);
}

#define KERNEL_FLOAT_DEFINE_POLY(NAME, N, ...)     \
    template<typename T>                           \
    struct NAME<T, N> {                            \
        template<typename T2>                      \
        static KERNEL_FLOAT_DEVICE T2 call(T2 x) { \
            return eval_poly<T>(x, __VA_ARGS__);   \
        }                                          \
    };

template<typename T, size_t N>
struct sin_poly: sin_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 1, 1.365)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 2, -21.56, 5.18)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 3, 53.53, -38.06, 6.184)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 4, -56.1, 77.94, -41.1, 6.277)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 5, 32.78, -74.5, 81.4, -41.34, 6.28)

template<typename T, size_t N>
struct cos_poly: cos_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 1, 0.0)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 2, -8.0, 0.6943)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 3, 38.94, -17.5, 0.9707)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 4, -59.66, 61.12, -19.56, 0.9985)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 5, 45.66, -82.4, 64.7, -19.73, 1.0)

template<typename T, size_t N>
struct asin_poly: asin_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 1, 1.531)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 2, -0.169, 1.567)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 3, 0.05167, -0.2057, 1.57)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 4, -0.02103, 0.077, -0.2129, 1.57)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 5, 0.009796, -0.03772, 0.0857, -0.2142, 1.57)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_DEVICE half2_t flipsign(half2_t input, half2_t sign) {
    // Flip signbit of input when sign<0
    uint32_t result;

#if KERNEL_FLOAT_IS_CUDA
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
        : "=r"(result)
        : "r"(0x80008000), "r"(transmute<uint32_t>(sign)), "r"(transmute<uint32_t>(input)));
#else
    result = uint32_t(transmute<uint32_t>(sign) & 0x80008000) ^ transmute<uint32_t>(input);
#endif

    return transmute<half2_t>(result);
}

KERNEL_FLOAT_DEVICE uint32_t half2_gt_mask(half2_t a, half2_t b) {
    uint32_t val;
#if KERNEL_FLOAT_IS_CUDA
    uint32_t ai = *(reinterpret_cast<const uint32_t*>(&a));
    uint32_t bi = *(reinterpret_cast<const uint32_t*>(&b));
    asm("{ set.gt.u32.f16x2 %0,%1,%2;\n}" : "=r"(val) : "r"(ai), "r"(bi));
#else
    val = transmute<uint32_t>(make_short2(a.x > b.x ? ~0 : 0, a.y > b.y ? ~0 : 0));
#endif
    return val;
}

KERNEL_FLOAT_INLINE half2_t make_half2(half x) {
    return {x, x};
}

KERNEL_FLOAT_DEVICE half2_t normalize_trig_input(half2_t x) {
    /* Using rint is too slow. Round using floating-point magic instead. */
    // half2_t x = arg * make_half2(-0.15915494309);
    // return __hfma2(arg, make_half2(0.15915494309),  h2rint(x));

    // 1/(2pi) = 0.15915494309189535
    static constexpr double ONE_OVER_TWOPI = 0.15915494309189535;
    static constexpr double OFFSET = -2042.0;

    half2_t ws = __hfma2(x, make_half2(-ONE_OVER_TWOPI), make_half2(-OFFSET)) + make_half2(OFFSET);
    return __hfma2(x, make_half2(ONE_OVER_TWOPI), ws);
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t cos(half2_t x) {
    half2_t xf = normalize_trig_input(x);
    return cos_poly<half, Iter + 1>::call(__hmul2(xf, xf));
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t sin(half2_t x) {
    half2_t xf = normalize_trig_input(x);
    return sin_poly<half, Iter>::call(__hmul2(xf, xf)) * xf;
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t rcp(half2_t x) {
    // Flip bits
    uint32_t m = ~transmute<uint32_t>(x);

    // Multiply by bias (add contant)
    half2_t y = transmute<half2_t>(uint32_t(0x776d776d) + m);

#pragma unroll
    for (int i = 0; i < Iter; i++) {
        // y += y * (1 - x * y)
        y = __hfma2(y, __hfma2(-x, y, make_half2(1.0)), y);
    }

    return y;
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t rsqrt(half2_t x) {
    // A small number added such that rsqrt(0) does not return NaN
    static constexpr double EPS = 0.00000768899917602539;

    // Set top and bottom bits for both halfs, then shift by 1, then invert
    uint32_t r = ~((uint32_t(transmute<uint32_t>(x) >> 1)) | ~uint32_t(0x3fff3fff));

    // Add bias
    static constexpr uint32_t BIAS = 0x199c199c;
    half2_t y = transmute<half2_t>(uint32_t(r) + BIAS);

    // Newton-Raphson iterations
#pragma unroll
    for (int i = 0; i < Iter; i++) {
        half2_t half_x = __hfma2(make_half2(-0.5), x, make_half2(-EPS));
        half2_t correction = __hfma2(half_x, y * y, make_half2(0.5));
        y = __hfma2(correction, y, y);  // y += y * correction
    }

    return y;
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t sqrt(half2_t x) {
    if (Iter == 1) {
        half2_t y = rsqrt<0>(x);

        // This method uses only 4 muls, instead of 5 muls when using `arg * approx_rsqrt<1>(arg)`
        half2_t xy = x * y;
        return xy * __hfma2(make_half2(-0.5) * y, xy, make_half2(1.5));
    }

    return x * rsqrt<Iter>(x);
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t asin(half2_t x) {
    static constexpr double HALF_PI = 1.57079632679;
    auto abs_x = __habs2(x);
    auto v = asin_poly<half, Iter + 1>::call(abs_x);
    auto abs_y = __hfma2(-v, sqrt<Iter>(make_half2(1) - abs_x), make_half2(HALF_PI));
    return flipsign(abs_y, x);
}

template<int Iter>
KERNEL_FLOAT_DEVICE half2_t acos(half2_t x) {
    static constexpr double HALF_PI = 1.57079632679;
    return make_half2(HALF_PI) - asin<Iter>(x);
}

template<int Deg>
KERNEL_FLOAT_DEVICE half2_t exp(half2_t x) {
    half2_t y;

    if (Deg == 0) {
        // Bring the value to range [32, 64]
        // 1.442 = 1/log(2)
        // 46.969 = 32.5/log(2)
        half2_t m = __hfma2(x, make_half2(1.442), make_half2(46.9375));

        // Transmute to int, shift higher mantissa bits into exponent field.
        y = transmute<half2_t>((transmute<uint32_t>(m) & 0x03ff03ff) << 5);
    } else {
        // Add a large number to round to an integer
        half2_t v = __hfma2(x, make_half2(1.442), make_half2(1231.0));

        // The exponent is now in the lower 5 bits. Shift that into the exponent field.
        half2_t exp = transmute<half2_t>((transmute<uint32_t>(v) & 0x001f001f) << 10);

        // The fractional part can be obtained from "1231-v".
        // 0.6934 = log(2)
        half2_t frac = __hfma2(make_half2(1231.0) - v, make_half2(0.6934), x);

        // This is the Taylor expansion of "exp(x)-1" around 0
        half2_t adjust;
        if (Deg == 1) {
            adjust = frac;
        } else if (Deg == 2) {
            // adjust = frac + 0.5 * frac^2
            adjust = __hfma2(frac, __hmul2(frac, make_half2(0.5)), frac);
        } else /* if (Deg == 2) */ {
            // adjust = frac + 0.5 * frac^2 + 1/6 * frac^3
            adjust = __hfma2(
                frac,
                __hmul2(__hfma2(frac, make_half2(0.1666), make_half2(0.5)), frac),
                frac);
        }

        // result = exp * (adjust + 1)
        y = __hfma2(exp, adjust, exp);
    }

    // Values below -10.39 (= -15*log(2)) become zero
    uint32_t zero_mask = half2_gt_mask(x, make_half2(-10.390625));
    return transmute<half2_t>(zero_mask & transmute<uint32_t>(y));
}

template<int = 0>
KERNEL_FLOAT_DEVICE half2_t log(half2_t arg) {
    // Shift exponent field into mantissa bits. Fill exponent bits with 0x5000 (= 32.0)
    uint32_t bits = bitwise_if_else(0x03ff03ff, transmute<uint32_t>(arg) >> 5, 0x50005000);

    // 0.6934 = log(2)
    // 32.53 = 46.969*log(2)
    return __hfma2(transmute<half2_t>(bits), make_half2(0.6934), make_half2(-32.53125));
}

template<int Deg>
KERNEL_FLOAT_DEVICE half2_t tanh(half2_t x) {
    if (Deg == 0) {
        return x * rcp<0>(make_half2(0.2869) + __habs2(x));
    } else {
        auto c0 = make_half2(0.4531);
        auto c1 = make_half2(0.5156);
        auto x2b = __hfma2(x, x, c1);
        return (x * x2b) * rcp<Deg>(__hfma2(x2b, __habs2(x), c0));
    }
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_DEVICE bfloat16x2_t make_bfloat162(bfloat16_t x) {
    return {x, x};
}

KERNEL_FLOAT_DEVICE bfloat16x2_t make_bfloat162(double x) {
    return {__double2bfloat16(x), __double2bfloat16(x)};
}

KERNEL_FLOAT_DEVICE bfloat16x2_t normalize_trig_input(bfloat16x2_t x) {
    static constexpr double ONE_OVER_TWOPI = 0.15915494309189535;
    static constexpr double OFFSET = -2042.0;

    bfloat16x2_t ws = __hadd2(
        __hfma2(x, make_bfloat162(-ONE_OVER_TWOPI), make_bfloat162(-OFFSET)),
        make_bfloat162(OFFSET));
    return __hfma2(x, make_bfloat162(ONE_OVER_TWOPI), ws);
}

template<int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t cos(bfloat16x2_t x) {
    bfloat16x2_t xf = normalize_trig_input(x);
    return cos_poly<__bfloat16, Iter + 1>::call(__hmul2(xf, xf));
}

template<int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t sin(bfloat16x2_t x) {
    bfloat16x2_t xf = normalize_trig_input(x);
    return __hmul2(sin_poly<__bfloat16, Iter>::call(__hmul2(xf, xf)), xf);
}

template<int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t rcp(bfloat16x2_t x) {
    bfloat16x2_t y = transmute<bfloat16x2_t>(uint32_t(0x7ef07ef0) + ~transmute<uint32_t>(x));

#pragma unroll
    for (int i = 0; i < Iter; i++) {
        y = __hfma2(y, __hfma2(__hneg2(x), y, make_bfloat162(1.0)), y);
    }

    return y;
}

template<int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t rsqrt(bfloat16x2_t x) {
    // Set top and bottom bits for both halfs, then shift by 1, then invert
    uint32_t r = ~((uint32_t(transmute<uint32_t>(x) >> 1)) | ~uint32_t(0x3fff3fff));

    // Add bias (0x1f36)
    bfloat16x2_t y = transmute<bfloat16x2_t>(uint32_t(r) + uint32_t(0x1f361f36));

    // Newton-Raphson iterations
#pragma unroll
    for (int i = 0; i < Iter; i++) {
        bfloat16x2_t half_x = __hmul2(make_bfloat162(-0.5), x);
        bfloat16x2_t correction = __hfma2(half_x, __hmul2(y, y), make_bfloat162(0.5));
        y = __hfma2(correction, y, y);  // y += y * correction
    }

    return y;
}

template<int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t sqrt(bfloat16x2_t x) {
    return __hmul2(x, rsqrt<Iter>(x));
}

template<int = 0>
KERNEL_FLOAT_DEVICE bfloat16x2_t exp(bfloat16x2_t arg) {
    static constexpr float SCALE = 1.44272065994 / 256.0;
    static constexpr float OFFSET = 382.4958400542335;
    static constexpr float MINIMUM = 382;

    float a = fmaxf(fmaf(__bfloat162float(arg.x), SCALE, OFFSET), MINIMUM);
    float b = fmaxf(fmaf(__bfloat162float(arg.y), SCALE, OFFSET), MINIMUM);

    return {
        transmute<bfloat16_t>(uint16_t(transmute<uint32_t>(a))),
        transmute<bfloat16_t>(uint16_t(transmute<uint32_t>(b)))};
}
#endif
}  // namespace approx

namespace detail {
template<int Level, typename F, typename T>
struct apply_impl<approx_level_policy<Level>, F, 1, T, T> {
    KERNEL_FLOAT_INLINE static void call(F fun, T* output, const T* input) {
        T in2[2], out2[2];
        in2[0] = input[0];
        apply_impl<approx_level_policy<Level>, F, 2, T, T>::call(fun, out2, in2);
        output[0] = out2[0];
    }
};
}  // namespace detail

#define KERNEL_FLOAT_DEFINE_APPROX_IMPL(T, FUN, DEFAULT_LEVEL)                         \
    namespace detail {                                                                 \
    template<int Degree>                                                               \
    struct apply_impl<approx_level_policy<Degree>, ops::FUN<T>, 2, T, T> {             \
        KERNEL_FLOAT_INLINE static void call(ops::FUN<T>, T* output, const T* input) { \
            auto res = approx::FUN<Degree>({input[0], input[1]});                      \
            output[0] = res.x;                                                         \
            output[1] = res.y;                                                         \
        }                                                                              \
    };                                                                                 \
                                                                                       \
    template<>                                                                         \
    struct apply_impl<approx_policy, ops::FUN<T>, 2, T, T>:                            \
        apply_impl<approx_level_policy<DEFAULT_LEVEL>, ops::FUN<T>, 2, T, T> {};       \
    }

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, sin, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, cos, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, rsqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, sqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, rcp, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, exp, 0)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, log, 0)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, asin, 2)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, acos, 2)
#endif

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, cos, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, sin, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, rcp, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, rsqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, sqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, exp, 0)
//KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, log, 0)
#endif

#define KERNEL_FLOAT_DEFINE_APPROX_FUN(FUN)                                              \
    template<int Level = -1, typename V>                                                 \
    KERNEL_FLOAT_INLINE into_vector_type<V> approx_##FUN(const V& args) {                \
        return map<approx_level_policy<Level>>(ops::FUN<vector_value_type<V>> {}, args); \
    }

KERNEL_FLOAT_DEFINE_APPROX_FUN(sin)
KERNEL_FLOAT_DEFINE_APPROX_FUN(cos)
KERNEL_FLOAT_DEFINE_APPROX_FUN(rsqrt)
KERNEL_FLOAT_DEFINE_APPROX_FUN(sqrt)
KERNEL_FLOAT_DEFINE_APPROX_FUN(rcp)
KERNEL_FLOAT_DEFINE_APPROX_FUN(exp)
KERNEL_FLOAT_DEFINE_APPROX_FUN(log)

}  // namespace kernel_float
