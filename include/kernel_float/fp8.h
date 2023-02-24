#ifndef KERNEL_FLOAT_FP8_H
#define KERNEL_FLOAT_FP8_H

#include "macros.h"

#if KERNEL_FLOAT_BF8_AVAILABLE

#include "bf16.h"
#include "fp16.h"

namespace kernel_float {
using float8_e4m3 = __nv_fp8_e4m3;
using float8_e5m2 = __nv_fp8_e5m2;

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float16, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(bfloat16, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float8_e4m3)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float8_e4m3)

KERNEL_FLOAT_DEFINE_COMMON_TYPE(float16, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(bfloat16, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float32, float8_e5m2)
KERNEL_FLOAT_DEFINE_COMMON_TYPE(float64, float8_e5m2)

namespace detail {
template<>
struct vec_storage<float8_e4m3, 2> {
    vec_storage(__nv_fp8_e4m3 x, float8_e4m3 y) : array {x, y} {}
    vec_storage(__nv_fp8x2_e4m3 v) : storage_(v) {}

    operator __nv_fp8x2_e4m3() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e4m3, 2)

  private:
    union {
        __nv_fp8x2_e4m3 storage_;
        __nv_fp8_e4m3 array_[2];
    }
};

template<>
struct vec_storage<float8_e4m3, 4> {
    vec_storage(__nv_fp8_e4m3 x, float8_e4m3 y, __nv_fp8_e4m3 z, float8_e4m3 w) :
        array {x, y, z, w} {}
    vec_storage(__nv_fp8x4_e4m3 v) : storage_(v) {}

    operator __nv_fp8x4_e4m3() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e4m3, 4)

    KERNEL_FLOAT_INLINE vec_storage<T, 2> get(index_sequence<4, 5>) const {
        return high_;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e4m3, 2> get(index_sequence<0, 1>) const {
        __nv_fp8x2_e4m3 out;
        out.__x = storage_.__x;
        return out;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e4m3, 2> get(index_sequence<2, 3>) const {
        __nv_fp8x2_e4m3 out;
        out.__x = storage_.__x >> 16;
        return out;
    }

  private:
    union {
        __nv_fp8x4_e4m3 storage_;
        __nv_fp8_e4m3 array_[4];
    }
};

template<>
struct vec_storage<float8_e5m2, 2> {
    vec_storage(__nv_fp8_e5m2 x, float8_e5m2 y) : array {x, y} {}
    vec_storage(__nv_fp8x2_e5m2 v) : storage_(v) {}

    operator __nv_fp8x2_e5m2() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e5m2, 2)

  private:
    union {
        __nv_fp8x2_e5m2 storage_;
        __nv_fp8_e5m2 array_[2];
    }
};

template<>
struct vec_storage<float8_e5m2, 4> {
    vec_storage(__nv_fp8_e5m2 x, float8_e5m2 y, __nv_fp8_e5m2 z, float8_e5m2 w) :
        array {x, y, z, w} {}
    vec_storage(__nv_fp8x4_e5m2 v) : storage_(v) {}

    operator __nv_fp8x4_e5m2() const {
        return storage_;
    }

    KERNEL_FLOAT_STORAGE_ACCESSORS_ARRAY(array_, float8_e5m2, 4)

    KERNEL_FLOAT_INLINE vec_storage<float8_e5m2, 2> get(index_sequence<0, 1>) const {
        __nv_fp8x2_e5m2 out;
        out.__x = (storage_.__x) & 0xffff;
        return out;
    }

    KERNEL_FLOAT_INLINE vec_storage<float8_e5m2, 2> get(index_sequence<2, 3>) const {
        __nv_fp8x2_e5m2 out;
        out.__x = (storage_.__x >> 16) & 0xffff;
        return out;
    }

  private:
    union {
        __nv_fp8x4_e5m2 storage_;
        __nv_fp8_e5m2 array_[4];
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e4m3, T>, float8_e4m3, 2> {
    KERNEL_FLOAT_INLINE static vec<T, 2> call(ops::cast<float8_e4m3, T>, __nv_fp8x2_e4m3 input) {
        return cast<T>(vec<half, 2>(__half2(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<T, float8_e4m3>, T, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e4m3, 2>
    call(ops::cast<T, float8_e4m3>, vec<T, 2> input) {
        return __nv_fp8x2_e4m3(__half2(cast<half, 2>(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e5m2, T>, float8_e5m2, 2> {
    KERNEL_FLOAT_INLINEstatic vec<T, 2> call(ops::cast<float8_e5m2, T>, __nv_fp8x2_e5m2 input) {
        return cast<T>(vec<half, 2>(__half2(input)));
    }
};

template<typename T>
struct map_helper<ops::cast<T, float8_e5m2>, T, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e5m2, 2>
    call(ops::cast<T, float8_e5m2>, vec<T, 2> input) {
        return __nv_fp8x2_e5m2(__half2(cast<half, 2>(input)));
    }
};

namespace ops {
struct cast<float8_e4m3, float8_e5m2> {
    KERNEL_FLOAT_INLINE float8_e5m2 operator()(float8_e4m3 v) const {
        return float8_e5m2(__half(v));
    }
};

struct cast<float8_e5m2, float8_e4m3> {
    KERNEL_FLOAT_INLINE float8_e4m3 operator()(float8_e5m2 v) const {
        return float8_e4m3(__half(v));
    }
};
}  // namespace ops

template<typename T>
struct map_helper<ops::cast<float8_e4m3, float8_e5m2>, float8_e4m3, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e5m2, 2>
    call(ops::cast<float8_e4m3, float8_e5m2>, __nv_fp8x2_e4m3 input) {
        return __nv_fp8x2_e5m2(__half2(input));
    }
};

template<typename T>
struct map_helper<ops::cast<float8_e5m2, float8_e4m3>, float8_e5m2, 2> {
    KERNEL_FLOAT_INLINE static vec<float8_e4m3, 2>
    call(ops::cast<float8_e5m2, float8_e4m3>, __nv_fp8x2_e5m2 input) {
        return __nv_fp8x2_e4m3(__half2(input));
    }
};

}  // namespace detail
}  // namespace kernel_float

#endif
#endif  //KERNEL_FLOAT_FP8_H
