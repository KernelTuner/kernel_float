#pragma once

#include <stdint.h>
#include <math.h>
#include <tgmath.h>

#include "catch2/catch_all.hpp"
#include "kernel_float.h"

namespace kf = kernel_float;

#if KERNEL_FLOAT_IS_HIP
#define cudaError_t           hipError_t
#define cudaSuccess           hipSuccess
#define cudaGetErrorString    hipGetErrorString
#define cudaGetLastError      hipGetLastError
#define cudaSetDevice         hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize

using __nv_bfloat16 = __hip_bfloat16;
#endif

namespace detail {
#if KERNEL_FLOAT_IS_CUDA
__attribute__((noinline)) static __host__ __device__ void
__assertion_failed(const char* expr, const char* file, int line) {
#if KERNEL_FLOAT_IS_HOST
    std::string msg =
        "assertion failed: " + std::string(expr) + " (" + file + ":" + std::to_string(line) + ")";
    throw std::runtime_error(msg);
#else
    printf("assertion failed: %s (%s:%d)\n", expr, file, line);
    asm("trap;");
    while (1)
        ;
#endif
}

#elif KERNEL_FLOAT_IS_HIP
__attribute__((noinline)) static __host__ void
__assertion_failed(const char* expr, const char* file, int line) {
    std::string msg =
        "assertion failed: " + std::string(expr) + " (" + file + ":" + std::to_string(line) + ")";
    throw std::runtime_error(msg);
}

__attribute__((noinline)) static __device__ void
__assertion_failed(const char* expr, const char* file, int line) {
    printf("assertion failed: %s (%s:%d)\n", expr, file, line);
    __builtin_trap();
    while (1)
        ;
}
#endif
}  // namespace detail

#define ASSERT(...)                                                         \
    do {                                                                    \
        bool __result = (__VA_ARGS__);                                      \
        if (!__result) {                                                    \
            ::detail::__assertion_failed(#__VA_ARGS__, __FILE__, __LINE__); \
        }                                                                   \
    } while (0)

#define ASSERT_EQ(A, B)     ASSERT(equals(A, B))
#define ASSERT_APPROX(A, B) ASSERT(approx(A, B))

#define ASSERT_ALL(E)           ASSERT((E) && ...)
#define ASSERT_EQ_ALL(A, B)     ASSERT_ALL(equals(A, B))
#define ASSERT_APPROX_ALL(A, B) ASSERT_ALL(approx(A, B))

namespace detail {
template<typename T>
struct equals_helper {
    static __host__ __device__ bool call(const T& left, const T& right) {
        return left == right;
    }
};

template<>
struct equals_helper<double> {
    static __host__ __device__ bool call(const double& left, const double& right) {
        return (std::isnan(left) && std::isnan(right)) || (left == right);
    }
};

template<>
struct equals_helper<float> {
    static __host__ __device__ bool call(const float& left, const float& right) {
        return (std::isnan(left) && std::isnan(right)) || (left == right);
    }
};

template<>
struct equals_helper<__half> {
    static __host__ __device__ bool call(const __half& left, const __half& right) {
        return equals_helper<float>::call(__half2float(left), __half2float(right));
    }
};

template<>
struct equals_helper<__nv_bfloat16> {
    static __host__ __device__ bool call(const __nv_bfloat16& left, const __nv_bfloat16& right) {
        return equals_helper<float>::call(__bfloat162float(left), __bfloat162float(right));
    }
};

template<typename T, size_t N>
struct equals_helper<kf::vec<T, N>> {
    static __host__ __device__ bool call(const kf::vec<T, N>& left, const kf::vec<T, N>& right) {
        for (int i = 0; i < N; i++) {
            if (!equals_helper<T>::call(left[i], right[i])) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace detail

template<typename T>
__host__ __device__ bool equals(const T& left, const T& right) {
    return detail::equals_helper<T>::call(left, right);
}

namespace detail {
template<typename T>
struct approx_helper {
    static __host__ __device__ bool call(const T& left, const T& right) {
        return equals_helper<T>::call(left, right);
    }
};

template<>
struct approx_helper<double> {
    static __host__ __device__ bool call(double left, double right, double threshold = 1e-8) {
        return equals_helper<double>::call(left, right)
            || ::fabs(left - right) < threshold * ::fabs(left);
    }
};

template<>
struct approx_helper<float> {
    static __host__ __device__ bool call(float left, float right) {
        return approx_helper<double>::call(double(left), double(right), 1e-4);
    }
};

template<>
struct approx_helper<__half> {
    static __host__ __device__ bool call(__half left, __half right) {
        return approx_helper<double>::call(__half2float(left), __half2float(right), 0.01);
    }
};

template<>
struct approx_helper<__nv_bfloat16> {
    static __host__ __device__ bool call(__nv_bfloat16 left, __nv_bfloat16 right) {
        return approx_helper<double>::call(__bfloat162float(left), __bfloat162float(right), 0.05);
    }
};
}  // namespace detail

template<typename T>
__host__ __device__ bool approx(const T& left, const T& right) {
    return detail::approx_helper<T>::call(left, right);
}

namespace detail {
template<typename T, typename... Us>
struct is_one_of_helper;

template<typename T>
struct is_one_of_helper<T>: std::false_type {};

template<typename T, typename... Us>
struct is_one_of_helper<T, T, Us...>: std::true_type {};

template<typename T, typename U, typename... Us>
struct is_one_of_helper<T, U, Us...>: is_one_of_helper<T, Us...> {};
}  // namespace detail

template<typename T, typename... Us>
static constexpr bool is_one_of = detail::is_one_of_helper<T, Us...>::value;

template<typename T, typename... Us>
static constexpr bool is_none_of = !detail::is_one_of_helper<T, Us...>::value;

namespace detail {
template<typename T, typename = void>
struct generator_value;

template<>
struct generator_value<bool> {
    static __host__ __device__ bool call(uint64_t bits) {
        return bool(bits % 2);
    }
};

template<typename T>
struct generator_value<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
    static constexpr T min_value = std::numeric_limits<T>::min();
    static constexpr T max_value = std::numeric_limits<T>::max();

    static __host__ __device__ T call(uint64_t bits) {
        if ((bits & 0xf) == 0xa) {
            return T(0);
        } else if ((bits & 0xf) == 0xb) {
            return min_value;
        } else if ((bits & 0xf) == 0xc) {
            return max_value;
        } else {
            return T(bits);
        }
    }
};

template<typename T>
struct generator_value<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static constexpr T max_value = std::numeric_limits<uint64_t>::max();

    __host__ __device__ static T call(uint64_t bits) {
        if ((bits & 0xf) == 0) {
            return T(0) / T(0);  // nan
        } else if ((bits & 0xf) == 1) {
            return T(1) / T(0);  // inf
        } else if ((bits & 0xf) == 2) {
            return -T(1) / T(0);  // +inf
        } else if ((bits & 0xf) == 3) {
            return T(0);
        } else {
            return (T(bits) / T(max_value)) * (bits % 2 ? T(-1) : T(+1));
        }
    }
};

template<>
struct generator_value<__half> {
    __host__ __device__ static __half call(uint64_t seed) {
        return __float2half(generator_value<float>::call(seed));
    }
};

template<>
struct generator_value<__nv_bfloat16> {
    __host__ __device__ static __nv_bfloat16 call(uint64_t seed) {
        return __float2bfloat16(generator_value<float>::call(seed));
    }
};
}  // namespace detail

template<typename T = int>
struct generator {
    __host__ __device__ generator(uint64_t seed = 6364136223846793005ULL) : seed_(seed) {
        next();
    }

    template<typename R = T>
    __host__ __device__ T next(R ignore = {}) {
        seed_ = 6364136223846793005ULL * seed_ + 1442695040888963407ULL;
        return detail::generator_value<T>::call(seed_);
    }

  private:
    uint64_t seed_;
};

template<typename T>
struct type_name {
    static constexpr const char* value = "???";
};

#define DEFINE_TYPE_NAME(T)                      \
    template<>                                   \
    struct type_name<T> {                        \
        static constexpr const char* value = #T; \
    };

DEFINE_TYPE_NAME(bool)
DEFINE_TYPE_NAME(signed char)
DEFINE_TYPE_NAME(char)
DEFINE_TYPE_NAME(short)
DEFINE_TYPE_NAME(int)
DEFINE_TYPE_NAME(long)
DEFINE_TYPE_NAME(long long)
DEFINE_TYPE_NAME(unsigned char)
DEFINE_TYPE_NAME(unsigned short)
DEFINE_TYPE_NAME(unsigned int)
DEFINE_TYPE_NAME(unsigned long)
DEFINE_TYPE_NAME(unsigned long long)
DEFINE_TYPE_NAME(__half)
DEFINE_TYPE_NAME(__nv_bfloat16)
DEFINE_TYPE_NAME(float)
DEFINE_TYPE_NAME(double)

template<typename T>
struct type_sequence {};

template<size_t... Is>
struct size_sequence {};

using default_size_sequence = size_sequence<1, 2, 3, 4, 5, 6, 7, 8>;

namespace detail {
template<typename T, typename F, size_t N, size_t... Ns>
void iterate_sizes(F runner, size_sequence<N, Ns...>) {
    runner.template run<T, N>();
    iterate_sizes<T>(runner, size_sequence<Ns...> {});
}

template<typename T, typename F>
void iterate_sizes(F, size_sequence<>) {}

template<typename F>
struct host_runner {
    F fun;

    host_runner(F fun) : fun(fun) {}

    template<typename T, size_t N>
    void run() {
        for (int seed = 0; seed < 5; seed++) {
            INFO("T=" << type_name<T>::value);
            INFO("N=" << N);
            INFO("seed=" << seed);

            if constexpr (std::is_invocable_v<F>) {
                fun();
            } else if constexpr (std::is_invocable_v<F, generator<T>>) {
                fun(generator<T>(seed));
            } else {
                fun(generator<T>(seed), std::make_index_sequence<N> {});
            }
        }
    }
};

template<typename F, typename... Args>
__global__ void kernel(F fun, Args... args) {
    fun(args...);
}

template<typename F>
struct device_runner {
    F fun;

    device_runner(F fun) : fun(fun) {}

    template<typename T, size_t N>
    void run() {
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL("failed to initialize CUDA device, does this machine have a GPU?");
        }

        for (int seed = 0; seed < 5; seed++) {
            INFO("T=" << type_name<T>::value);
            INFO("N=" << N);
            INFO("seed=" << seed);

            CHECK(cudaDeviceSynchronize() == cudaSuccess);

            if constexpr (std::is_invocable_v<F>) {
                kernel<<<1, 1>>>(fun);
            } else if constexpr (std::is_invocable_v<F, generator<T>>) {
                kernel<<<1, 1>>>(fun, generator<T>(seed));
            } else {
                kernel<<<1, 1>>>(fun, generator<T>(seed), std::make_index_sequence<N> {});
            }

            CHECK(cudaDeviceSynchronize() == cudaSuccess);
        }
    }
};
}  // namespace detail

template<typename F, typename T, size_t... Ns>
void run_tests_host(F fun, type_sequence<T>, size_sequence<Ns...>) {
    detail::iterate_sizes<T>(detail::host_runner<F>(fun), size_sequence<Ns...> {});
}

template<typename F, typename T, size_t... Ns>
void run_tests_device(F fun, type_sequence<T>, size_sequence<Ns...>) {
    detail::iterate_sizes<T>(detail::device_runner<F>(fun), size_sequence<Ns...> {});
}

#define REGISTER_TEST_CASE_CPU(NAME, F, ...)                                        \
    TEMPLATE_TEST_CASE(NAME " - CPU", "", __VA_ARGS__) {                            \
        run_tests_host(F {}, type_sequence<TestType> {}, default_size_sequence {}); \
        CHECK("done");                                                              \
    }

#define REGISTER_TEST_CASE_GPU(NAME, F, ...)                                          \
    TEMPLATE_TEST_CASE(NAME " - GPU", "[GPU]", __VA_ARGS__) {                         \
        run_tests_device(F {}, type_sequence<TestType> {}, default_size_sequence {}); \
        CHECK("done");                                                                \
    }

#undef REGISTER_TEST_CASE
#define REGISTER_TEST_CASE(NAME, F, ...)         \
    REGISTER_TEST_CASE_CPU(NAME, F, __VA_ARGS__) \
    REGISTER_TEST_CASE_GPU(NAME, F, __VA_ARGS__)
