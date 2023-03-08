#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <random>
#include <string>

#include "catch2/catch_all.hpp"
#include "kernel_float.h"

#define ASSERT(expr) check_assertions((expr), #expr, __FILE__, __LINE__);

static __host__ __device__ int
check_assertions(bool result, const char* expr, const char* file, int line) {
    if (result)
        return 0;

#ifndef __CUDA_ARCH__
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

template<typename... Ts>
__host__ __device__ void ignore(Ts...) {}

template<typename T>
__host__ __device__ bool bitwise_equal(T left, T right) {
    union {
        T item;
        char bytes[sizeof(T)];
    } a, b;

    a.item = left;
    b.item = right;

    for (int i = 0; i < sizeof(T); i++) {
        if (a.bytes[i] != b.bytes[i]) {
            for (int j = 0; j < sizeof(T); j++) {
                printf("byte %d] %d != %d\n", j, a.bytes[j], b.bytes[j]);
            }
            return false;
        }
    }

    return true;
}

template<typename... Ts>
struct type_sequence {};

template<size_t... Is>
struct size_sequence {};

template<typename T>
struct type_name {};
#define DEFINE_TYPE_NAME(T)                      \
    template<>                                   \
    struct type_name<T> {                        \
        static constexpr const char* value = #T; \
    };

DEFINE_TYPE_NAME(bool)
DEFINE_TYPE_NAME(char)
DEFINE_TYPE_NAME(short)
DEFINE_TYPE_NAME(int)
DEFINE_TYPE_NAME(unsigned int)
DEFINE_TYPE_NAME(long)
DEFINE_TYPE_NAME(unsigned long)
DEFINE_TYPE_NAME(long long)
DEFINE_TYPE_NAME(__half)
DEFINE_TYPE_NAME(__nv_bfloat16)
DEFINE_TYPE_NAME(float)
DEFINE_TYPE_NAME(double)

template<typename T, typename = void>
struct generate_value;

template<>
struct generate_value<bool> {
    __host__ __device__ static bool call(uint64_t value) {
        return bool(value & 0x1);
    }
};

template<typename T>
struct generate_value<
    T,
    typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type> {
    __host__ __device__ static T call(uint64_t value) {
        return T(value);
    }
};

template<typename T>
struct generate_value<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    __host__ __device__ static T call(uint64_t value) {
        if ((value & 0xf) == 0) {
            return T(0) / T(0);  // nan
        } else if ((value & 0xf) == 1) {
            return T(1) / T(0);  // inf
        } else if ((value & 0xf) == 2) {
            return -T(0) / T(0);  // +inf
        } else if ((value & 0xf) == 3) {
            return 0;
        } else {
            return T(value) / T(UINT64_MAX);
        }
    }
};

template<>
struct generate_value<__half> {
    __host__ __device__ static __half call(uint64_t seed) {
        return __half(generate_value<float>::call(seed));
    }
};

template<>
struct generate_value<__nv_bfloat16> {
    __host__ __device__ static __nv_bfloat16 call(uint64_t seed) {
        return __nv_bfloat16(generate_value<float>::call(seed));
    }
};

template<typename T>
struct generator {
    __host__ __device__ generator(uint64_t seed = 6364136223846793005ULL) : seed_(seed) {
        next();
    }

    __host__ __device__ T next(uint64_t ignore = 0) {
        seed_ = 6364136223846793005ULL * seed_ + 1442695040888963407ULL;
        return generate_value<T>::call(seed_);
    }

    __host__ __device__ T operator()() {
        return next();
    }

  private:
    uint64_t seed_;
};

template<template<typename, size_t> class F, typename T>
void run_sizes(size_sequence<>) {
    // empty
}

template<template<typename, size_t> class F, typename T, size_t N, size_t... Is, typename... Args>
void run_sizes(size_sequence<N, Is...>, Args... args) {
    //SECTION("size=" + std::to_string(N))
    {
        INFO("N=" << N);
        F<T, N> {}(args...);
    }

    run_sizes<F, T>(size_sequence<Is...> {}, args...);
}

template<
    template<typename, size_t>
    class F,
    typename T,
    typename... Ts,
    size_t... Is,
    typename... Args>
void run_combinations(type_sequence<T, Ts...>, size_sequence<Is...>, Args... args) {
    //SECTION(std::string("type=") + type_name<T>::value)
    {
        INFO("T=" << type_name<T>::value);
        run_sizes<F, T>(size_sequence<Is...> {});
    }

    run_combinations<F>(type_sequence<Ts...> {}, size_sequence<Is...> {}, args...);
}

template<template<typename, size_t> class F, typename... Ts, size_t... Is, typename... Args>
void run_combinations(type_sequence<>, size_sequence<Is...>, Args... args) {}

template<template<typename, size_t> class F, typename T, size_t N>
struct host_runner {
    template<typename... Args>
    void operator()(Args... args) {
        for (size_t i = 0; i < 5; i++) {
            INFO("seed=" << i);
            F<T, N> {}(generator<T>(i), args...);
        }
    }
};

template<template<typename, size_t> class F>
struct host_runner_helper {
    template<typename T, size_t N>
    using type = host_runner<F, T, N>;
};

template<template<typename, size_t> class F, typename... Ts, size_t... Is>
void run_on_host(type_sequence<Ts...>, size_sequence<Is...>) {
    run_combinations<host_runner_helper<F>::template type>(
        type_sequence<Ts...> {},
        size_sequence<Is...> {});
}

template<template<typename, size_t> class F, typename... Ts>
void run_on_host(type_sequence<Ts...> = {}) {
    run_on_host<F>(type_sequence<Ts...> {}, size_sequence<1, 2, 3, 4, 5, 6, 7, 8> {});
}

template<typename F, typename... Args>
__global__ void kernel(F fun, Args... args) {
    fun(args...);
}

template<template<typename, size_t> class F, typename T, size_t N>
struct device_runner {
    template<typename... Args>
    void operator()(Args... args) {
        static bool gpu_enabled = true;
        if (!gpu_enabled) {
            return;
        }

        cudaError_t code = cudaSetDevice(0);
        if (code != cudaSuccess) {
            gpu_enabled = false;
            WARN("skipping device code");
            return;
        }

        //SECTION("environment=GPU")
        {
            for (size_t i = 0; i < 5; i++) {
                INFO("seed=" << i);
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
                kernel<<<1, 1>>>(F<T, N> {}, generator<T>(i), args...);
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
            }
        }
    }
};

template<template<typename, size_t> class F>
struct device_runner_helper {
    template<typename T, size_t N>
    using type = device_runner<F, T, N>;
};

template<template<typename, size_t> class F, typename... Ts, size_t... Is>
void run_on_device(type_sequence<Ts...>, size_sequence<Is...>) {
    run_combinations<device_runner_helper<F>::template type>(
        type_sequence<Ts...> {},
        size_sequence<Is...> {});
}

template<template<typename, size_t> class F, typename... Ts>
void run_on_device(type_sequence<Ts...> = {}) {
    run_on_device<F>(type_sequence<Ts...> {}, size_sequence<1, 2, 3, 4, 5, 6, 7, 8> {});
}

template<template<typename, size_t> class F, typename... Ts>
void run_on_host_and_device(type_sequence<Ts...> = {}) {
    run_on_host<F>(type_sequence<Ts...> {});
    run_on_device<F>(type_sequence<Ts...> {});
}
