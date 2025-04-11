#include <stdio.h>
#include <stdlib.h>

#include "../hip_compat.h"
#include "kernel_float.h"

#define CUDA_CHECK(call)                                     \
    do {                                                     \
        cudaError_t __err = call;                            \
        if (__err != cudaSuccess) {                          \
            fprintf(                                         \
                stderr,                                      \
                "CUDA error at %s:%d (%s): %s (code %d) \n", \
                __FILE__,                                    \
                __LINE__,                                    \
                #call,                                       \
                cudaGetErrorString(__err),                   \
                __err);                                      \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

// Alias `kernel_float` as `kf`
namespace kf = kernel_float;

// Define the float type and vector size
using float_type = float;
static constexpr int VECTOR_SIZE = 4;

__global__ void calculate_pi_kernel(int nx, int ny, int* global_count) {
    // Calculate the global x and y indices for this thread within the grid
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the x and y coordinates as integers.
    // The x coordinates are: [thread_x * VECTOR_SIZE, thread_x * VECTOR_SIZE + 1, ...]
    // The y coordinates are: [thread_y,               thread_y,                   ...]
    kf::vec<int, VECTOR_SIZE> xi = thread_x * VECTOR_SIZE + kf::range<int, VECTOR_SIZE>();
    kf::vec<int, VECTOR_SIZE> yi = thread_y;

    // Normalize the integers to values between 0 and 1.
    kf::vec<float_type, VECTOR_SIZE> xf = kf::cast<float_type>(xi) / float_type(nx);
    kf::vec<float_type, VECTOR_SIZE> yf = kf::cast<float_type>(yi) / float_type(ny);

    // Compute the squared distance to the origin and then take the
    // square root to get the distance to the origin.
    kf::vec<float_type, VECTOR_SIZE> dist_squared = xf * xf + yf * yf;
    kf::vec<float_type, VECTOR_SIZE> dist = kf::sqrt(dist_squared);

    // Count the number of points within the unit circle.
    // The expression `dist <= 1` returns a boolean vector
    // and `kf::count` counts how many elements are `true`.
    int n = kf::count(dist <= float_type(1));

    // Atomically add 'n' to 'global_count'
    atomicAdd(global_count, n);
}

double calculate_pi(int nx, int ny) {
    // Allocate memory on the device (GPU) for 'global_count' to accumulate the count of points inside the circle
    int* d_global_count;
    CUDA_CHECK(cudaMalloc(&d_global_count, sizeof(int)));

    // Initialize the device memory to zero
    CUDA_CHECK(cudaMemset(d_global_count, 0, sizeof(int)));

    // Each thread processes 'VECTOR_SIZE' points in the x-direction
    int num_threads_x = (nx + VECTOR_SIZE - 1) / VECTOR_SIZE;

    // Define the dimensions of each thread block (number of threads per block)
    dim3 block_size(16, 16);  // Each block contains 16 threads in x and y directions

    // Calculate the number of blocks needed in the grid to cover all threads
    dim3 grid_size(
        (num_threads_x + block_size.x - 1) / block_size.x,  // Number of blocks in x-direction
        (ny + block_size.y - 1) / block_size.y  // Number of blocks in y-direction
    );

    // Launch the kernel on the GPU with the calculated grid and block dimensions
    calculate_pi_kernel<<<grid_size, block_size>>>(nx, ny, d_global_count);

    // Check for any errors during kernel launch (asynchronous)
    CUDA_CHECK(cudaGetLastError());

    // Wait for the kernel to finish executing and check for errors (synchronization point)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from device memory back to host memory
    int h_global_count = 0;  // Host variable to store the count
    CUDA_CHECK(cudaMemcpy(&h_global_count, d_global_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Free the allocated device memory
    CUDA_CHECK(cudaFree(d_global_count));

    // Calculate the estimated value of Pi using the ratio of points inside the circle to the total points
    int total_points = nx * ny;
    double pi_estimate = 4.0 * (double(h_global_count) / total_points);

    return pi_estimate;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    for (int n = 1; n <= 16384; n *= 2) {
        double pi = calculate_pi(n, n);

        printf("nx=%d ny=%d pi=%f\n", n, n, pi);
    }

    return EXIT_SUCCESS;
}
