#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

#include <float.h>

#include "math.h"
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "Softmax.h"

namespace cuBERT {

    template <typename T>
    struct exp_functor {
        __device__ T operator()(const T& x) const {
            return T(__expf((float) x));
        }
    };

    template <typename T>
    __global__ void kernel_max_cub(const T *__restrict__ in,
                                   const int batch_size,
                                   const int channel,
                                   T *max_out) {
        __shared__ typename cub::BlockReduce<float, 128>::TempStorage temp_storage;
        for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
            float val = -FLT_MAX;
            for (int j = threadIdx.x; j < channel; j += blockDim.x) {
                val = CUB_MAX((float) __ldg(in + i * channel + j), val);
            }
            val = cub::BlockReduce<float, 128>(temp_storage).Reduce(val, cub::Max());
            if (threadIdx.x == 0) {
                max_out[i] = val;
            }
            __syncthreads();
        }
    }

    template <typename T>
    __global__ void kernel_substract_(T *inout, const int batch_size, const int channel, T *max_in) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        inout[idx] = (float) __ldg(inout + idx) - (float) __ldg(max_in + batch_idx);
    }

    template <typename T>
    __global__ void kernel_sum_cub(const T *__restrict__ in,
                                   const int batch_size,
                                   const int channel,
                                   T *sum_out) {
        __shared__ typename cub::BlockReduce<float , 128>::TempStorage s_storage;
        for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
            float s_val = 0.f;
            for (int j = threadIdx.x; j < channel; j += blockDim.x) {
                s_val += (float) __ldg(in + i * channel + j);
            }
            s_val = cub::BlockReduce<float, 128>(s_storage).Sum(s_val);
            if (threadIdx.x == 0) {
                sum_out[i] = s_val;
            }
            __syncthreads();
        }
    }

    template <typename T>
    __global__ void kernel_scale_(T *inout, const int batch_size, const int channel, T *sum_in) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * channel) {
            return;
        }

        int batch_idx = idx / channel;
        inout[idx] = (float) __ldg(inout + idx) / (float) __ldg(sum_in + batch_idx);
    }

    template <typename T>
    __host__ void softmax_(T *inout, const int batch_size, const int channel, T *sum_gpu, void* stream) {
        const int all_blocks = (batch_size * channel + 127) / 128;

        kernel_max_cub<T> <<<batch_size, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);
        kernel_substract_<T> <<<all_blocks, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);

        thrust::device_ptr<T> dev_ptr(inout);
        thrust::transform(thrust::cuda::par.on((cudaStream_t) stream), dev_ptr, dev_ptr + batch_size * channel, dev_ptr, exp_functor<T>());

        kernel_sum_cub<T> <<<batch_size, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);
        kernel_scale_<T> <<<all_blocks, 128, 0, (cudaStream_t) stream>>> (inout, batch_size, channel, sum_gpu);
    }

    template
    __host__ void softmax_<float>(float *inout, const int batch_size, const int channel, float *sum_gpu, void *stream);

    template
    __host__ void softmax_<half>(half *inout, const int batch_size, const int channel, half *sum_gpu, void *stream);
}
