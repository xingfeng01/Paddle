/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/for_range.h"

#include "paddle/fluid/platform/cuda_device_function.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

template <typename T>
__device__ __forceinline__ T logT(T x) {
  return std::log(x);
};

template <>
__device__ __forceinline__ paddle::platform::float16 logT(
    paddle::platform::float16 x) {
  return (paddle::platform::float16)std::log((float)x);
};

int inline log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
};

template <typename T, int BATCH_SIZE, int warp_size>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BATCH_SIZE, int warp_size>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T>
__global__ void CrossEntropyHardLabel(T* loss, const T* softmax,
                                      const int64_t* labels, const int n,
                                      const int dim, const int d,
                                      const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;
  if (ids < n * d) {
    int64_t idx = idx_n * dim * d + labels[ids] * d + idx_d;
    if (labels[ids] == ignore_idx) {
      loss[ids] = (T)0.0;
    } else {
      loss[ids] = -logT(softmax[idx]);
    };
  };
};

template <typename T>
__global__ void CrossEntropySoftLabel(T* loss, const T* softmax,
                                      const T* labels, const int n,
                                      const int dim, const int d) {
  const int warp_size = 32;
  const int BATCH_SIZE = 1;
  int ITERATIONS = (dim + warp_size - 1) / warp_size;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;
  int local_batches = n * d - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  T sum[BATCH_SIZE]{(T)0.0};
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (i >= local_batches) break;
    for (int it = 0; it < ITERATIONS; it++) {
      int ids = first_batch + i;
      int idx_n = ids / d;
      int idx_d = ids % d;
      int idx_dim = it * warp_size + threadIdx.x;
      int idx = idx_n * dim * d + idx_d + idx_dim * d;
      if (idx_dim < dim) {
        sum[i] -= logT(softmax[idx]) * labels[idx];
      }
    };
  };
  WarpReduceSum<T, BATCH_SIZE, warp_size>(sum);

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < BATCH_SIZE; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sum[0];
      }
    }
  }
};

template <typename T, typename VECT, typename ACCT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoftLabel(T* loss, T* softmax, const T* src,
                                            const T* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count) {
  const bool isLog = true;

  constexpr int dim_ceil = 1 << Log2Elements;
  constexpr int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
  constexpr int VSIZE = sizeof(VECT) / sizeof(T);
  constexpr int ITERATIONS = dim_ceil / warp_size;
  constexpr int ITERATIONS_V = (ITERATIONS >= VSIZE) ? (ITERATIONS / VSIZE) : 1;
  constexpr int BATCH_SIZE = (dim_ceil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;
  int local_batches = batch_size - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  // read data from global memory
  VECT srcdata[BATCH_SIZE][ITERATIONS_V];
  VECT labeldata[BATCH_SIZE][ITERATIONS_V];

  for (int i = 0; i < BATCH_SIZE; ++i) {
    const VECT* src_v =
        reinterpret_cast<const VECT*>(&src[(first_batch + i) * stride]);
    const VECT* label_v =
        reinterpret_cast<const VECT*>(&label[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / VSIZE;

    // read data
    for (int it = 0; it < ITERATIONS_V; ++it) {
      int src_idx = threadIdx.x + it * warp_size;
      if (src_idx < idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
        labeldata[i][it] = label_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < VSIZE; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<ACCT>::max();
          reinterpret_cast<T*>(&labeldata[i][it])[s] = 0.0;
        };
      };
    };
  };

  // compute max value
  ACCT max_value[BATCH_SIZE];
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    max_value[i] = -std::numeric_limits<ACCT>::infinity();
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = valvp[0];
#pragma unroll
      for (int s = 1; s < VSIZE; ++s) {
        valmax = (valmax > valvp[s]) ? valmax : valvp[s];
      }
      max_value[i] =
          (max_value[i] > (ACCT)valmax) ? max_value[i] : (ACCT)valmax;
    };
  };
  WarpReduceMax<ACCT, BATCH_SIZE, warp_size>(max_value);

  // compute sum
  ACCT sum[BATCH_SIZE]{0.0};
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvp = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          sum[i] += std::exp((ACCT)valvp[s] - max_value[i]);
        } else {
          valvp[s] = std::exp((ACCT)valvp[s] - max_value[i]);
          sum[i] += (ACCT)valvp[s];
        }
      };
    }
  }
  WarpReduceSum<ACCT, BATCH_SIZE, warp_size>(sum);

  // log_softmax and loss
  ACCT sumloss[BATCH_SIZE]{0.0};
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (i >= local_batches) break;

    VECT* softmax_v =
        reinterpret_cast<VECT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / VSIZE;

    if (isLog) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T* labelvp = reinterpret_cast<T*>(&labeldata[i][it]);
      VECT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          ACCT logsoftmax = (ACCT)srcvp[s] - max_value[i] - sum[i];
          sumloss[i] -= logsoftmax * (ACCT)labelvp[s];
          tmpvp[s] = std::exp(logsoftmax);
        } else {
          tmpvp[s] = (ACCT)srcvp[s] / sum[i];
        };
      };

      int idx = threadIdx.x + it * warp_size;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      };
    };
  };

  // loss
  WarpReduceSum<ACCT, BATCH_SIZE, warp_size>(sumloss);

  for (int i = 0; i < BATCH_SIZE; i++) {
    if (i >= local_batches) break;
    loss[first_batch + i] = sumloss[i];
  };
};

template <typename T, typename VECT, typename ACCT, int Log2Elements>
__global__ void WarpSoftmaxForwardHardLabel(T* loss, T* softmax, const T* src,
                                            const int64_t* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count,
                                            const int ignore_index) {
  const bool isLog = true;

  constexpr int dim_ceil = 1 << Log2Elements;
  constexpr int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
  constexpr int VSIZE = sizeof(VECT) / sizeof(T);
  constexpr int ITERATIONS = dim_ceil / warp_size;
  constexpr int ITERATIONS_V = (ITERATIONS >= VSIZE) ? (ITERATIONS / VSIZE) : 1;
  constexpr int BATCH_SIZE = (dim_ceil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;
  int local_batches = batch_size - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  // read data from global memory
  VECT srcdata[BATCH_SIZE][ITERATIONS_V];
  for (int i = 0; i < BATCH_SIZE; ++i) {
    const VECT* src_v =
        reinterpret_cast<const VECT*>(&src[(first_batch + i) * stride]);

    // max index to read
    int src_idx_max = (i < local_batches) ? element_count : 0;
    int src_idx_max_v = src_idx_max / VSIZE;

    // read data
    for (int it = 0; it < ITERATIONS_V; ++it) {
      int src_idx = threadIdx.x + it * warp_size;
      if (src_idx < src_idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < VSIZE; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<ACCT>::infinity();
        };
      };
    };
  };

  // compute max value
  ACCT max_value[BATCH_SIZE];
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    max_value[i] = -std::numeric_limits<ACCT>::infinity();
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = valvp[0];
#pragma unroll
      for (int s = 1; s < VSIZE; ++s) {
        valmax = (valmax > valvp[s]) ? valmax : valvp[s];
      }
      max_value[i] =
          (max_value[i] > (ACCT)valmax) ? max_value[i] : (ACCT)valmax;
    };
  };
  WarpReduceMax<ACCT, BATCH_SIZE, warp_size>(max_value);

  // compute sum
  ACCT sum[BATCH_SIZE]{0.0};
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvp = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          sum[i] += std::exp((ACCT)valvp[s] - max_value[i]);
        } else {
          valvp[s] = std::exp((ACCT)valvp[s] - max_value[i]);
          sum[i] += (ACCT)valvp[s];
        }
      };
    }
  }
  WarpReduceSum<ACCT, BATCH_SIZE, warp_size>(sum);

  // log_softmax
  VECT* softmax_v = reinterpret_cast<VECT*>(softmax);
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (i >= local_batches) break;

    VECT* softmax_v =
        reinterpret_cast<VECT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / VSIZE;

    if (isLog) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      VECT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          ACCT logsoftmax = (ACCT)srcvp[s] - max_value[i] - sum[i];
          tmpvp[s] = std::exp(logsoftmax);

          int loss_idx = (threadIdx.x + it * warp_size) * VSIZE + s;
          if (label[first_batch + i] == loss_idx) {
            if (label[first_batch + i] != ignore_index) {
              loss[first_batch + i] = -logsoftmax;
            } else {
              loss[first_batch + i] = (T)0.0;
            }
          }
        } else {
          tmpvp[s] = (ACCT)srcvp[s] / sum[i];
        };
      };

      int idx = threadIdx.x + it * warp_size;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      };
    };
  };
};

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VECT, ACCT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardSoftLabel<T, VECT, ACCT,                                 \
                                Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count);         \
    break;

template <typename T>
void SwitchWarpSoftmaxForwardSoftLabel(const int blocks, const dim3 threads,
                                       gpuStream_t stream, T* loss, T* softmax,
                                       const T* src, const T* label,
                                       const int batch_size, const int stride,
                                       const int element_count,
                                       const int Log2Elements) {
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, T, T);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxForwardSoftLabel<paddle::platform::float16>(
    const int blocks, const dim3 threads, gpuStream_t stream,
    paddle::platform::float16* loss, paddle::platform::float16* softmax,
    const paddle::platform::float16* src,
    const paddle::platform::float16* label, const int batch_size,
    const int stride, const int element_count, int Log2Elements) {
#define T paddle::platform::float16
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
};

#undef SOFTMAX_WARP_FORWARD_SOFT_CASE

#define SOFTMAX_WARP_FORWARD_HARD_CASE(Log2Elements, VECT, ACCT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardHardLabel<T, VECT, ACCT,                                 \
                                Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count,          \
        ignore_index);                                                         \
    break;

template <typename T>
void SwitchWarpSoftmaxForwardHardLabel(const int blocks, const dim3 threads,
                                       gpuStream_t stream, T* loss, T* softmax,
                                       const T* src, const int64_t* label,
                                       const int batch_size, const int stride,
                                       const int element_count,
                                       int Log2Elements,
                                       const int ignore_index) {
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_HARD_CASE(0, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(1, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(2, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(3, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(4, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(5, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(6, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(7, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(8, T, T);
    SOFTMAX_WARP_FORWARD_HARD_CASE(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxForwardHardLabel<paddle::platform::float16>(
    const int blocks, const dim3 threads, gpuStream_t stream,
    paddle::platform::float16* loss, paddle::platform::float16* softmax,
    const paddle::platform::float16* src, const int64_t* label,
    const int batch_size, const int stride, const int element_count,
    int Log2Elements, const int ignore_index) {
#define T paddle::platform::float16
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_HARD_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_HARD_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
};

template <typename T>
static void SoftmaxWithCrossEntropyHardLabel(
    const framework::ExecutionContext& ctx, int rank, int axis,
    const T* logits_data, const int64_t* labels_data, T* loss_data,
    T* softmax_data, int N, int dim, int D, const int ignore_index) {
  constexpr int kMaxBlockDim = 512;
  int64_t block_dim = dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(dim)));
  int64_t grid_dim = N * D;
  auto stream = ctx.cuda_device_context().stream();

#ifdef __HIPCC__
#define CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)      \
  case BlockDim: {                                                             \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, d* dim, dim);                                \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSum<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, softmax_data, d* dim, dim);                  \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForDiff<T, BlockDim>),      \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, softmax_data, d* dim, dim);                  \
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, n* d* dim); \
    if (ignore_idx >= 0 && ignore_idx < dim) {                                 \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx<T>(       \
          labels_data, loss_data, softmax_data, dim * d, dim, ignore_idx));    \
    } else {                                                                   \
      for_range(HardLabelSoftmaxWithCrossEntropyFunctor<T>(                    \
          labels_data, loss_data, softmax_data, dim * d, dim, ignore_idx));    \
    }                                                                          \
  } break
#endif
#undef CALL_HARD_LABEL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL

  constexpr int max_dim = 320;
  constexpr int warps_per_block = 4;

  if (D == 1 && dim <= max_dim) {
    const int dim_log2 = static_cast<int>(log2_ceil(dim));
    const int dim_ceil = 1 << dim_log2;
    int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
    int batches_per_warp = (dim_ceil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    SwitchWarpSoftmaxForwardHardLabel<T>(
        blocks, threads, ctx.cuda_device_context().stream(), loss_data,
        softmax_data, logits_data, labels_data, N, dim, dim, dim_log2,
        ignore_index);
  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward(
        handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
        platform::CudnnDataType<T>::kZero(), desc_, out_data));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, mode,
        platform::CudnnDataType<T>::kOne(), desc_, logits_data,
        platform::CudnnDataType<T>::kZero(), desc_, softmax_data));
#endif
    int threads = 128;
    int blocks = (N * D + threads - 1) / threads;
    CrossEntropyHardLabel<T><<<blocks, threads>>>(
        loss_data, softmax_data, labels_data, N, dim, D, ignore_index);
  };
};

template <typename T>
static void SoftmaxWithCrossEntropySoftLabel(
    const framework::ExecutionContext& ctx, const int rank, const int axis,
    const T* logits_data, const T* labels_data, T* softmax_data, T* loss_data,
    int N, int dim, int D) {
  constexpr int kMaxBlockDim = 512;
  int64_t block_dim = dim >= kMaxBlockDim
                          ? kMaxBlockDim
                          : (1 << static_cast<int>(std::log2(dim)));

  int64_t grid_dim = N * D;

#ifdef __HIPCC__
#define CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL(BlockDim)                 \
  case BlockDim:                                                               \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, d* dim, dim);                                \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForSum<T, BlockDim>),       \
                       dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, \
                       loss_data, softmax_data, d* dim, dim);                  \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(RowReductionForSoftmaxAndCrossEntropy<T, BlockDim>),   \
        dim3(grid_dim), dim3(BlockDim), 0, stream, logits_data, labels_data,   \
        loss_data, softmax_data, d* dim, dim);                                 \
    break
#endif
#undef CALL_SOFTMAX_WITH_CROSS_ENTROPY_FUSED_KERNEL

  constexpr int max_dim = 320;
  constexpr int warps_per_block = 4;

  if (D == 1 && dim <= max_dim) {
    const int dim_log2 = static_cast<int>(log2_ceil(dim));
    const int dim_ceil = 1 << dim_log2;
    int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
    int batches_per_warp = (dim_ceil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    SwitchWarpSoftmaxForwardSoftLabel<T>(
        blocks, threads, ctx.cuda_device_context().stream(), loss_data,
        softmax_data, logits_data, labels_data, N, dim, dim, dim_log2);

  } else {
    ScopedTensorDescriptor desc;
    std::vector<int> tensor_dims = {N, dim, D, 1};
    DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
    miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
    cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
    auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                 : MIOPEN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxForward(
        handle, platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
        platform::CudnnDataType<T>::kZero(), desc_, out_data));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, mode,
        platform::CudnnDataType<T>::kOne(), desc_, logits_data,
        platform::CudnnDataType<T>::kZero(), desc_, softmax_data));
#endif

    int warp_size = 32;  // (dim < 32) ? dim : 32;
    int batches_per_warp = 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N * D + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    CrossEntropySoftLabel<T><<<blocks, threads>>>(loss_data, softmax_data,
                                                  labels_data, N, dim, D);
  }
}
}
}