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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/gpu_launch_config.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

template <typename T>
__device__ __forceinline__ T logT(T x) {
  return std::log(x);
}

template <>
__device__ __forceinline__ paddle::platform::float16 logT(
    paddle::platform::float16 x) {
  return (paddle::platform::float16)std::log((float)x);
}

int inline log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
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
      loss[ids] = static_cast<T>(0.0);
    } else {
      loss[ids] = -logT(softmax[idx]);
    }
  }
}

template <typename T, typename VecT, int Log2Elements>
__global__ void CrossEntropySoftLabel(T* loss, const T* softmax,
                                      const T* labels, const int n,
                                      const int dim, const int d) {
  const int kDimCeil = 1 << Log2Elements;
  const int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  const int kVSize = sizeof(VecT) / sizeof(T);
  const int kIterations = kDimCeil / kWarpSize;
  const int kIterationsV = (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  const int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int batch_size = n * d;
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  T sum[kBatchSize]{static_cast<T>(0.0)};
// for (int i = 0; i < kBatchSize; ++i) {
//   if (i >= local_batches) break;
//   for (int it = 0; it < kIterations; it++) {
//     int ids = first_batch + i;
//     int idx_n = ids / d;
//     int idx_d = ids % d;
//     int idx_dim = it * kWarpSize + threadIdx.x;
//     int idx = idx_n * dim * d + idx_d + idx_dim * d;
//     if (idx_dim < dim) {
//       sum[i] -= logT(softmax[idx]) * labels[idx];
//     }
//   }
// }

#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int ids = first_batch + i;
      int idx_n = ids / d;
      int idx_d = ids % d;
      int idx_dim = it * kWarpSize + threadIdx.x;
      int idx = idx_n * dim * d + idx_d + idx_dim * d;

      const VecT* softmaxptr_v = reinterpret_cast<const VecT*>(&softmax[idx]);
      const VecT* labelsptr_v = reinterpret_cast<const VecT*>(&labels[idx]);

      // max index to read
      // int idx_max = (i < local_batches) ? dim : 0;
      // int idx_max_v = idx_max / kVSize;
      // int idx = threadIdx.x + it * kWarpSize;

      if (idx < idx_dim) {
        int idx = threadIdx.x + it * kWarpSize;
        VecT softmaxdata = softmaxptr_v[idx];
        VecT labelsdata = labelsptr_v[idx];
        T* softmaxptr = reinterpret_cast<T*>(&softmaxdata);
        T* labelsptr = reinterpret_cast<T*>(&labelsdata);
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          sum[i] -= logT(softmaxptr[s]) * labelsptr[s];
        }
      }
    }
  }
  WarpReduceSum<T, kBatchSize, kWarpSize>(sum);

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < kBatchSize; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sum[0];
      }
    }
  }
}

template <typename T, typename VecT>
__global__ void CrossEntropySoftLabel(T* loss, const T* softmax,
                                      const T* labels, const int n,
                                      const int dim, const int d,
                                      int Log2Elements) {
  const int kDimCeil = 1 << Log2Elements;
  // const int kWarpSize = 32;
  const int kVSize = sizeof(VecT) / sizeof(T);
  // const int kIterations = kDimCeil / kWarpSize;

  const int kWarpSize = 32;
  const int kBatchSize = 1;
  const int kIterations = (dim + kWarpSize - 1) / kWarpSize;

  const int kIterationsV = (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  // const int kBatchSize = 1;

  int batch_size = n * d;
  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  T sum[kBatchSize]{static_cast<T>(0.0)};
// for (int i = 0; i < kBatchSize; ++i) {
//   if (i >= local_batches) break;
//   for (int it = 0; it < kIterations; it++) {
//     int ids = first_batch + i;
//     int idx_n = ids / d;
//     int idx_d = ids % d;
//     int idx_dim = it * kWarpSize + threadIdx.x;
//     int idx = idx_n * dim * d + idx_d + idx_dim * d;
//     if (idx_dim < dim) {
//       sum[i] -= logT(softmax[idx]) * labels[idx];
//     }
//   }
// }

#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      int ids = first_batch + i;
      int idx_n = ids / d;
      int idx_d = ids % d;
      int idx_dim = it * kWarpSize + threadIdx.x;
      int idx = idx_n * dim * d + idx_d + idx_dim * d;

      if (idx_dim < dim) {
        // max index to read
        // int idx_max = (i < local_batches) ? dim : 0;
        // int idx_max_v = idx_max / kVSize;
        // int idx = threadIdx.x + it * kWarpSize;

        // int idx = threadIdx.x + it * kWarpSize;
        VecT softmaxdata = reinterpret_cast<const VecT*>(&softmax[idx])[0];
        VecT labelsdata = reinterpret_cast<const VecT*>(&labels[idx])[0];
        T* softmaxptr = reinterpret_cast<T*>(&softmaxdata);
        T* labelsptr = reinterpret_cast<T*>(&labelsdata);
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          sum[i] -= logT(softmaxptr[s]) * labelsptr[s];
        }
      }
    }
  }
  WarpReduceSum<T, kBatchSize, kWarpSize>(sum);

  // write
  if (threadIdx.x == 0) {
    for (int i = 0; i < kBatchSize; i++) {
      int ids = first_batch + i;
      if (ids < n * d) {
        loss[ids] = sum[0];
      }
    }
  }
}

#define CROSS_ENTROPY_SOFT_CASE(Log2Elements, VecT)                      \
  case Log2Elements:                                                     \
    CrossEntropySoftLabel<T, VecT,                                       \
                          Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, labels, n, dim, d);                               \
    break;

template <typename T>
void SwitchCrossEntropySoftLabel(const int blocks, const dim3 threads,
                                 gpuStream_t stream, T* loss, const T* softmax,
                                 const T* labels, const int n, const int dim,
                                 const int d, const int Log2Elements) {
  switch (-1) {
    CROSS_ENTROPY_SOFT_CASE(0, T);
    CROSS_ENTROPY_SOFT_CASE(1, T);
    CROSS_ENTROPY_SOFT_CASE(2, T);
    CROSS_ENTROPY_SOFT_CASE(3, T);
    CROSS_ENTROPY_SOFT_CASE(4, T);
    CROSS_ENTROPY_SOFT_CASE(5, T);
    CROSS_ENTROPY_SOFT_CASE(6, T);
    CROSS_ENTROPY_SOFT_CASE(7, T);
    CROSS_ENTROPY_SOFT_CASE(8, T);
    CROSS_ENTROPY_SOFT_CASE(9, T);
    default:
      CrossEntropySoftLabel<T, T><<<blocks, threads, 0, stream>>>(
          loss, softmax, labels, n, dim, d, Log2Elements);
      break;
  }
}

template <typename T, typename VecT, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoftLabel(T* loss, T* softmax, const T* src,
                                            const T* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count) {
  const bool isLog = true;

  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT srcdata[kBatchSize][kIterationsV];
  VecT labeldata[kBatchSize][kIterationsV];

  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);
    const VecT* label_v =
        reinterpret_cast<const VecT*>(&label[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    // read data
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
        labeldata[i][it] = label_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<AccT>::max();
          reinterpret_cast<T*>(&labeldata[i][it])[s] = 0.0;
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    max_value[i] = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = srcptr_v[0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcptr_v[s]) ? valmax : srcptr_v[s];
      }
      max_value[i] = (max_value[i] > static_cast<AccT>(valmax))
                         ? max_value[i]
                         : static_cast<AccT>(valmax);
    }
  }
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (isLog) {
          sum[i] += std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
        } else {
          srcptr_v[s] = std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
          sum[i] += static_cast<AccT>(srcptr_v[s]);
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // log_softmax and loss
  AccT sumloss[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    if (isLog) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      T* labelvp = reinterpret_cast<T*>(&labeldata[i][it]);
      VecT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (isLog) {
          AccT logsoftmax = static_cast<AccT>(srcvp[s]) - max_value[i] - sum[i];
          sumloss[i] -= logsoftmax * static_cast<AccT>(labelvp[s]);
          tmpvp[s] = std::exp(logsoftmax);
        } else {
          tmpvp[s] = static_cast<AccT>(srcvp[s]) / sum[i];
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      }
    }
  }

  // loss
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sumloss);

  for (int i = 0; i < kBatchSize; i++) {
    if (i >= local_batches) break;
    loss[first_batch + i] = sumloss[i];
  }
}

template <typename T, typename VecT, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForwardHardLabel(T* loss, T* softmax, const T* src,
                                            const int64_t* label,
                                            const int batch_size,
                                            const int stride,
                                            const int element_count,
                                            const int ignore_index) {
  const bool isLog = true;

  constexpr int kDimCeil = 1 << Log2Elements;
  constexpr int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
  constexpr int kVSize = sizeof(VecT) / sizeof(T);
  constexpr int kIterations = kDimCeil / kWarpSize;
  constexpr int kIterationsV =
      (kIterations >= kVSize) ? (kIterations / kVSize) : 1;
  constexpr int kBatchSize = (kDimCeil <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchSize;
  int local_batches = batch_size - first_batch;
  if (local_batches > kBatchSize) {
    local_batches = kBatchSize;
  }

  // read data from global memory
  VecT srcdata[kBatchSize][kIterationsV];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    const VecT* src_v =
        reinterpret_cast<const VecT*>(&src[(first_batch + i) * stride]);

    // max index to read
    int src_idx_max = (i < local_batches) ? element_count : 0;
    int src_idx_max_v = src_idx_max / kVSize;

// read data
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      int src_idx = threadIdx.x + it * kWarpSize;
      if (src_idx < src_idx_max_v) {
        srcdata[i][it] = src_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < kVSize; s++) {
          reinterpret_cast<T*>(&srcdata[i][it])[s] =
              -std::numeric_limits<AccT>::infinity();
        }
      }
    }
  }

  // compute max value
  AccT max_value[kBatchSize];
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    max_value[i] = -std::numeric_limits<AccT>::infinity();
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
      T valmax = srcptr_v[0];
#pragma unroll
      for (int s = 1; s < kVSize; ++s) {
        valmax = (valmax > srcptr_v[s]) ? valmax : srcptr_v[s];
      }
      max_value[i] = (max_value[i] > static_cast<AccT>(valmax))
                         ? max_value[i]
                         : static_cast<AccT>(valmax);
    }
  }
  WarpReduceMax<AccT, kBatchSize, kWarpSize>(max_value);

  // compute sum
  AccT sum[kBatchSize]{0.0};
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcptr_v = reinterpret_cast<T*>(&srcdata[i][it]);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (isLog) {
          sum[i] += std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
        } else {
          srcptr_v[s] = std::exp(static_cast<AccT>(srcptr_v[s]) - max_value[i]);
          sum[i] += static_cast<AccT>(srcptr_v[s]);
        }
      }
    }
  }
  WarpReduceSum<AccT, kBatchSize, kWarpSize>(sum);

  // log_softmax
  VecT* softmax_v = reinterpret_cast<VecT*>(softmax);
#pragma unroll
  for (int i = 0; i < kBatchSize; ++i) {
    if (i >= local_batches) break;

    VecT* softmax_v =
        reinterpret_cast<VecT*>(&softmax[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / kVSize;

    if (isLog) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < kIterationsV; ++it) {
      T* srcvp = reinterpret_cast<T*>(&srcdata[i][it]);
      VecT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < kVSize; ++s) {
        if (isLog) {
          AccT logsoftmax = static_cast<AccT>(srcvp[s]) - max_value[i] - sum[i];
          tmpvp[s] = std::exp(logsoftmax);

          int loss_idx = (threadIdx.x + it * kWarpSize) * kVSize + s;
          if (label[first_batch + i] == loss_idx) {
            if (label[first_batch + i] != ignore_index) {
              loss[first_batch + i] = -logsoftmax;
            } else {
              loss[first_batch + i] = static_cast<T>(0.0);
            }
          }
        } else {
          tmpvp[s] = static_cast<AccT>(srcvp[s]) / sum[i];
        }
      }

      int idx = threadIdx.x + it * kWarpSize;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      }
    }
  }
}

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VecT, AccT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardSoftLabel<T, VecT, AccT,                                 \
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
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_SOFT_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_SOFT_CASE(9, T, AccT);
    default:
      break;
  }
}

// template <>
// void SwitchWarpSoftmaxForwardSoftLabel<paddle::platform::float16>(
//     const int blocks, const dim3 threads, gpuStream_t stream,
//     paddle::platform::float16* loss, paddle::platform::float16* softmax,
//     const paddle::platform::float16* src,
//     const paddle::platform::float16* label, const int batch_size,
//     const int stride, const int element_count, int Log2Elements) {
// #define T paddle::platform::float16
//   switch (Log2Elements) {
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(0, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(1, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(2, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(3, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(4, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(5, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(6, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(7, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(8, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_SOFT_CASE(9, paddle::platform::float16, float);
//     default:
//       break;
//   };
// #undef T
// };

#undef SOFTMAX_WARP_FORWARD_SOFT_CASE

#define SOFTMAX_WARP_FORWARD_HARD_CASE(Log2Elements, VecT, AccT)               \
  case Log2Elements:                                                           \
    WarpSoftmaxForwardHardLabel<T, VecT, AccT,                                 \
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
  using AccT = typename details::MPTypeTrait<T>::Type;
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_HARD_CASE(0, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(1, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(2, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(3, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(4, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(5, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(6, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(7, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(8, T, AccT);
    SOFTMAX_WARP_FORWARD_HARD_CASE(9, T, AccT);
    default:
      break;
  }
}

// template <>
// void SwitchWarpSoftmaxForwardHardLabel<paddle::platform::float16>(
//     const int blocks, const dim3 threads, gpuStream_t stream,
//     paddle::platform::float16* loss, paddle::platform::float16* softmax,
//     const paddle::platform::float16* src, const int64_t* label,
//     const int batch_size, const int stride, const int element_count,
//     int Log2Elements, const int ignore_index) {
// #define T paddle::platform::float16
//   switch (Log2Elements) {
//     SOFTMAX_WARP_FORWARD_HARD_CASE(0, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(1, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(2, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(3, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(4, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(5, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(6, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(7, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(8, paddle::platform::float16, float);
//     SOFTMAX_WARP_FORWARD_HARD_CASE(9, paddle::platform::float16, float);
//     default:
//       break;
//   };
// #undef T
// };

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
    const int kDimLog2 = static_cast<int>(log2_ceil(dim));
    const int kDimCeil = 1 << kDimLog2;
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchWarpSoftmaxForwardHardLabel<T>(
        blocks, threads, ctx.cuda_device_context().stream(), loss_data,
        softmax_data, logits_data, labels_data, N, dim, dim, kDimLog2,
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
  }
}

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

  const int kDimLog2 = static_cast<int>(log2_ceil(dim));
  const int kDimCeil = 1 << kDimLog2;

  if (D == 1 && dim <= max_dim) {
    int kWarpSize = (kDimCeil < 32) ? kDimCeil : 32;
    int batches_per_warp = (kDimCeil <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;
    int warps_per_block = (threads_per_block / kWarpSize);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchWarpSoftmaxForwardSoftLabel<T>(
        blocks, threads, ctx.cuda_device_context().stream(), loss_data,
        softmax_data, logits_data, labels_data, N, dim, dim, kDimLog2);

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
        handle, CUDNN_SOFTMAX_ACCURATE, mode,
        platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
        platform::CudnnDataType<T>::kZero(), desc_, out_data));
#else
    auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                 : CUDNN_SOFTMAX_MODE_CHANNEL;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, mode,
        platform::CudnnDataType<T>::kOne(), desc_, logits_data,
        platform::CudnnDataType<T>::kZero(), desc_, softmax_data));
#endif

    int kWarpSize = 32;  // (dim < 32) ? dim : 32;
    int batches_per_warp = 1;

    const int threads_per_block = 128;
    const int warps_per_block = 4;
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (N * D + batches_per_block - 1) / batches_per_block;
    dim3 threads(kWarpSize, warps_per_block, 1);

    SwitchCrossEntropySoftLabel<T>(
        blocks, threads, ctx.cuda_device_context().stream(), loss_data,
        softmax_data, labels_data, N, dim, D, kDimLog2);

    // CrossEntropySoftLabelGeneral<T><<<blocks, threads>>>(
    //   loss_data,
    //   softmax_data, labels_data, N, dim, D);
  }
}
}
}