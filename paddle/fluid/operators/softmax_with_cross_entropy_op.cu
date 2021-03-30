/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
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
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/for_range.h"
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

namespace {

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

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
__global__ void CrossEntropyGrad(T* logit_grad, const int64_t* labels,
                                 const int64_t n, const int64_t d,
                                 const int64_t remain, const int ignore_index) {
  CUDA_KERNEL_LOOP_TYPE(index, n * remain, int64_t) {
    int64_t idx_n = index / remain;
    int64_t idx_remain = index % remain;
    int64_t tmp = labels[index];
    if (ignore_index != tmp) {
      int64_t idx = idx_n * d + tmp * remain + idx_remain;
      logit_grad[idx] -= static_cast<T>(1.);
    }
  }
}

template <typename T>
__global__ void Scale(T* logit_grad, const T* loss_grad, const int64_t num,
                      const int64_t d, const int64_t remain,
                      const int64_t* labels, const int ignore_index) {
  CUDA_KERNEL_LOOP_TYPE(index, num, int64_t) {
    int64_t idx_n = index / d;
    int64_t idx_remain = index % remain;
    int64_t idx_lbl = idx_n * remain + idx_remain;
    if (labels[idx_lbl] == ignore_index) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels, const int64_t n,
                                               const int64_t d,
                                               const int64_t remain) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int64_t idx_n = ids / d;
    int64_t idx_remain = ids % remain;
    int64_t idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
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
    for (int it = 0; it < ITERATIONS; it++) {
      int ids = first_batch + i;
      int idx_n = ids / d;
      int idx_d = ids % d;
      int idx_dim = it * warp_size + threadIdx.x;
      int idx = idx_n * dim * d + idx_d + idx_dim * d;
      if (idx_dim < dim) {
        sum[i] -= (T)std::log((float)softmax[idx]) * labels[idx];
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

}  // namespace

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
      loss[ids] = 0.0;
    } else {
      loss[ids] = -std::log(softmax[idx]);
    }
  };
};

template <>
__global__ void CrossEntropyHardLabel<paddle::platform::float16>(
    paddle::platform::float16* loss, const paddle::platform::float16* softmax,
    const int64_t* labels, const int n, const int dim, const int d,
    const int ignore_idx) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = ids / d;
  int64_t idx_d = ids % d;
  if (ids < n * d) {
    int64_t idx = idx_n * dim * d + labels[ids] * d + idx_d;
    if (labels[ids] == ignore_idx) {
      loss[ids] = (paddle::platform::float16)0.0;
    } else {
      loss[ids] = -(paddle::platform::float16)std::log((float)softmax[idx]);
    }
  };
};

static __device__ __forceinline__ platform::float16 exp_on_device(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ platform::float16 log_on_device(
    platform::float16 x) {
  return math::TolerableValue<platform::float16>()(::Eigen::numext::log(x));
}
static __device__ __forceinline__ float log_on_device(float x) {
  return math::TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return math::TolerableValue<double>()(log(x));
}

template <typename T, typename VECT, typename ACCT, int Log2Elements>
__global__ void WarpSoftmaxForwardSoft(T* loss, T* softmax, const T* src,
                                       const T* label, const int batch_size,
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
__global__ void WarpSoftmaxForwardHard(T* loss, T* softmax, const T* src,
                                       const int64_t* label,
                                       const int batch_size, const int stride,
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

#define SOFTMAX_WARP_FORWARD_SOFT_CASE(Log2Elements, VECT, ACCT)          \
  case Log2Elements:                                                      \
    WarpSoftmaxForwardSoft<T, VECT, ACCT,                                 \
                           Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count);    \
    break;

template <typename T>
void SwitchWarpSoftmaxForwardSoft(const int blocks, const dim3 threads,
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
void SwitchWarpSoftmaxForwardSoft<paddle::platform::float16>(
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

#define SOFTMAX_WARP_FORWARD_HARD_CASE(Log2Elements, VECT, ACCT)          \
  case Log2Elements:                                                      \
    WarpSoftmaxForwardHard<T, VECT, ACCT,                                 \
                           Log2Elements><<<blocks, threads, 0, stream>>>( \
        loss, softmax, src, label, batch_size, stride, element_count,     \
        ignore_index);                                                    \
    break;

template <typename T>
void SwitchWarpSoftmaxForwardHard(const int blocks, const dim3 threads,
                                  gpuStream_t stream, T* loss, T* softmax,
                                  const T* src, const int64_t* label,
                                  const int batch_size, const int stride,
                                  const int element_count, int Log2Elements,
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
void SwitchWarpSoftmaxForwardHard<paddle::platform::float16>(
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

/** In the following codes, 3 CUDA kernels are implemented to calculate softmax
 * and loss **/
/*
  Supposing the x is `logits` and y is `labels`, the equations are as
followings:
  cross\_entropy_i = \sum_{j}[- y_i_j * log({e^{x_i_j}/\sum_{j}e^{x_i_j}})]
        = \sum_{j}[- y_i_j * log({e^{x_i_j - max_i}/\sum_{j}e^{x_i_j-max_i}})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - log\sum_{j}e^{x_i_j - max_i})]
        = \sum_{j}[-y_i_j * (x_i_j - max_i - logDiffMaxSum_i)]
        = \sum_{j}(-y_i_j * tmp_i_j)
  softmax_i_j = e^{tmp_i_j}
where:
  max_i = \max_{j}{x_i_j}
  logDiffMaxSum_i = log\sum_{j}e^{x_i_j - max_i}
  tmp_i_j = x_i_j - max_i - logDiffMaxSum_i
Therefore, the calculation can be separated into 3 steps:
Step 1: row-wise operation to calculate max_i
Step 2: row-wise operation to calculate logDiffMaxSum_i
Step 3: calculate tmp_i_j, and finally get softmax_i_j and cross\_entropy_i
To save memory, we can share memory among max_i, logDiffMaxSum_i and
cross\_entropy_i.
In this way, the 3 steps should be changed to:
Step 1 (RowReductionForMax): row-wise operation to calculate max_i
Step 2 (RowReductionForDiffMaxSum): calculate immediate result of softmax'_i_j =
x_i_j - max_i, and row-wise operation to calculate logDiffMaxSum_i
Step 3 (RowReductionForSoftmaxAndCrossEntropy): calculate tmp_i_j = softmax'_i_j
- logDiffMaxSum_i, and finally get softmax_i_j and cross\_entropy_i
*/

// There are 3 kinds of reduce algorithms in cub:
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING
// BLOCK_REDUCE_WARP_REDUCTIONS (default)
template <typename T, int BlockDim>
using BlockReduce =
    cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

// Make sure that BlockDim <= dim
// This kernel is used to calculate the max element of each row
template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int64_t d, int dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits_data view as [n, dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  int64_t step = BlockDim * remain;
  T cur_max = logits_data[beg_idx];
  beg_idx += step;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += step;
  }

  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());

  if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
}

// Make sure that BlockDim <= dim
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
                                                 T* max_data, T* softmax,
                                                 int64_t d, int dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax data view as [n, dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int64_t step = BlockDim * remain;

  // In numeric stable mode softmax_with_loss, we calc loss with
  // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
  // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
  // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
  // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
  // So there is no need to clip on shift_softmax.
  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());

  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);

  if (!CalculateLogSoftmax) return;
  __syncthreads();
  diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }

  // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
  // calculate diff_max_sum, __syncthreads() is needed here.
  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}

#ifdef __HIPCC__  // @{ HIP Seperate Kernel for RowReductionForDiffMaxSum
// Note(qili93): HIP do not support return in kernel, need to seperate
// RowReductionForDiffMaxSum into two kernels below
template <typename T, int BlockDim>
static __global__ void RowReductionForSum(const T* logits_data, T* max_data,
                                          T* softmax, int64_t d, int dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  int64_t remain = d / dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int64_t step = BlockDim * remain;

  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    idx += step;
  }

  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
  if (threadIdx.x == 0) max_data[blockIdx.x] = log_on_device(diff_max_sum);
}

template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiff(const T* logits_data, T* max_data,
                                           T* softmax, int d, int dim) {
  int remain = d / dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;
  int step = BlockDim * remain;

  T diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    beg_idx += step;
  }

  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
}
#endif  // @} End HIP Seperate Kernel for RowReductionForDiffMaxSum

// Make sure that BlockDim <= dim
template <typename T, int BlockDim>
static __global__ void RowReductionForSoftmaxAndCrossEntropy(
    const T* logits_data, const T* labels_data, T* loss_data, T* softmax,
    int64_t d, int dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax, labels data view as [n, dim, remain]
  // loss_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int64_t remain = d / dim;
  int64_t idx_n = blockIdx.x / remain;
  int64_t idx_remain = blockIdx.x % remain;
  int64_t beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int64_t end_idx = (idx_n + 1) * d;

  // log_diff_max_sum shares memory with loss
  auto block_log_diff_max_sum = loss_data[blockIdx.x];
  auto tmp = softmax[beg_idx] - block_log_diff_max_sum;
  softmax[beg_idx] = exp_on_device(tmp);
  auto loss = -labels_data[beg_idx] * tmp;
  int64_t step = BlockDim * remain;
  beg_idx += step;
  while (beg_idx < end_idx) {
    tmp = softmax[beg_idx] - block_log_diff_max_sum;
    softmax[beg_idx] = exp_on_device(tmp);
    loss -= (labels_data[beg_idx] * tmp);
    beg_idx += step;
  }

  loss = BlockReduce<T, BlockDim>(temp_storage).Reduce(loss, cub::Sum());
  if (threadIdx.x == 0) loss_data[blockIdx.x] = loss;
}

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctor {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctor(const int64_t* labels, T* loss,
                                          T* log_softmax, int64_t d, int dim,
                                          int ignore_idx)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        dim_(dim),
        ignore_idx_(ignore_idx) {}

  __device__ void operator()(int64_t idx) const {
    // logits view as [n, dim, remain], where d = dim * remain
    int64_t remain = d_ / dim_;
    int64_t idx_n = idx / d_;
    int64_t idx_axis = (idx % d_) / remain;
    int64_t idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int64_t idx_lbl = idx_n * remain + idx_remain;
    PADDLE_ENFORCE(labels_[idx_lbl] >= 0 && labels_[idx_lbl] < d_ ||
                       labels_[idx_lbl] == ignore_idx_,
                   "The value of label[%ld] expected >= 0 and < %ld, or == %d,"
                   "but got %ld. Please check input value.",
                   idx_lbl, d_, ignore_idx_, labels_[idx_lbl]);

    // It also would ignore labels not in range(class_num).
    if (idx_axis != labels_[idx_lbl]) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int64_t d_;
  int dim_;
  int ignore_idx_;
};

template <typename T>
struct HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx {
 public:
  HardLabelSoftmaxWithCrossEntropyFunctorWithIgnoreIdx(const int64_t* labels,
                                                       T* loss, T* log_softmax,
                                                       int64_t d, int dim,
                                                       int ignore_idx)
      : labels_(labels),
        loss_(loss),
        log_softmax_(log_softmax),
        d_(d),
        dim_(dim),
        ignore_idx_(ignore_idx) {}

  __device__ void operator()(int64_t idx) const {
    // logits view as [n, dim, remain], where d = dim * remain
    int64_t remain = d_ / dim_;
    int64_t idx_n = idx / d_;
    int64_t idx_axis = (idx % d_) / remain;
    int64_t idx_remain = idx % remain;
    // labels, loss view as [n, remain]
    int64_t idx_lbl = idx_n * remain + idx_remain;
    if (idx_axis != labels_[idx_lbl] || idx_axis == ignore_idx_) {
      log_softmax_[idx] = exp_on_device(log_softmax_[idx]);
    } else {
      auto softmax = log_softmax_[idx];
      log_softmax_[idx] = exp_on_device(softmax);
      loss_[idx_lbl] = -softmax;
    }
  }

 private:
  const int64_t* labels_;
  T* loss_;
  T* log_softmax_;
  int64_t d_;
  int dim_;
  int ignore_idx_;
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

    SwitchWarpSoftmaxForwardHard<T>(
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

    SwitchWarpSoftmaxForwardSoft<T>(
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

template <typename T>
class SoftmaxWithCrossEntropyCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));

    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");
    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    const bool soft_label = context.Attr<bool>("soft_label");
    const int ignore_index = context.Attr<int>("ignore_index");
    const bool softmax_switch = context.Attr<bool>("softmax_switch");

    int dim = logits->dims()[axis];
    const int64_t n = SizeToAxis(axis, logits->dims());
    const int64_t d = SizeOutAxis(axis, logits->dims());

    auto* softmax_data = softmax->mutable_data<T>(context.GetPlace());
    auto* loss_data = loss->mutable_data<T>(context.GetPlace());

    if (dim == 1) {
      math::SetConstant<platform::CUDADeviceContext, T> set_constant;
      set_constant(context.cuda_device_context(), softmax, static_cast<T>(1));
      set_constant(context.cuda_device_context(), loss, static_cast<T>(0));
      return;
    }

    if (!softmax_switch) {
      if (soft_label) {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<T>();

        int warp_size = 32;  // (dim < 32) ? dim : 32;
        int batches_per_warp = 1;
        constexpr int warps_per_block = 4;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (n * d + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        CrossEntropySoftLabel<T><<<blocks, threads>>>(loss_data, logits_data,
                                                      labels_data, n, dim, d);
      } else {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<int64_t>();
        int threads = 128;
        int blocks = (n * d + threads - 1) / threads;
        CrossEntropyHardLabel<T><<<blocks, threads>>>(
            loss_data, logits_data, labels_data, n, dim, d, ignore_index);
      }

      framework::TensorCopy(*logits, context.GetPlace(),
                            context.device_context(), softmax);
    } else {
      if (soft_label) {
        auto* logits_data = logits->data<T>();
        auto* labels_data = labels->data<T>();
        SoftmaxWithCrossEntropySoftLabel<T>(context, rank, axis, logits_data,
                                            labels_data, softmax_data,
                                            loss_data, n, dim, d);
      } else {
        if (!context.Attr<bool>("numeric_stable_mode")) {
          // CUDNN kernel only suppoer 2-D tensor and perfome softmax on last
          // dim
          Tensor logits_2d, softmax_2d, labels_2d, loss_2d;
          logits_2d.ShareDataWith(*logits).Resize({n, d * dim});
          softmax_2d.ShareDataWith(*softmax).Resize({n, d * dim});
          labels_2d.ShareDataWith(*labels).Resize({n, labels->numel() / n});
          loss_2d.ShareDataWith(*loss).Resize({n, 1});
          math::SoftmaxCUDNNFunctor<T>()(context.cuda_device_context(),
                                         &logits_2d, &softmax_2d);
          math::CrossEntropyFunctor<platform::CUDADeviceContext, T>()(
              context.cuda_device_context(), &loss_2d, &softmax_2d, &labels_2d,
              false, ignore_index, dim);
        } else {
          auto* logits_data = logits->data<T>();
          auto* labels_data = labels->data<int64_t>();
          SoftmaxWithCrossEntropyHardLabel<T>(
              context, rank, axis, logits_data, labels_data, loss_data,
              softmax_data, n, dim, d, ignore_index);
        }
      }
    }  // end of softmax_switch
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("softmax_with_cross_entropy operator's "
                                      "CUDA kernel only runs on GPU device."));

    const Tensor* labels = context.Input<Tensor>("Label");
    const T* loss_grad_data =
        context.Input<Tensor>(framework::GradVarName("Loss"))->data<T>();
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* softmax = context.Input<Tensor>("Softmax");

    if (logit_grad != softmax) {
      framework::TensorCopy(*softmax, context.GetPlace(),
                            context.device_context(), logit_grad);
    }
    T* logit_grad_data = logit_grad->data<T>();

    const int rank = logit_grad->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    int dim = logit_grad->dims()[axis];

    const int64_t n = SizeToAxis(axis, logit_grad->dims());
    const int64_t d = SizeFromAxis(axis, logit_grad->dims());
    const int64_t remain = d / dim;

    int block = 512;
    auto stream = context.cuda_device_context().stream();
    auto ignore_index = context.Attr<int>("ignore_index");
    if (context.Attr<bool>("soft_label")) {
      int64_t grid = (n * d + block - 1) / block;
      const T* label_data = labels->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      int64_t grid = (n * remain + block - 1) / block;
      const int64_t* label_data = labels->data<int64_t>();
      CrossEntropyGrad<T><<<grid, block, 0, stream>>>(
          logit_grad_data, label_data, n, d, remain, ignore_index);
      int64_t num = n * d;
      grid = (num + block - 1) / block;
      Scale<T><<<grid, block, 0, stream>>>(logit_grad_data, loss_grad_data, num,
                                           d, remain, label_data, ignore_index);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>);
#else
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<float>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<paddle::platform::float16>,
    ops::SoftmaxWithCrossEntropyGradCUDAKernel<double>);
#endif
