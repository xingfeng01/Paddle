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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#else
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

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

template <typename T, typename VECT, typename ACCT, int Log2Elements,
          bool isLog = false>
__global__ void WarpSoftmaxForward(T* softmax, const T* src,
                                   const int batch_size, const int stride,
                                   const int element_count) {
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

  // write result to global memory
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
      T* valvp = reinterpret_cast<T*>(&srcdata[i][it]);
      VECT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          tmpvp[s] = (ACCT)valvp[s] - max_value[i] - sum[i];
        } else {
          tmpvp[s] = (ACCT)valvp[s] / sum[i];
        };
      };

      int idx = threadIdx.x + it * warp_size;
      if (idx < idx_max_v) {
        softmax_v[idx] = tmpv;
      };
    };
  };
};

template <typename T, typename VECT, typename ACCT, int Log2Elements,
          bool isLog = false>
__global__ void WarpSoftmaxBackward(T* dst, const T* grad, const T* src,
                                    int batch_size, int stride,
                                    int element_count) {
  constexpr int VSIZE = sizeof(VECT) / sizeof(T);
  constexpr int dim_ceil = 1 << Log2Elements;
  constexpr int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
  constexpr int ITERATIONS = dim_ceil / warp_size;
  constexpr int BATCH_SIZE = (dim_ceil <= 128) ? 2 : 1;
  constexpr int ITERATIONS_V = (ITERATIONS >= VSIZE) ? (ITERATIONS / VSIZE) : 1;
  int element_count_v = element_count / VSIZE;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;
  int local_batches = batch_size - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  // read data from global memory
  VECT src_reg[BATCH_SIZE][ITERATIONS_V];
  VECT grad_reg[BATCH_SIZE][ITERATIONS_V];

  for (int i = 0; i < BATCH_SIZE; ++i) {
    const VECT* src_v =
        reinterpret_cast<const VECT*>(&src[(first_batch + i) * stride]);
    const VECT* grad_v =
        reinterpret_cast<const VECT*>(&grad[(first_batch + i) * stride]);

    // max index to read
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / VSIZE;

    // read data
    for (int it = 0; it < ITERATIONS_V; ++it) {
      int src_idx = threadIdx.x + it * warp_size;
      if (src_idx < idx_max_v) {
        src_reg[i][it] = src_v[src_idx];
        grad_reg[i][it] = grad_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < VSIZE; s++) {
          reinterpret_cast<T*>(&src_reg[i][it])[s] = 0.0;
          reinterpret_cast<T*>(&grad_reg[i][it])[s] = 0.0;
        };
      };
    };
  };

  // compute sum
  ACCT sum[BATCH_SIZE]{0.0};
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvpg = reinterpret_cast<T*>(&grad_reg[i][it]);
      T* valvps = reinterpret_cast<T*>(&src_reg[i][it]);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          sum[i] += (ACCT)valvpg[s];
        } else {
          sum[i] += (ACCT)(valvpg[s] * valvps[s]);
        }
      };
    }
  }
  WarpReduceSum<ACCT, BATCH_SIZE, warp_size>(sum);

// write result
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (i >= local_batches) break;

    VECT* dst_v = reinterpret_cast<VECT*>(&dst[(first_batch + i) * stride]);

    // max index to write
    int idx_max = (i < local_batches) ? element_count : 0;
    int idx_max_v = idx_max / VSIZE;

#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      VECT tmpv;
      T* tmpvp = reinterpret_cast<T*>(&tmpv);
      T* valvpg = reinterpret_cast<T*>(&grad_reg[i][it]);
      T* valvps = reinterpret_cast<T*>(&src_reg[i][it]);
#pragma unroll
      for (int s = 0; s < VSIZE; ++s) {
        if (isLog) {
          tmpvp[s] = (ACCT)valvpg[s] - std::exp((ACCT)valvps[s]) * sum[i];
        } else {
          tmpvp[s] = (ACCT)valvps[s] * ((ACCT)valvpg[s] - sum[i]);
        };
      };

      int idx = threadIdx.x + it * warp_size;
      if (idx < idx_max_v) {
        dst_v[idx] = tmpv;
      };
    };
  };
};

#define SOFTMAX_WARP_FORWARD_CASE(Log2Elements, VECT, ACCT)                 \
  case Log2Elements:                                                        \
    WarpSoftmaxForward<                                                     \
        T, VECT, ACCT, Log2Elements,                                        \
        isLog><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, src, batch_size, stride, element_count);                       \
    break;

template <typename T, bool isLog>
void SwitchWarpSoftmaxForward(const int blocks, const dim3 threads,
                              const framework::ExecutionContext& ctx, T* dst,
                              const T* src, const int batch_size,
                              const int stride, const int element_count,
                              int Log2Elements) {
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, T, T);
    SOFTMAX_WARP_FORWARD_CASE(1, T, T);
    SOFTMAX_WARP_FORWARD_CASE(2, T, T);
    SOFTMAX_WARP_FORWARD_CASE(3, T, T);
    SOFTMAX_WARP_FORWARD_CASE(4, T, T);
    SOFTMAX_WARP_FORWARD_CASE(5, T, T);
    SOFTMAX_WARP_FORWARD_CASE(6, T, T);
    SOFTMAX_WARP_FORWARD_CASE(7, T, T);
    SOFTMAX_WARP_FORWARD_CASE(8, T, T);
    SOFTMAX_WARP_FORWARD_CASE(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxForward<paddle::platform::float16, false>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* src, const int batch_size,
    const int stride, const int element_count, int Log2Elements) {
#define T paddle::platform::float16
#define isLog false
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
#undef isLog
};

template <>
void SwitchWarpSoftmaxForward<paddle::platform::float16, true>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* src, const int batch_size,
    const int stride, const int element_count, int Log2Elements) {
#define T paddle::platform::float16
#define isLog true
  switch (Log2Elements) {
    SOFTMAX_WARP_FORWARD_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_FORWARD_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
#undef isLog
};

#undef SOFTMAX_WARP_FORWARD_CASE

#define SOFTMAX_WARP_BACKWARD_CASE(Log2Elements, VECT, ACCT)                \
  case Log2Elements:                                                        \
    WarpSoftmaxBackward<                                                    \
        T, VECT, ACCT, Log2Elements,                                        \
        isLog><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, grad, src, batch_size, stride, element_count);                 \
    break;

template <typename T, bool isLog>
void SwitchWarpSoftmaxBackward(const int blocks, const dim3 threads,
                               const framework::ExecutionContext& ctx, T* dst,
                               const T* grad, const T* src,
                               const int batch_size, const int stride,
                               const int element_count, int Log2Elements) {
  switch (Log2Elements) {
    SOFTMAX_WARP_BACKWARD_CASE(0, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(1, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(2, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(3, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(4, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(5, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(6, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(7, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(8, T, T);
    SOFTMAX_WARP_BACKWARD_CASE(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxBackward<paddle::platform::float16, false>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* grad, const paddle::platform::float16* src,
    const int batch_size, const int stride, const int element_count,
    int Log2Elements) {
#define T paddle::platform::float16
#define isLog false
  switch (Log2Elements) {
    SOFTMAX_WARP_BACKWARD_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
#undef isLog
};

template <>
void SwitchWarpSoftmaxBackward<paddle::platform::float16, true>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* grad, const paddle::platform::float16* src,
    const int batch_size, const int stride, const int element_count,
    int Log2Elements) {
#define T paddle::platform::float16
#define isLog true
  switch (Log2Elements) {
    SOFTMAX_WARP_BACKWARD_CASE(0, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(1, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(2, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(3, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(4, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(5, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(6, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(7, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(8, paddle::platform::float16, float);
    SOFTMAX_WARP_BACKWARD_CASE(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
#undef isLog
};

#undef SwitchWarpSoftmaxBackward

template <typename T, bool isLog = false>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    constexpr int warps_per_block = 4;

    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
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

      SwitchWarpSoftmaxForward<T, isLog>(blocks, threads, ctx, out_data,
                                         x->data<T>(), N, dim, dim, dim_log2);

    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
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
      if (isLog) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
            handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
            desc_, x->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            out_data));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
            handle, CUDNN_SOFTMAX_ACCURATE, mode,
            platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, out_data));
      }
#endif
    }
  }
};

template <typename T, bool isLog = false>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int max_dim = 320;
    constexpr int warps_per_block = 4;

    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
      const int dim_log2 = log2_ceil(dim);
      const int dim_ceil = 1 << dim_log2;
      int warp_size = (dim_ceil < 32) ? dim_ceil : 32;
      int batches_per_warp = (dim_ceil <= 128) ? 2 : 1;
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / warp_size);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(warp_size, warps_per_block, 1);

      SwitchWarpSoftmaxBackward<T, isLog>(blocks, threads, ctx, dx_data,
                                          dout->data<T>(), out->data<T>(), N,
                                          dim, dim, dim_log2);

    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
#ifdef PADDLE_WITH_HIP
      miopenTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#else
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);
#endif

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();

#ifdef PADDLE_WITH_HIP
      auto mode = axis == rank - 1 ? MIOPEN_SOFTMAX_MODE_INSTANCE
                                   : MIOPEN_SOFTMAX_MODE_CHANNEL;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSoftmaxBackward(
          handle, platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(),
          desc_, dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
#else
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;
      if (isLog) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
            handle, CUDNN_SOFTMAX_LOG, mode, platform::CudnnDataType<T>::kOne(),
            desc_, out->data<T>(), desc_, dout->data<T>(),
            platform::CudnnDataType<T>::kZero(), desc_, dx_data));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
            handle, CUDNN_SOFTMAX_ACCURATE, mode,
            platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
            dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
            dx_data));
      }
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
#else
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);

// REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
//                    ops::SoftmaxCUDNNKernel<float, true>,
//                    ops::SoftmaxCUDNNKernel<double, true>,
//                    ops::SoftmaxCUDNNKernel<plat::float16, true>);
// REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
//                    ops::SoftmaxGradCUDNNKernel<float, true>,
//                    ops::SoftmaxGradCUDNNKernel<double, true>,
//                    ops::SoftmaxGradCUDNNKernel<plat::float16, true>);

#endif
