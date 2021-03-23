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

#define __MY_DEBUG__

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

template <typename T, typename VECT, typename ACCT, int Log2Elements>
__global__ void WarpSoftmaxForward(T* dst, const T* src, const int batch_size,
                                   const int stride, const int element_count) {
  const bool isLog = false;

  constexpr int NumElements = 1 << Log2Elements;
  constexpr int warp_size = (NumElements < 32) ? NumElements : 32;

  constexpr int VSIZE = sizeof(VECT) / sizeof(T);
  constexpr int ITERATIONS = NumElements / warp_size;
  constexpr int ITERATIONS_V = (ITERATIONS >= VSIZE) ? (ITERATIONS / VSIZE) : 1;

  constexpr int BATCH_SIZE = (NumElements <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;
  int local_batches = batch_size - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  // read data from global memory
  VECT elements[BATCH_SIZE][ITERATIONS_V];

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
        elements[i][it] = src_v[src_idx];
      } else {
#pragma unroll
        for (int s = 0; s < VSIZE; s++) {
          reinterpret_cast<T*>(&elements[i][it])[s] =
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
      T* valvp = reinterpret_cast<T*>(&elements[i][it]);
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
      T* valvp = reinterpret_cast<T*>(&elements[i][it]);
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
  VECT* dst_v = reinterpret_cast<VECT*>(dst);
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (i >= local_batches) break;

    VECT* dst_v = reinterpret_cast<VECT*>(&dst[(first_batch + i) * stride]);

    // max index to write
    int dst_idx_max = (i < local_batches) ? element_count : 0;
    int dst_idx_max_v = dst_idx_max / VSIZE;

    if (isLog) {
      sum[i] = std::log(sum[i]);
    }
#pragma unroll
    for (int it = 0; it < ITERATIONS_V; ++it) {
      T* valvp = reinterpret_cast<T*>(&elements[i][it]);
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

      int dst_idx = threadIdx.x + it * warp_size;
      if (dst_idx < dst_idx_max_v) {
        dst_v[dst_idx] = tmpv;
      };
    }
  };
};

template <typename T, typename VECT, typename ACCT, int Log2Elements>
__global__ void WarpSoftmaxBackward(T* dst, const T* grad, const T* src,
                                    int batch_size, int stride,
                                    int element_count) {
  const bool isLog = false;

  constexpr int VSIZE = sizeof(VECT) / sizeof(T);

  constexpr int NumElements = 1 << Log2Elements;
  constexpr int warp_size = (NumElements < 32) ? NumElements : 32;

  constexpr int ITERATIONS = NumElements / warp_size;
  constexpr int BATCH_SIZE = (NumElements <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * BATCH_SIZE;

  int local_batches = batch_size - first_batch;
  if (local_batches > BATCH_SIZE) {
    local_batches = BATCH_SIZE;
  };

  constexpr int ITERATIONS_V = (ITERATIONS >= VSIZE) ? (ITERATIONS / VSIZE) : 1;

  int element_count_v = element_count / VSIZE;

  const VECT* grad_v = reinterpret_cast<const VECT*>(grad);
  const VECT* src_v = reinterpret_cast<const VECT*>(src);
  VECT* dst_v = reinterpret_cast<VECT*>(dst);

  int local_idx = threadIdx.x + (first_batch * stride / VSIZE);

  // read data from global memory
  VECT src_reg[BATCH_SIZE][ITERATIONS_V];
  VECT grad_reg[BATCH_SIZE][ITERATIONS_V];

  for (int i = 0; i < BATCH_SIZE; ++i) {
    int batch_element_count = (i < local_batches) ? element_count : 0;
    for (int it = 0; it < ITERATIONS_V; ++it) {
      int local_index = local_idx + i * element_count_v + it * warp_size;
      grad_reg[i][it] = grad_v[local_index];
      src_reg[i][it] = src_v[local_index];
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
      int local_index = local_idx + i * element_count_v + it * warp_size;
      dst_v[local_index] = tmpv;
    };
  }
};

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements, VECT, ACCT)      \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, VECT, ACCT, Log2Elements><<<             \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, src, batch_size, stride, element_count);              \
    break;

template <typename T>
void SwitchWarpSoftmaxForward(const int blocks, const dim3 threads,
                              const framework::ExecutionContext& ctx, T* dst,
                              const T* src, const int batch_size,
                              const int stride, const int element_count,
                              int Log2Elements) {
  switch (Log2Elements) {
    LAUNCH_SOFTMAX_WARP_FORWARD(0, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(1, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(2, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(3, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(4, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(5, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(6, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(7, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(8, T, T);
    LAUNCH_SOFTMAX_WARP_FORWARD(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxForward<paddle::platform::float16>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* src, const int batch_size,
    const int stride, const int element_count, int Log2Elements) {
#define T paddle::platform::float16
  switch (Log2Elements) {
    LAUNCH_SOFTMAX_WARP_FORWARD(0, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(1, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(2, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(3, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(4, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(5, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(6, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(7, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(8, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_FORWARD(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
};

#undef LAUNCH_SOFTMAX_WARP_FORWARD

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements, VECT, ACCT)     \
  case Log2Elements:                                               \
    WarpSoftmaxBackward<T, VECT, ACCT, Log2Elements><<<            \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dst, grad, src, batch_size, stride, element_count);        \
    break;

template <typename T>
void SwitchWarpSoftmaxBackward(const int blocks, const dim3 threads,
                               const framework::ExecutionContext& ctx, T* dst,
                               const T* grad, const T* src,
                               const int batch_size, const int stride,
                               const int element_count, int Log2Elements) {
  switch (Log2Elements) {
    LAUNCH_SOFTMAX_WARP_BACKWARD(0, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(1, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(2, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(3, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(4, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(5, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(6, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(7, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(8, T, T);
    LAUNCH_SOFTMAX_WARP_BACKWARD(9, T, T);
    default:
      break;
  };
};

template <>
void SwitchWarpSoftmaxBackward<paddle::platform::float16>(
    const int blocks, const dim3 threads,
    const framework::ExecutionContext& ctx, paddle::platform::float16* dst,
    const paddle::platform::float16* grad, const paddle::platform::float16* src,
    const int batch_size, const int stride, const int element_count,
    int Log2Elements) {
#define T paddle::platform::float16
  switch (Log2Elements) {
    LAUNCH_SOFTMAX_WARP_BACKWARD(0, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(1, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(2, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(3, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(4, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(5, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(6, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(7, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(8, paddle::platform::float16, float);
    LAUNCH_SOFTMAX_WARP_BACKWARD(9, paddle::platform::float16, float);
    default:
      break;
  };
#undef T
};

#undef SwitchWarpSoftmaxBackward

template <typename T>
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
      int log2_elements = static_cast<int>(log2_ceil(dim));
      const int NumElements = 1 << log2_elements;
      int warp_size = (NumElements < 32) ? NumElements : 32;
      int batches_per_warp = (NumElements <= 128) ? 2 : 1;

      // use 128 threads per block to maximimize gpu utilization
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / warp_size);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(warp_size, warps_per_block, 1);

      SwitchWarpSoftmaxForward<T>(blocks, threads, ctx, out_data, x->data<T>(),
                                  N, dim, dim, log2_elements);

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
      const bool isLog = false;
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

template <typename T>
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
      int log2_elements = log2_ceil(dim);
      const int NumElements = 1 << log2_elements;
      int warp_size = (NumElements < 32) ? NumElements : 32;
      int batches_per_warp = (NumElements <= 128) ? 2 : 1;
      constexpr int threads_per_block = 128;

      int warps_per_block = (threads_per_block / warp_size);
      int batches_per_block = warps_per_block * batches_per_warp;
      int blocks = (N + batches_per_block - 1) / batches_per_block;
      dim3 threads(warp_size, warps_per_block, 1);

      SwitchWarpSoftmaxBackward<T>(blocks, threads, ctx, dx_data,
                                   dout->data<T>(), out->data<T>(), N, dim, dim,
                                   log2_elements);

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
      const bool isLog = false;
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
#endif
