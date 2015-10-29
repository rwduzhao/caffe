#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"

namespace caffe {

// sigmoid {{{

template <typename Dtype>
__global__ void sigmoid_kernel(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

template <>
void caffe_gpu_sigmoid(const int N, const float* x, float* y) {
  sigmoid_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_sigmoid(const int N, const double* x, double* y) {
  sigmoid_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, y);
  CUDA_POST_KERNEL_CHECK;
}

// sigmoid }}}

// sigmoid diff {{{

template <typename Dtype>
__global__ void sigmoid_diff_kernel(const int n, const Dtype* in_diff, const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

template <>
void caffe_gpu_sigmoid_diff(const int N, const float* y, const float* y_diff, float* x_diff) {
   sigmoid_diff_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y_diff, y, x_diff);
   CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_sigmoid_diff(const int N, const double* y, const double* y_diff, double* x_diff) {
   sigmoid_diff_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y_diff, y, x_diff);
   CUDA_POST_KERNEL_CHECK;
}

// sigmoid diff }}}

// tanh {{{

template <typename Dtype>
__global__ void tanh_kernel(const int N, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, N) {
    out[index] = tanh(in[index]);
  }
}

template <>
void caffe_gpu_tanh(const int N, const float* x, float* y) {
  tanh_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, y);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_tanh(const int N, const double* x, double* y) {
  tanh_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, x, y);
  CUDA_POST_KERNEL_CHECK;
}

// tanh }}}

// tanh diff {{{

template <typename Dtype>
__global__ void tanh_diff_kernel(const int N, const Dtype* in_diff, const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, N) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

template <>
void caffe_gpu_tanh_diff(const int N, const float* y, const float* y_diff, float* x_diff) {
  tanh_diff_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y_diff, y, x_diff);
  CUDA_POST_KERNEL_CHECK;
}
template <>
void caffe_gpu_tanh_diff(const int N, const double* y, const double* y_diff, double* x_diff) {
  tanh_diff_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, y_diff, y, x_diff);
  CUDA_POST_KERNEL_CHECK;
}

// tanh diff }}}

// bound {{{

template <typename Dtype>
__global__ void bound_kernel(const int n, const Dtype* a, const Dtype min_val, const Dtype max_val, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = min(max(a[index], min_val), max_val);
  }
}

template <>
void caffe_gpu_bound<float>(const int N, const float* a, const float min_val, const float max_val, float* y) {
  bound_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, min_val, max_val, y);
  CUDA_POST_KERNEL_CHECK;
}
template <>
void caffe_gpu_bound<double>(const int N, const double* a, const double min_val, const double max_val, double* y) {
  bound_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, min_val, max_val, y);
  CUDA_POST_KERNEL_CHECK;
}

// bound }}}

template <typename Dtype>
__global__ void row_sum_kernel(const int row_count, const int row_dim, const Dtype *x, Dtype *y) {
  CUDA_KERNEL_LOOP(row_id, row_count) {
    Dtype result = 0.0;
    for (int d = 0; d < row_dim; ++d)
      result += x[row_id * row_dim + d];
    y[row_id] = result;
  }
}
template <>
void caffe_gpu_row_sum(const int row_count, const int row_dim, const float *x, float *y) {
  row_sum_kernel<float><<<CAFFE_GET_BLOCKS(row_count), CAFFE_CUDA_NUM_THREADS>>>(row_count, row_dim, x, y);
  CUDA_POST_KERNEL_CHECK;
}
template <>
void caffe_gpu_row_sum(const int row_count, const int row_dim, const double *x, double *y) {
  row_sum_kernel<double><<<CAFFE_GET_BLOCKS(row_count), CAFFE_CUDA_NUM_THREADS>>>(row_count, row_dim, x, y);
  CUDA_POST_KERNEL_CHECK;
}

}
