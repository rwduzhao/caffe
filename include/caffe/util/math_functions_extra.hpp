#ifndef CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_HPP_
#define CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_HPP_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return Dtype(1.) / (Dtype(1.) + exp(-x));
}

template <typename Dtype>
void caffe_cpu_set(const int N, const Dtype alpha, Dtype *X);
template <typename Dtype>
void caffe_cpu_axpy(const int N, const Dtype alpha, const Dtype* X, const int incX, Dtype* Y, const int incY);
template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X, const int incX, Dtype* Y, const int incY);

template <typename Dtype> void caffe_cpu_sigmoid(const int N, const Dtype* x, Dtype* y);
template <typename Dtype> void caffe_gpu_sigmoid(const int N, const Dtype* x, Dtype* y);
template <typename Dtype> void caffe_cpu_sigmoid_diff(const int N, const Dtype* y, const Dtype* y_diff, Dtype* x_diff);
template <typename Dtype> void caffe_gpu_sigmoid_diff(const int N, const Dtype* y, const Dtype* y_diff, Dtype* x_diff);

template <typename Dtype> void caffe_cpu_tanh(const int N, const Dtype* x, Dtype* y);
template <typename Dtype> void caffe_gpu_tanh(const int N, const Dtype* x, Dtype* y);
template <typename Dtype> void caffe_cpu_tanh_diff(const int N, const Dtype* y, const Dtype* y_diff, Dtype* x_diff);
template <typename Dtype> void caffe_gpu_tanh_diff(const int N, const Dtype* y, const Dtype* y_diff, Dtype* x_diff);

template <typename Dtype> void caffe_cpu_bound(const int n, const Dtype* a, const Dtype min, const Dtype max, Dtype* y);
template <typename Dtype> void caffe_gpu_bound(const int n, const Dtype* a, const Dtype min, const Dtype max, Dtype* y);

template <typename Dtype> Dtype caffe_cpu_max(const int count, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_min(const int count, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_std(const int count, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_sum(const int count, const Dtype* x);
template <typename Dtype> Dtype caffe_cpu_mean(const int count, const Dtype* x);

template <typename Dtype> void caffe_cpu_row_sum(const int row_count, const int row_dim, const Dtype *x, Dtype *y);
template <typename Dtype> void caffe_gpu_row_sum(const int row_count, const int row_dim, const Dtype *x, Dtype *y);

}

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_HPP_
