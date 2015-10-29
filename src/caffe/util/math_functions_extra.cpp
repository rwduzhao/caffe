#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// sigmoid {{{

template <>
void caffe_cpu_sigmoid(const int N, const float* x, float* y) {
  for (int i = 0; i < N; ++i)
    y[i] = sigmoid(x[i]);
}

template <>
void caffe_cpu_sigmoid(const int N, const double* x, double* y) {
  for (int i = 0; i < N; ++i)
    y[i] = sigmoid(x[i]);
}

// sigmoid }}}

// sigmoid diff {{{

template <>
void caffe_cpu_sigmoid_diff(const int N, const float* y, const float* y_diff, float* x_diff) {
  for (int i = 0; i < N; ++i) {
    const float sigmoid_x = y[i];
    x_diff[i] = y_diff[i] * sigmoid_x * (1. - sigmoid_x);
  }
}

template <>
void caffe_cpu_sigmoid_diff(const int N, const double* y, const double* y_diff, double* x_diff) {
  for (int i = 0; i < N; ++i) {
    const double sigmoid_x = y[i];
    x_diff[i] = y_diff[i] * sigmoid_x * (1. - sigmoid_x);
  }
}

// sigmoid diff }}}

// tanh {{{

template <>
void caffe_cpu_tanh(const int N, const float* x, float* y) {
  for (int i = 0; i < N; ++i)
    y[i] = tanh(x[i]);
}

template <>
void caffe_cpu_tanh(const int N, const double* x, double* y) {
  for (int i = 0; i < N; ++i)
    y[i] = tanh(x[i]);
}

// tanh }}}

// tanh diff {{{

template <>
void caffe_cpu_tanh_diff(const int N, const float* y, const float* y_diff, float* x_diff) {
  for (int i = 0; i < N; ++i) {
    const float tanh_x = y[i];
    x_diff[i] = y_diff[i] * (1. - tanh_x * tanh_x);
  }
}
template <>
void caffe_cpu_tanh_diff(const int N, const double* y, const double* y_diff, double* x_diff) {
  for (int i = 0; i < N; ++i) {
    const double tanh_x = y[i];
    x_diff[i] = y_diff[i] * (1. - tanh_x * tanh_x);
  }
}

// tanh diff }}}

// bound {{{

template <>
void caffe_cpu_bound(const int N, const float* a, const float min, const float max, float* y) {
  for (int i = 0; i < N; ++i)
    y[i] = std::min(std::max(a[i], min), max);
}

template <>
void caffe_cpu_bound(const int N, const double* a, const double min, const double max, double* y) {
  for (int i = 0; i < N; ++i)
    y[i] = std::min(std::max(a[i], min), max);
}

// bound }}}

template <>
float caffe_cpu_max(const int count, const float* x) {
  float result = x[0];
  for (int n = 0; n < count; ++n) {
    if (x[n] > result)
      result = x[n];
  }
  return result;
}
template <>
double caffe_cpu_max(const int count, const double* x) {
  double result = x[0];
  for (int n = 0; n < count; ++n) {
    if (x[n] > result)
      result = x[n];
  }
  return result;
}

template <>
float caffe_cpu_min(const int count, const float* x) {
  float result = x[0];
  for (int n = 0; n < count; ++n) {
    if (x[n] < result)
      result = x[n];
  }
  return result;
}
template <>
double caffe_cpu_min(const int count, const double* x) {
  double result = x[0];
  for (int n = 0; n < count; ++n) {
    if (x[n] < result)
      result = x[n];
  }
  return result;
}

template <>
float caffe_cpu_sum(const int count, const float* x) {
  float result = 0.0;
  for (int n = 0; n < count; ++n)
    result += x[n];
  return result;
}
template <>
double caffe_cpu_sum(const int count, const double* x) {
  double result = 0.0;
  for (int n = 0; n < count; ++n)
    result += x[n];
  return result;
}

template <>
float caffe_cpu_mean(const int count, const float* x) {
  float result = caffe_cpu_sum(count, x) / float(count);
  return result;
}
template <>
double caffe_cpu_mean(const int count, const double* x) {
  double result = caffe_cpu_sum(count, x) / double(count);
  return result;
}

template <>
float caffe_cpu_std(const int count, const float* x) {
  float result = 0.0;
  const float mean = caffe_cpu_mean(count, x);
  for (int n = 0; n < count; ++n) {
    const float diff = x[n] - mean;
    result += (diff * diff);
  }
  result /= float(count);
  return result;
}
template <>
double caffe_cpu_std(const int count, const double* x) {
  double result = 0.0;
  const double mean = caffe_cpu_mean(count, x);
  for (int n = 0; n < count; ++n) {
    const double diff = x[n] - mean;
    result += (diff * diff);
  }
  result /= double(count);
  return result;
}

template <>
void caffe_cpu_row_sum(const int row_count, const int row_dim, const float *x, float *y) {
  for (int row_id = 0; row_id < row_count; ++row_id) {
    const float *row_x = x + row_id * row_dim;
    y[row_id] = caffe_cpu_sum(row_dim, row_x);
  }
}
template <>
void caffe_cpu_row_sum(const int row_count, const int row_dim, const double *x, double *y) {
  for (int row_id = 0; row_id < row_count; ++row_id) {
    const double *row_x = x + row_id * row_dim;
    y[row_id] = caffe_cpu_sum(row_dim, row_x);
  }
}

}
