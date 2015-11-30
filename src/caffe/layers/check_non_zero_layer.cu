/*=============================================================================
#     FileName: check_non_zero_layer.cu
#   Desciption: check non zero layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-11-24 21:15:55
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io_extra.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/check_non_zero_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void row_check_kernel(const int row_count, const int row_dim, const Dtype *x, Dtype *y) {
  CUDA_KERNEL_LOOP(row_id, row_count) {
    y[row_id] = 0;
    for (int d = 0; d < row_dim; ++d)
      if (x[row_id * row_dim + d] != 0)
        y[row_id] = 1;
  }
}

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  row_check_kernel<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(num, dim, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(CheckNonZeroLayer);

}  // namespace caffe

