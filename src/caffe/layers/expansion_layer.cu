/*=============================================================================
#     FileName: expansion_layer.cu
#   Desciption: expansion layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2016-01-27 23:17:11
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
#include "caffe/layers/expansion_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExpansionLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *multiplier_data = multiplier_.gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, dim_, 1,
                 Dtype(1.), bottom_data, multiplier_data, Dtype(0.), top_data);
}

template <typename Dtype>
void ExpansionLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom) {

  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_row_sum(batch_size_, dim_, top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpansionLayer);

}  // namespace caffe
