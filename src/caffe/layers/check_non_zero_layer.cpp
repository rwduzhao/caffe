/*=============================================================================
#     FileName: check_non_zero_layer.cpp
#   Desciption: check non zero layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-11-24 21:10:24
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/check_non_zero_layer.hpp"

namespace caffe {

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
}

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  // check input size
  const Blob<Dtype> *input_blob = bottom[0];
  const int num = input_blob->num();

  vector<int> blob_shape;
  blob_shape.clear();
  blob_shape.push_back(num);
  top[0]->Reshape(blob_shape);
}

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CheckNonZeroLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CheckNonZeroLayer);
#endif

INSTANTIATE_CLASS(CheckNonZeroLayer);
REGISTER_LAYER_CLASS(CheckNonZero);

}  // namespace caffe
