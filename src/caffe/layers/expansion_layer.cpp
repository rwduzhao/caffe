/*=============================================================================
#     FileName: expansion_layer.cpp
#   Desciption: expansion layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2016-01-27 22:52:06
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/expansion_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExpansionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  const ExpansionParameter layer_param = this->layer_param_.expansion_param();
  dim_ = layer_param.dim();
  CHECK_GE(dim_, 1);

  vector<int> blob_shape;
  blob_shape.clear();
  blob_shape.push_back(dim_);
  multiplier_.Reshape(blob_shape);
  Dtype *multiplier_data = multiplier_.mutable_cpu_data();
  const int multiplier_count = multiplier_.count();
  caffe_set(multiplier_count, Dtype(1.), multiplier_data);
}

template <typename Dtype>
void ExpansionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  // check input size
  batch_size_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), 1);

  vector<int> blob_shape;
  blob_shape.clear();
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(dim_);
  top[0]->Reshape(blob_shape);
}

template <typename Dtype>
void ExpansionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ExpansionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ExpansionLayer);
#endif

INSTANTIATE_CLASS(ExpansionLayer);
REGISTER_LAYER_CLASS(Expansion);

}  // namespace caffe
