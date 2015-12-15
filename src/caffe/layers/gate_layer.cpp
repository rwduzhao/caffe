/*=============================================================================
#     FileName: gate_layer.cpp
#   Desciption: gate layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-26 21:13:09
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gate_layer.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  const GateParameter layer_param = this->layer_param_.gate_param();
  clipping_threshold_ = layer_param.clipping_threshold();
  CHECK_GE(clipping_threshold_, 0.);

  const Blob<Dtype> *input_blob = bottom[0];
  input_dim_ = input_blob->channels();
  CHECK_GE(input_dim_, 1);
  output_dim_ = input_dim_;

  const Blob<Dtype> *gate_input_blob = bottom[1];
  gate_input_dim_ = gate_input_blob->channels();
  if (gate_input_dim_ == 1 && output_dim_ != 1) {
    is_unit_gated_ = true;
  } else {
    is_unit_gated_ = false;
    CHECK_GE(gate_input_dim_, input_dim_);
  }

  if (is_unit_gated_) {
    vector<int> blob_shape;
    blob_shape.clear();
    blob_shape.push_back(output_dim_);
    gate_dim_multiplier_.Reshape(blob_shape);
    Dtype *gate_dim_multiplier_data = gate_dim_multiplier_.mutable_cpu_data();
    const int gate_dim_multiplier_count = gate_dim_multiplier_.count();
    caffe_set(gate_dim_multiplier_count, Dtype(1.), gate_dim_multiplier_data);
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  // check input size
  const Blob<Dtype> *input_blob = bottom[0];
  batch_size_ = input_blob->num();
  CHECK_EQ(input_dim_, input_blob->channels());
  // check gate net input size
  const Blob<Dtype> *gate_input_blob = bottom[1];
  CHECK_EQ(batch_size_, gate_input_blob->num());
  CHECK_EQ(gate_input_dim_, gate_input_blob->channels());

  vector<int> blob_shape;

  blob_shape.clear();
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(output_dim_);
  top[0]->Reshape(blob_shape);
  gate_.Reshape(blob_shape);
  if (!is_unit_gated_) {
    gate_.ShareData(*gate_input_blob);
    gate_.ShareDiff(*gate_input_blob);
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(GateLayer);
#endif

INSTANTIATE_CLASS(GateLayer);
REGISTER_LAYER_CLASS(Gate);

}  // namespace caffe
