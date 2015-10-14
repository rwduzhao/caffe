/*=============================================================================
#     FileName: one_step_unified_forget_gate_memory_layer.cpp
#   Desciption: One step unified forget gate memory layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-12 14:29:16
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/one_step_unified_forget_gate_memory_layer.hpp"

namespace caffe {

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  num_gate_ = 1;  // output gate only
  unified_dim_ = 1;

  const OneStepUnifiedForgetGateMemoryParameter layer_param = this->layer_param_.one_step_unified_forget_gate_memory_param();
  clipping_threshold_ = layer_param.clipping_threshold();
  hidden_dim_ = layer_param.num_output();

  if (this->blobs_.size() == 4)
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const OneStepUnifiedForgetGateMemoryParameter layer_param = this->layer_param_.one_step_unified_forget_gate_memory_param();

  // check ups
  // input
  const Blob<Dtype> *input_blob = bottom[0];
  time_step_ = 1;
  batch_size_ = input_blob->num() / time_step_;
  input_dim_ = input_blob->channels();
  CHECK_EQ(input_dim_, input_blob->count() / input_blob->num());
  // c_0
  const Blob<Dtype> *c_0 = bottom[1];
  CHECK_EQ(batch_size_, c_0->num());
  CHECK_EQ(hidden_dim_, c_0->channels());
  // e_0
  const Blob<Dtype> *e_0 = bottom[2];
  CHECK_EQ(batch_size_, e_0->num());
  extra_dim_ = e_0->channels();

  if (this->blobs_.size() == 0) {  // init weights and biases
    const int num_blob = 4;
    this->blobs_.resize(num_blob);
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(layer_param.weight_filler()));

    // bias
    int blob_id = 0;
    vector<int> bias_shape(1, num_gate_ * 1);
    this->blobs_[blob_id].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(layer_param.bias_filler()));
    bias_filler->Fill(this->blobs_[blob_id].get());

    vector<int> weight_shape;
    // weight_i
    blob_id = 1;
    weight_shape.clear();
    weight_shape.push_back(num_gate_ * 1);
    weight_shape.push_back(input_dim_);
    this->blobs_[blob_id].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[blob_id].get());
    // weight_h
    blob_id = 2;
    weight_shape.clear();
    weight_shape.push_back(num_gate_ * 1);
    weight_shape.push_back(hidden_dim_);
    this->blobs_[blob_id].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[blob_id].get());
    // weight_e
    blob_id = 3;
    weight_shape.clear();
    weight_shape.push_back(num_gate_ * 1);
    weight_shape.push_back(extra_dim_);
    this->blobs_[blob_id].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[blob_id].get());
  }

  vector<int> blob_shape;

  // c_0
  blob_shape.clear();
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(hidden_dim_);
  c_0_.Reshape(blob_shape);
  // h_0
  h_0_.Reshape(blob_shape);

  // e_0
  blob_shape.clear();
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(extra_dim_);
  e_0_.Reshape(blob_shape);

  // unified_pre_gate
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(num_gate_);
  blob_shape.push_back(unified_dim_);
  unified_pre_gate_.Reshape(blob_shape);
  // pre_gate
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(num_gate_);
  blob_shape.push_back(hidden_dim_);
  pre_gate_.Reshape(blob_shape);
  // gate
  gate_.Reshape(blob_shape);

  // original top
  blob_shape.clear();
  blob_shape.push_back(time_step_ * batch_size_);
  blob_shape.push_back(hidden_dim_);
  top[0]->Reshape(blob_shape);
  // top
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(hidden_dim_);
  top_.Reshape(blob_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);
  // cell
  cell_.Reshape(blob_shape);

  // bias multiplier
  vector<int> multiplier_shape(1, batch_size_ * time_step_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(OneStepUnifiedForgetGateMemoryLayer);
#endif

INSTANTIATE_CLASS(OneStepUnifiedForgetGateMemoryLayer);
REGISTER_LAYER_CLASS(OneStepUnifiedForgetGateMemory);

}  // namespace caffe
