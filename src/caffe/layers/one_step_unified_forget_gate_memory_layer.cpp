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

  const OneStepUnifiedForgetGateMemoryParameter layer_param = this->layer_param_.one_step_unified_forget_gate_memory_param();
  clipping_threshold_ = layer_param.clipping_threshold();
  hidden_dim_ = layer_param.num_output();

  vector<int> clip_mul_shape(1, hidden_dim_);
  clip_multiplier_.Reshape(clip_mul_shape);
  caffe_set(clip_multiplier_.count(), Dtype(1), clip_multiplier_.mutable_cpu_data());

  if (this->blobs_.size() == 3) {
    // blobs_[0]: weight_i (1 * hidden_dim_ by input_dim_)
    // blobs_[1]: weight_e (1 * hidden_dim_ by hidden_dim_)
    // blobs_[2]: bias (1 * hidden_dim_)
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const OneStepUnifiedForgetGateMemoryParameter layer_param = this->layer_param_.one_step_unified_forget_gate_memory_param();

  // check ups
  // input
  const Blob<Dtype> *x_0 = bottom[0];
  time_step_ = 1;
  batch_size_ = x_0->num() / time_step_;
  input_dim_ = x_0->channels();
  CHECK_EQ(input_dim_, x_0->count() / x_0->num());
  // c_0
  const Blob<Dtype> *c_0 = bottom[1];
  CHECK_EQ(batch_size_, c_0->num());
  CHECK_EQ(hidden_dim_, c_0->channels());
  // e_0
  const Blob<Dtype> *e_0 = bottom[2];
  CHECK_EQ(batch_size_, e_0->num());
  extra_dim_ = e_0->channels();

  if (this->blobs_.size() == 0) {  // init weights and biases
    this->blobs_.resize(3);
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(layer_param.weight_filler()));

    // weight_hi
    vector<int> weight_shape;
    weight_shape.push_back(1);
    weight_shape.push_back(input_dim_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // weight_he
    weight_shape.clear();
    weight_shape.push_back(1);
    weight_shape.push_back(extra_dim_);
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // bias term
    vector<int> bias_shape(1, 1);
    this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(layer_param.bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }

  // c_0
  vector<int> cell_shape;
  cell_shape.push_back(batch_size_);
  cell_shape.push_back(hidden_dim_);
  c_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);

  // e_0
  vector<int> extra_shape;
  extra_shape.clear();
  extra_shape.push_back(batch_size_);
  extra_shape.push_back(extra_dim_);
  e_0_.Reshape(extra_shape);

  vector<int> unified_pre_gate_shape;
  unified_pre_gate_shape.push_back(time_step_);
  unified_pre_gate_shape.push_back(batch_size_);
  unified_pre_gate_shape.push_back(1);
  unified_pre_gate_shape.push_back(1);
  unified_pre_gate_.Reshape(unified_pre_gate_shape);

  // gate and pre_gate
  vector<int> gate_shape;
  gate_shape.push_back(time_step_);
  gate_shape.push_back(batch_size_);
  gate_shape.push_back(1);
  gate_shape.push_back(hidden_dim_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);

  // top and cell
  vector<int> original_top_shape;
  original_top_shape.push_back(time_step_ * batch_size_);
  original_top_shape.push_back(hidden_dim_);
  top[0]->Reshape(original_top_shape);
  vector<int> top_shape;
  top_shape.push_back(time_step_);
  top_shape.push_back(batch_size_);
  top_shape.push_back(hidden_dim_);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);
  cell_.Reshape(top_shape);

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
