/*=============================================================================
#     FileName: self_gated_layer.cpp
#   Desciption: self gated layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-16 15:26:34
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/self_gated_layer.hpp"

namespace caffe {

template <typename Dtype>
void SelfGatedLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  time_step_ = 1;  // force one time step
  num_gate_ = 1;  // self gate only

  const Blob<Dtype> *input_blob = bottom[0];
  input_dim_ = input_blob->channels();

  const SelfGatedParameter layer_param = this->layer_param_.self_gated_param();
  CHECK_GE(layer_param.num_gate_layer(), 1);
  num_gate_layer_ = layer_param.num_gate_layer();
  CHECK_GE(layer_param.gate_net_dim(), 0);
  gate_net_dim_ = layer_param.gate_net_dim() > 0 ? layer_param.gate_net_dim() : input_dim_;
  hidden_dim_ = layer_param.num_output();
  clipping_threshold_ = layer_param.clipping_threshold();

  gate_layer_input_dims_.clear();
  gate_layer_input_dims_.resize(num_gate_layer_);
  gate_layer_output_dims_.clear();
  gate_layer_output_dims_.resize(num_gate_layer_);

  if (this->blobs_.size() == 0) {  // init weights and biases
    this->blobs_.resize(num_gate_layer_ * 2);

    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_layer_; ++gate_net_layer_id) {
      const int gate_layer_input_dim = gate_net_layer_id == 0 ? input_dim_ : num_gate_ * input_dim_;
      const int gate_layer_output_dim = gate_net_layer_id < (num_gate_layer_ - 1) ? num_gate_ * gate_net_dim_ : num_gate_ * hidden_dim_;
      gate_layer_input_dims_[gate_net_layer_id] = gate_layer_input_dim;
      gate_layer_output_dims_[gate_net_layer_id] = gate_layer_output_dim;

      // bias
      const int bias_blob_id = gate_net_layer_id * 2;
      vector<int> bias_shape(1, gate_layer_output_dim);
      this->blobs_[bias_blob_id].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(layer_param.bias_filler()));
      bias_filler->Fill(this->blobs_[bias_blob_id].get());

      // weight
      const int weight_blob_id = gate_net_layer_id * 2 + 1;
      vector<int> weight_shape;
      weight_shape.clear();
      weight_shape.push_back(gate_layer_output_dim);
      weight_shape.push_back(gate_layer_input_dim);
      this->blobs_[weight_blob_id].reset(new Blob<Dtype>(weight_shape));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(layer_param.weight_filler()));
      weight_filler->Fill(this->blobs_[weight_blob_id].get());
    }
  } else {
    CHECK_EQ(this->blobs_.size(), num_gate_layer_);

    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_layer_; ++gate_net_layer_id) {
      // bias
      const int bias_blob_id = gate_net_layer_id * 2;
      const int gate_layer_output_dim = this->blobs_[bias_blob_id]->num();
      if (gate_net_layer_id < num_gate_layer_ - 1)
        CHECK_EQ(gate_layer_output_dim, gate_net_dim_);
      else
        CHECK_EQ(gate_layer_output_dim, hidden_dim_);
      // weight
      const int weight_blob_id = gate_net_layer_id * 2 + 1;
      CHECK_EQ(gate_layer_output_dim, this->blobs_[weight_blob_id]->num());
      const int gate_layer_input_dim = this->blobs_[weight_blob_id]->channels();
      if (gate_net_layer_id == 0)
        CHECK_EQ(gate_layer_input_dim, input_dim_);
      else
        CHECK_EQ(gate_layer_input_dim, gate_layer_output_dims_[gate_net_layer_id - 1]);

      gate_layer_input_dims_[gate_net_layer_id] = gate_layer_input_dim;
      gate_layer_output_dims_[gate_net_layer_id] = gate_layer_output_dim;
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  gate_net_pre_tops_.clear();
  gate_net_pre_tops_.resize(num_gate_layer_);
  gate_net_tops_.clear();
  gate_net_tops_.resize(num_gate_layer_);
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_layer_; ++gate_net_layer_id) {
    const int gate_layer_output_dim = gate_layer_output_dims_[gate_net_layer_id];
    vector<int> blob_shape;
    blob_shape.clear();
    blob_shape.push_back(time_step_);
    blob_shape.push_back(1);
    blob_shape.push_back(num_gate_);
    blob_shape.push_back(gate_layer_output_dim / num_gate_);
    gate_net_pre_tops_[gate_net_layer_id].reset(new Blob<Dtype>(blob_shape));
    gate_net_tops_[gate_net_layer_id].reset(new Blob<Dtype>(blob_shape));
  }
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // check ups
  const Blob<Dtype> *input_blob = bottom[0];
  batch_size_ = input_blob->num() / time_step_;
  CHECK_EQ(input_dim_, input_blob->count() / input_blob->num());

  // bias multiplier
  vector<int> multiplier_shape(1, time_step_ * batch_size_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());

  vector<int> blob_shape;

  // pre_gate_tops
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_layer_; ++gate_net_layer_id) {
    const int gate_layer_output_dim = gate_layer_output_dims_[gate_net_layer_id];
    // gate_net_pre_tops and gate_net_tops
    blob_shape.clear();
    blob_shape.push_back(time_step_);
    blob_shape.push_back(batch_size_);
    blob_shape.push_back(num_gate_);
    blob_shape.push_back(gate_layer_output_dim / num_gate_);
    gate_net_pre_tops_[gate_net_layer_id]->Reshape(blob_shape);
    gate_net_tops_[gate_net_layer_id]->Reshape(blob_shape);
  }

  // pre_gate and gate
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(num_gate_);
  blob_shape.push_back(hidden_dim_);
  pre_gate_.Reshape(blob_shape);
  gate_.Reshape(blob_shape);
  pre_gate_.ShareData(*gate_net_pre_tops_[gate_net_pre_tops_.size() - 1]);
  pre_gate_.ShareDiff(*gate_net_pre_tops_[gate_net_pre_tops_.size() - 1]);
  gate_.ShareData(*gate_net_tops_[gate_net_tops_.size() - 1]);
  gate_.ShareDiff(*gate_net_tops_[gate_net_tops_.size() - 1]);

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
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SelfGatedLayer);
#endif

INSTANTIATE_CLASS(SelfGatedLayer);
REGISTER_LAYER_CLASS(SelfGated);

}  // namespace caffe
