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
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  time_step_ = 1;  // force one time step

  num_gate_ = 1;  // output gate only

  const SelfGatedParameter layer_param = this->layer_param_.self_gated_param();
  num_gate_net_layer_ = layer_param.num_gate_net_layer();
  CHECK_GE(num_gate_net_layer_, 1);

  const Blob<Dtype> *input_blob = bottom[0];
  input_dim_ = input_blob->channels();
  CHECK_GE(input_dim_, 1);

  // output_dim_ = layer_param.num_output();
  output_dim_ = input_dim_;
  CHECK_GE(output_dim_, 1);

  const Blob<Dtype> *gate_net_input_blob = bottom[1];
  gate_net_input_dim_ = gate_net_input_blob->channels();
  CHECK_GE(gate_net_input_dim_, 1);

  gate_net_hidden_dim_ = layer_param.gate_net_hidden_dim();
  CHECK_GE(gate_net_hidden_dim_, 1);

  is_last_unit_gated_ = layer_param.is_last_unit_gated();
  gate_net_output_dim_ = is_last_unit_gated_ ? 1 : output_dim_;

  clipping_threshold_ = layer_param.clipping_threshold();
  CHECK_GE(clipping_threshold_, 0.);

  gate_net_layer_input_dims_.clear();
  gate_net_layer_input_dims_.resize(num_gate_net_layer_);
  gate_net_layer_output_dims_.clear();
  gate_net_layer_output_dims_.resize(num_gate_net_layer_);

  if (this->blobs_.size() == 0) {  // init weights and biases
    this->blobs_.resize(num_gate_net_layer_ * 2);  // pairs of weights and biases

    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
      // calculate gate net input, output dims
      const int gate_net_layer_input_dim = gate_net_layer_id == 0 ? gate_net_input_dim_ : num_gate_ * gate_net_hidden_dim_;
      const int gate_net_layer_output_dim = gate_net_layer_id < (num_gate_net_layer_ - 1) ? num_gate_ * gate_net_hidden_dim_ : num_gate_ * gate_net_output_dim_;

      // assign gate net input, output dims
      gate_net_layer_input_dims_[gate_net_layer_id] = gate_net_layer_input_dim;
      gate_net_layer_output_dims_[gate_net_layer_id] = gate_net_layer_output_dim;

      // parameter blobs
      vector<int> blob_shape;
      // init gate net layer weight
      const int weight_blob_id = gate_net_layer_id * 2 + 0;
      blob_shape.clear();
      blob_shape.push_back(gate_net_layer_output_dim);
      blob_shape.push_back(gate_net_layer_input_dim);
      this->blobs_[weight_blob_id].reset(new Blob<Dtype>(blob_shape));
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(layer_param.weight_filler()));
      weight_filler->Fill(this->blobs_[weight_blob_id].get());
      // init gate net layer bias
      const int bias_blob_id = gate_net_layer_id * 2 + 1;
      blob_shape.clear();
      blob_shape.push_back(gate_net_layer_output_dim);
      this->blobs_[bias_blob_id].reset(new Blob<Dtype>(blob_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(layer_param.bias_filler()));
      bias_filler->Fill(this->blobs_[bias_blob_id].get());
    }
  } else {
    CHECK_EQ(this->blobs_.size(), num_gate_net_layer_ * 2);

    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
      // check gate net layer input dim
      const int weight_blob_id = gate_net_layer_id * 2 + 0;
      const int gate_net_layer_input_dim = this->blobs_[weight_blob_id]->channels();
      if (gate_net_layer_id == 0)
        CHECK_EQ(gate_net_layer_input_dim, input_dim_);
      else
        CHECK_EQ(gate_net_layer_input_dim, gate_net_layer_output_dims_[gate_net_layer_id - 1]);
      // check gate net layer output dim
      const int gate_net_layer_output_dim = this->blobs_[weight_blob_id]->num();
      if (gate_net_layer_id < num_gate_net_layer_ - 1)
        CHECK_EQ(gate_net_layer_output_dim, num_gate_ * gate_net_hidden_dim_);
      else
        CHECK_EQ(gate_net_layer_output_dim, num_gate_ * gate_net_output_dim_);
      const int bias_blob_id = gate_net_layer_id * 2 + 1;
      CHECK_EQ(gate_net_layer_output_dim, this->blobs_[bias_blob_id]->num());

      // assign gate net layer input, output dims
      gate_net_layer_input_dims_[gate_net_layer_id] = gate_net_layer_input_dim;
      gate_net_layer_output_dims_[gate_net_layer_id] = gate_net_layer_output_dim;
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  if (this->layer_param_.param_size() == 0) {
    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
      ParamSpec *weight_param_spec = this->layer_param_.add_param();
      weight_param_spec->set_lr_mult(1.);
      weight_param_spec->set_decay_mult(1.);
      ParamSpec *bias_param_spec = this->layer_param_.add_param();
      bias_param_spec->set_lr_mult(2.);
      bias_param_spec->set_decay_mult(0.);
    }
  }

  if (is_last_unit_gated_) {
    vector<int> blob_shape;
    blob_shape.clear();
    blob_shape.push_back(output_dim_);
    last_gate_net_top_multiplier_.Reshape(blob_shape);
    Dtype *last_gate_net_top_multiplier_data = last_gate_net_top_multiplier_.mutable_cpu_data();
    const int last_gate_net_top_multiplier_count = last_gate_net_top_multiplier_.count();
    caffe_set(last_gate_net_top_multiplier_count, Dtype(1.), last_gate_net_top_multiplier_data);
  }

  gate_net_pre_tops_.clear();
  gate_net_pre_tops_.resize(num_gate_net_layer_);
  gate_net_tops_.clear();
  gate_net_tops_.resize(num_gate_net_layer_);
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
    const int gate_net_layer_output_dim = gate_net_layer_output_dims_[gate_net_layer_id];
    vector<int> blob_shape;
    blob_shape.clear();
    blob_shape.push_back(time_step_);
    blob_shape.push_back(1);
    blob_shape.push_back(num_gate_);
    blob_shape.push_back(gate_net_layer_output_dim / num_gate_);
    gate_net_pre_tops_[gate_net_layer_id].reset(new Blob<Dtype>(blob_shape));
    gate_net_tops_[gate_net_layer_id].reset(new Blob<Dtype>(blob_shape));
  }
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  // check input size
  const Blob<Dtype> *input_blob = bottom[0];
  batch_size_ = input_blob->num() / time_step_;
  CHECK_EQ(input_dim_, input_blob->channels());
  CHECK_EQ(input_dim_, input_blob->count() / time_step_ / batch_size_);
  // check gate net input size
  const Blob<Dtype> *gate_net_input_blob = bottom[1];
  CHECK_EQ(batch_size_, gate_net_input_blob->num());
  CHECK_EQ(gate_net_input_dim_, gate_net_input_blob->channels());
  CHECK_EQ(gate_net_input_dim_, gate_net_input_blob->count() / time_step_ / batch_size_);

  vector<int> blob_shape;

  // bias multiplier
  blob_shape.clear();
  blob_shape.push_back(time_step_ * batch_size_);
  bias_multiplier_.Reshape(blob_shape);
  Dtype *bias_multiplier_data = bias_multiplier_.mutable_cpu_data();
  const int bias_multiplier_count = bias_multiplier_.count();
  caffe_set(bias_multiplier_count, Dtype(1.), bias_multiplier_data);

  // gate_net_pre_tops, gate_net_tops
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
    const int gate_net_layer_output_dim = gate_net_layer_output_dims_[gate_net_layer_id];
    blob_shape.clear();
    blob_shape.push_back(time_step_);
    blob_shape.push_back(batch_size_);
    blob_shape.push_back(num_gate_);
    blob_shape.push_back(gate_net_layer_output_dim / num_gate_);
    gate_net_pre_tops_[gate_net_layer_id]->Reshape(blob_shape);
    gate_net_tops_[gate_net_layer_id]->Reshape(blob_shape);
  }

  // pre_gate, gate
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(num_gate_);
  blob_shape.push_back(output_dim_);
  gate_.Reshape(blob_shape);
  if (!is_last_unit_gated_) {
    gate_.ShareData(*gate_net_tops_[gate_net_tops_.size() - 1]);
    gate_.ShareDiff(*gate_net_tops_[gate_net_tops_.size() - 1]);
  }

  // original top, top
  blob_shape.clear();
  blob_shape.push_back(time_step_ * batch_size_);
  blob_shape.push_back(output_dim_);
  top[0]->Reshape(blob_shape);
  blob_shape.clear();
  blob_shape.push_back(time_step_);
  blob_shape.push_back(batch_size_);
  blob_shape.push_back(output_dim_);
  top_.Reshape(blob_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);

  if (top.size() >= 2) {
    blob_shape.clear();
    blob_shape.push_back(time_step_ * batch_size_);
    blob_shape.push_back(num_gate_ * output_dim_);
    top[1]->Reshape(blob_shape);
    top[1]->ShareData(gate_);
  }
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
