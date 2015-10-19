/*=============================================================================
#     FileName: self_gated_layer.hpp
#   Desciption: self gated layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-16 15:21:06
#      History:
=============================================================================*/

#ifndef __CAFFE_LAYERS_SELF_GATED_LAYER_HPP__
#define __CAFFE_LAYERS_SELF_GATED_LAYER_HPP__

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
class SelfGatedLayer : public Layer<Dtype> {

public:
  explicit SelfGatedLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelfGated"; }
  virtual bool IsRecurrent() const { return true; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int time_step_;
  int num_gate_;
  int batch_size_;
  int input_dim_;
  int output_dim_;
  int num_gate_net_layer_;
  int gate_net_input_dim_;
  int gate_net_hidden_dim_;
  int gate_net_output_dim_;
  vector<int> gate_net_layer_input_dims_;
  vector<int> gate_net_layer_output_dims_;

  Dtype clipping_threshold_;

  Blob<Dtype> top_;
  Blob<Dtype> gate_;
  Blob<Dtype> bias_multiplier_;

  bool is_last_unit_gated_;
  Blob<Dtype> last_gate_net_top_multiplier_;

  vector<shared_ptr<Blob<Dtype> > > gate_net_tops_;
  vector<shared_ptr<Blob<Dtype> > > gate_net_pre_tops_;
  vector<shared_ptr<Blob<Dtype> > > gate_net_bias_multipliers_;
};

}  // namespace caffe

#endif  // __CAFFE_LAYERS_SELF_GATED_LAYER_HPP__
