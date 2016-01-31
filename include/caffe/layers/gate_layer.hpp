/*=============================================================================
#     FileName: gate_layer.hpp
#   Desciption: gate layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-26 21:15:47
#      History:
=============================================================================*/

#ifndef __CAFFE_LAYERS_GATE_LAYER_HPP__
#define __CAFFE_LAYERS_GATE_LAYER_HPP__

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GateLayer : public Layer<Dtype> {

public:
  explicit GateLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Gate"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int batch_size_;
  int input_dim_;
  int gate_input_dim_;
  int output_dim_;
  bool is_unit_gated_;

  Dtype clipping_threshold_;

  Blob<Dtype> gate_;
  Blob<Dtype> gate_dim_multiplier_;
};

}  // namespace caffe

#endif  // __CAFFE_LAYERS_GATE_LAYER_HPP__
