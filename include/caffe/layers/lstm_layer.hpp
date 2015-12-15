/**
 * @brief Long-short term memory layer.
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

#ifndef __CAFFE_LAYERS_LSTM_LAYER_HPP__
#define __CAFFE_LAYERS_LSTM_LAYER_HPP__

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LstmLayer : public Layer<Dtype> {

public:
  explicit LstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Lstm"; }
  virtual bool IsRecurrent() const { return true; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int N_; // batch size

  Dtype clipping_threshold_; // threshold for clipped gradient

  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> clip_multiplier_;

  Blob<Dtype> top_;
  Blob<Dtype> pre_gate_;  // gate values before nonlinearity
  Blob<Dtype> gate_;      // gate values after nonlinearity
  Blob<Dtype> cell_;      // memory cell
  Blob<Dtype> tanh_cell_; // tanh(memory cell)
  Blob<Dtype> clip_mask_; // mask for sequence clipping

  Blob<Dtype> c_0_; // previous cell state value
  Blob<Dtype> h_0_; // previous hidden activation value
  Blob<Dtype> c_T_; // next cell state value
  Blob<Dtype> h_T_; // next hidden activation value

  // intermediate values
  Blob<Dtype> fdc_;
  Blob<Dtype> ig_;
  Blob<Dtype> clipped_;
};

}  // namespace caffe

#endif  // __CAFFE_LAYERS_LSTM_LAYER_HPP__
