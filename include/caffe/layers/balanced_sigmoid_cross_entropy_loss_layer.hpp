/*
 * =====================================================================================
 *       Filename:  balanced_sigmoid_cross_entropy_loss_layer.hpp
 *    Description:  Balanced sigmoid cross entropy loss layer.
 *        Version:  1.0
 *        Created:  09/21/2015 06:34:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  rw.du.zhao@gmail.com
 * =====================================================================================
 */

#ifndef CAFFE_LAYERS_BALANCED_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_LAYERS_BALANCED_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class BalancedSigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
public:
  explicit BalancedSigmoidCrossEntropyLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param),
    sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
    sigmoid_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BalancedSigmoidCrossEntropyLoss"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;

  bool has_ignore_label_;
  int ignore_label_;
  LossParameter_NormalizationMode normalization_;
  Dtype normalizer_;
  int outer_num_, inner_num_;
  Dtype pos_count_;
  Dtype neg_count_;
  int prop_skip_period_;
  int prop_skip_index_;
  int prop_skip_count_;
};

}

#endif  // CAFFE_LAYERS_BALANCED_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
