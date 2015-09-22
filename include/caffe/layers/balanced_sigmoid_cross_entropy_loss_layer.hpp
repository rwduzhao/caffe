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

#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"

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

  virtual inline const char* type() const { return "MultilabelSigmoidCrossEntropyLoss"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

}

#endif  // CAFFE_LAYERS_BALANCED_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
