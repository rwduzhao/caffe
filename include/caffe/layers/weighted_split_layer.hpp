/*
 * =====================================================================================
 *       Filename:  weighted_split_layer.hpp
 *    Description:  weighted split layer
 *        Version:  1.0
 *        Created:  09/21/2015 06:34:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  rw.du.zhao@gmail.com
 * =====================================================================================
 */

#ifndef CAFFE_LAYERS_WEIGHTED_SPLIT_LAYER_HPP_
#define CAFFE_LAYERS_WEIGHTED_SPLIT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
class WeightedSplitLayer : public Layer<Dtype> {

public:
  explicit WeightedSplitLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WeightedSplit"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int count_;
  Blob<Dtype> diff_weights_;
};

}

#endif  // CAFFE_LAYERS_WEIGHTED_SPLIT_LAYER_HPP_
