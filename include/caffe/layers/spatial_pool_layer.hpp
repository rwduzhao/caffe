#ifndef CAFFE_LAYERS_ROI_POOLING_LAYERS_HPP_
#define CAFFE_LAYERS_ROI_POOLING_LAYERS_HPP_

#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layer.hpp"
#include "caffe/internal_thread.hpp"

namespace caffe {

template <typename Dtype>
class SpatialPoolLayer : public Layer<Dtype> {
public:
  explicit SpatialPoolLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialPool"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int bot_c_;
  int bot_h_;
  int bot_w_;
  int top_n_;
  int top_c_;
  int top_h_;
  int top_w_;
  Blob<int> max_idx_;
  Blob<int> pool_locations_;
};

}

#endif  // CAFFE_LAYERS_ROI_POOLING_LAYERS_HPP_
