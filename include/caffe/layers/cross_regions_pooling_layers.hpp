#ifndef CAFFE_LAYERS_CROSS_REGIONS_POOLING_LAYERS_HPP_
#define CAFFE_LAYERS_CROSS_REGIONS_POOLING_LAYERS_HPP_

#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layer.hpp"
#include "caffe/internal_thread.hpp"

namespace caffe {

template <typename Dtype>
class CrossRegionsPoolingLayer : public Layer<Dtype> {
public:
  explicit CrossRegionsPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossRegionsPooling"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int bottom_channels_;
  int top_channels_;
  int num_scale_;
  int num_image_;
  int pool_method_;
  Blob<Dtype> scale_levels_;
  Blob<Dtype> image_areas_;
  Blob<int> max_idx_;
  Blob<int> roi_scale_ids_;
  Blob<int> top_roi_ids_;
};

}

#endif  // CAFFE_LAYERS_CROSS_REGIONS_POOLING_LAYERS_HPP_
