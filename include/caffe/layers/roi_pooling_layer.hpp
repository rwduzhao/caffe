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
class ROIPoolingLayer : public Layer<Dtype> {
public:
  explicit ROIPoolingLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual bool CheckRoisSanity(const Blob<Dtype> *image_blob, const Blob<Dtype> *roi_blob);

  int channels_;  // bottom feature map channels
  int height_;  // bottom feature map height
  int width_;  // bottom feature map width
  int pooled_height_;
  int pooled_width_;
  int pooled_channels_;
  Dtype spatial_scale_;
  Blob<int> max_idx_;
  vector<shared_ptr<Blob<Dtype> > > temp_blobs_;
};

}

#endif  // CAFFE_LAYERS_ROI_POOLING_LAYERS_HPP_
