#include <cfloat>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/spatial_pool_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void SpatialPoolLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // layer parameters
  ROIPoolingParameter param = this->layer_param_.roi_pooling_param();

  // pooled sizes
  top_h_ = param.pooled_h();
  CHECK_GT(top_h_, 0) << "pooled height must be greater 0";
  top_w_ = param.pooled_w();
  CHECK_GT(top_w_, 0) << "pooled width must be greater 0";
}

template <typename Dtype>
void SpatialPoolLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 3);
  const Blob<Dtype>* data_blob = bottom[0];
  const Blob<Dtype>* rois_blob = bottom[1];  // [image_id, x0, y0, x1, y1]
  const Blob<Dtype>* imsz_blob = bottom[2];  // image size blob, [n, w, h]

  CHECK_EQ(data_blob->num(), rois_blob->num());

  // bottom feature map sizes
  bot_c_ = data_blob->channels();
  bot_h_ = data_blob->height();
  bot_w_ = data_blob->width();

  // top feature map sizes
  top_n_ = imsz_blob->num();
  top_c_ = data_blob->count() / data_blob->num();
  Blob<Dtype>* top_blob = top[0];
  top_blob->Reshape(top_n_, top_c_, top_h_, top_w_);
  max_idx_.Reshape(top_n_, top_c_, top_h_, top_w_);

  // pool locations
  const int num_rois = rois_blob->num();
  pool_locations_.Reshape(num_rois, 1, top_h_, top_w_);
}

template <typename Dtype>
void SpatialPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SpatialPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(SpatialPoolLayer);
#endif

INSTANTIATE_CLASS(SpatialPoolLayer);
REGISTER_LAYER_CLASS(SpatialPool);

}  // namespace caffe
