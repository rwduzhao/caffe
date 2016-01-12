#include <cfloat>
#include "caffe/layers/cross_regions_pooling_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  CrossRegionsPoolingParameter param = this->layer_param_.cross_regions_pooling_param();

  pool_method_ = param.pool();

  num_scale_ = param.scale_levels_size() - 1;
  LOG(INFO) << "Number of scales: " << num_scale_;
  scale_levels_.Reshape(num_scale_, 1, 1, 1);
  Dtype *scale_levels_data = scale_levels_.mutable_cpu_data();
  for (int scale_id = 0; scale_id <= num_scale_; ++scale_id)
    scale_levels_data[scale_id] = param.scale_levels(scale_id);

  bottom_channels_ = bottom[0]->channels();
  top_channels_ = num_scale_ * bottom_channels_;
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // bottom[0]: computed roi features
  // bottom[1]: roi boxes
  // bottom[2]: image areas
  // top[0]: cross pooled data

  CHECK_EQ(bottom_channels_, bottom[0]->channels());

  const int num_rois = bottom[0]->num();
  CHECK_EQ(num_rois, bottom[1]->num());
  roi_scale_ids_.Reshape(num_rois, 1, 1, 1);

  num_image_ = bottom[2]->num();
  image_areas_.Reshape(num_image_, 1, 1, 1);
  image_areas_.ShareData(*bottom[2]);

  if (pool_method_ == CrossRegionsPoolingParameter_PoolMethod_STOCHASTIC) {
    Dtype *bottom_rois = bottom[1]->mutable_cpu_data();
    for (int roi_id = 0; roi_id < num_rois; ++roi_id) {
      const int roi_offset = roi_id * 5;
      bottom_rois[roi_offset + 0] = Dtype(roi_id);
    }
    top[0]->Reshape(num_rois, top_channels_, 1, 1);
    max_idx_.Reshape(num_rois, top_channels_, 1, 1);
  } else {
    top[0]->Reshape(num_image_, top_channels_, 1, 1);
    max_idx_.Reshape(num_image_, top_channels_, 1, 1);
  }

  if (pool_method_ == CrossRegionsPoolingParameter_PoolMethod_TOP)
    top_roi_ids_.Reshape(num_image_, 1, 1, 1);

  if (top.size() >= 2) {
    vector<int> blob_shape;
    blob_shape.push_back(num_rois);
    blob_shape.push_back(top_channels_);
    top[1]->Reshape(blob_shape);
  }
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CrossRegionsPoolingLayer);
#endif

INSTANTIATE_CLASS(CrossRegionsPoolingLayer);
REGISTER_LAYER_CLASS(CrossRegionsPooling);

}  // namespace caffe
