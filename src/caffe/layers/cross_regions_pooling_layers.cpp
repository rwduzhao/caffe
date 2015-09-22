#include <cfloat>
#include "caffe/layers/cross_regions_pooling_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

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
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // bottom[0]: computed roi features
  // bottom[1]: roi boxes
  // bottom[2]: image areas
  // top[0]: cross pooled data

  num_image_ = bottom[2]->num();
  bottom_channels_ = bottom[0]->channels();
  top_channels_ = num_scale_ * bottom_channels_;
  top[0]->Reshape(num_image_, top_channels_, 1, 1);
  max_idx_.Reshape(num_image_, top_channels_, 1, 1);

  const int num_roi = bottom[1]->num();
  roi_scale_ids_.Reshape(num_roi, 1, 1, 1);

  CHECK_EQ(num_image_, bottom[2]->num());
  image_areas_.Reshape(num_image_, 1, 1, 1);
  const Dtype *scaled_image_areas_data = bottom[2]->cpu_data();
  Dtype *image_areas_data = image_areas_.mutable_cpu_data();
  for (int image_id = 0; image_id < num_image_; ++image_id)
    image_areas_data[image_id] = scaled_image_areas_data[image_id];
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
//  const Dtype* bottom_data = bottom[0]->cpu_data();
//  const Dtype* bottom_roi_data = bottom[1]->cpu_data();
//  const int num_roi = bottom[1]->num();
//  const int batch_size = num_image;
//
//  const int top_count = top[0]->count();
//  Dtype* top_data = top[0]->mutable_cpu_data();
//  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
//  int* argmax_data = max_idx_.mutable_cpu_data();
//  caffe_set(top_count, -1, argmax_data);
//
//  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
//  for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
//    const int bottom_roi_offset = roi_id * bottom[1]->channels() * bottom[1]->height() * bottom[1]->width();
//    const int roi_batch_ind = bottom_roi_data[bottom_roi_offset];
//    CHECK_GE(roi_batch_ind, 0);
//    CHECK_LT(roi_batch_ind, batch_size);
//
//    for (int c = 0; c < channels_; ++c) {
//      const int index = roi_id * channels_ + c;
//      const int pool_index = roi_batch_ind * channels_ + c;
//      if (bottom_data[index] > top_data[pool_index]) {
//        top_data[pool_index] = bottom_data[index];
//        argmax_data[pool_index] = index;
//      }
//    }
//    // bottom_roi_data += bottom[1]->offset(1);
//  }
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
//  if (!propagate_down[0])
//    return;
//
//  const int bottom_count = bottom[0]->count();
//  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//  for (int bottom_index = 0; bottom_index < bottom_count; ++bottom_index)
//    bottom_diff[bottom_index] = 0.0;
//
//  const Dtype* top_diff = top[0]->cpu_diff();
//  const int* argmax_data = max_idx_.cpu_data();
//
//  const int top_count = top[0]->count();
//  for (int top_index = 0; top_index < top_count; ++top_index) {
//    const int bottom_index = argmax_data[top_index];
//    bottom_diff[bottom_index] = top_diff[top_index];
//  }
}

#ifdef CPU_ONLY
STUB_GPU(CrossRegionsPoolingLayer);
#endif

INSTANTIATE_CLASS(CrossRegionsPoolingLayer);
REGISTER_LAYER_CLASS(CrossRegionsPooling);

}  // namespace caffe
