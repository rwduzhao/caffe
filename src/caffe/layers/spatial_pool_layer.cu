#include <cfloat>
#include "caffe/layers/spatial_pool_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void SpatialPoolForward(
  const int top_count, const Dtype* bot_data,
  const int bot_c, const int bot_h, const int bot_w,
  const int top_c, const int top_h, const int top_w,
  const int num_roi, const Dtype* roi_data, const int* loc_data,
  Dtype* top_data, int* argmax_data) {

  CUDA_KERNEL_LOOP(top_id, top_count) {
    const int tw = top_id % top_w;
    const int th = (top_id / top_w) % top_h;
    const int tc = (top_id / top_w / top_h) % bot_c;
    const int tn = top_id / top_w / top_h / bot_c;

    argmax_data[top_id] = -1;
    for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
      const int image_id = roi_data[roi_id * 5 + 0];
      const int loc_id = roi_id * top_h * top_w + th * top_w + tw;
      if (image_id == tn && loc_data[loc_id] == 1) {
        const int bw = tc % bot_w;
        const int bh = (tc / bot_w) % bot_h;
        const int bc = tc / bot_w / bot_h;
        const int bn = roi_id;
        const int bot_id = bn * bot_c * bot_h * bot_w +
          bc * bot_h * bot_w + bh * bot_w + bw;
        if (top_data[top_id] < bot_data[bot_id]) {
          top_data[top_id] = bot_data[bot_id];
          argmax_data[top_id] = bot_id;
        }
      }
    }
  }
}

template <typename Dtype>
void CalcPoolLocations(const int num_roi, const Dtype* rois_data,
                       const Dtype* imsz_data,
                       const int pooled_h, const int pooled_w,
                       int* location_data) {
  for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
    const int image_id = static_cast<int>(rois_data[roi_id * 5 + 0]);
    const Dtype image_w = imsz_data[image_id * 2 + 0];
    const Dtype image_h = imsz_data[image_id * 2 + 1];
    const Dtype x0 = rois_data[roi_id * 5 + 1];
    const Dtype y0 = rois_data[roi_id * 5 + 2];
    const Dtype x1 = rois_data[roi_id * 5 + 3];
    const Dtype y1 = rois_data[roi_id * 5 + 4];

    const int px0 = min(static_cast<int>((x0 + 1.) / image_w * static_cast<Dtype>(pooled_w)), pooled_w - 1);
    const int py0 = min(static_cast<int>((y0 + 1.) / image_h * static_cast<Dtype>(pooled_h)), pooled_h - 1);
    const int px1 = min(static_cast<int>((x1 + 1.) / image_w * static_cast<Dtype>(pooled_w)), pooled_w - 1);
    const int py1 = min(static_cast<int>((y1 + 1.) / image_h * static_cast<Dtype>(pooled_h)), pooled_h - 1);

    CHECK_GE(px0, 0);
    CHECK_GE(py0, 0);
    CHECK_LE(px0, px1);
    CHECK_LE(py0, py1);
    for (int xx = px0; xx <= px1; ++xx) {
      for (int yy = px0; yy <= px1; ++yy) {
        const int index = roi_id * pooled_h * pooled_w + yy * pooled_w + xx;
        location_data[index] = 1;
      }
    }
  }
}

template <typename Dtype>
void SpatialPoolLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Blob<Dtype>* data_blob = bottom[0];
  const Blob<Dtype>* rois_blob = bottom[1];
  const Blob<Dtype>* imsz_blob = bottom[2];

  const Dtype* rois_data = rois_blob->cpu_data();
  const Dtype* imsz_data = imsz_blob->cpu_data();
  caffe_gpu_set(pool_locations_.count(), 0,
                pool_locations_.mutable_gpu_data());
  CalcPoolLocations(rois_blob->num(), rois_data, imsz_data, top_h_, top_w_,
                    pool_locations_.mutable_cpu_data());

  Blob<Dtype>* top_blob = top[0];
  Dtype* top_data = top_blob->mutable_gpu_data();

  int* argmax_data = max_idx_.mutable_gpu_data();
  caffe_gpu_set(max_idx_.count(), 0, argmax_data);

  const Dtype* bot_data = data_blob->gpu_data();
  const int* location_data = pool_locations_.gpu_data();
  int top_count = top_blob->count();
  ROIPoolingParameter param = this->layer_param_.roi_pooling_param();
  switch (param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      SpatialPoolForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, bot_data,
        bot_c_, bot_h_, bot_w_,
        top_c_, top_h_, top_w_,
        rois_blob->num(), rois_data, location_data,
        top_data, argmax_data);
      CUDA_POST_KERNEL_CHECK;
      break;
    case PoolingParameter_PoolMethod_AVE:
      NOT_IMPLEMENTED;
      break;
    default:
      NOT_IMPLEMENTED;
      break;
  }
}

template <typename Dtype> __global__
void SpatialPoolBackward_MAX(const int bot_count, const Dtype* top_diff,
                             const int* argmax_data, const int num_rois,
                             const int bot_c, const int bot_h, const int bot_w,
                             const int top_c, const int top_h, const int top_w,
                             Dtype* bot_diff, const Dtype* rois_data) {
  CUDA_KERNEL_LOOP(bot_id, bot_count) {
    // (n, c, h, w) coords in bottom data
    // int bw = bot_id % bot_w;
    // int bh = (bot_id / bot_w) % bot_h;
    int bc = (bot_id / bot_w / bot_h) % bot_c;
    int bn = bot_id / bot_w / bot_h / bot_c;

    Dtype gradient = 0.;
    for (int h = 0; h < top_h; ++h) {
      for (int w = 0; w < top_w; ++w) {
        const int tn = static_cast<int>(rois_data[bn * 5 + 0]);
        const int top_id = tn * top_c * top_h * top_w +
          bc * top_h * top_w + h * top_w + w;
        if (argmax_data[top_id] == bot_id) {
          gradient += top_diff[top_id];
        }
      }
    }
    bot_diff[bot_id] += gradient;
  }
}

template <typename Dtype>
void SpatialPoolLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  Blob<Dtype>* data_blob = bottom[0];
  Blob<Dtype>* rois_blob = bottom[1];
  Blob<Dtype>* imsz_blob = bottom[2];  // image size blob
  const Blob<Dtype>* top_blob = top[0];

  const Dtype* roi_data = rois_blob->gpu_data();
  const Dtype* top_diff = top_blob->gpu_diff();
  Dtype* bot_diff = data_blob->mutable_gpu_diff();
  const int bot_count = bottom[0]->count();
  caffe_gpu_set(bot_count, Dtype(0.), bot_diff);  // TODO
  const int* argmax_data = max_idx_.gpu_data();

  ROIPoolingParameter param = this->layer_param_.roi_pooling_param();
  switch (param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      SpatialPoolBackward_MAX<Dtype><<<CAFFE_GET_BLOCKS(bot_count), CAFFE_CUDA_NUM_THREADS>>>(
        bot_count, top_diff, argmax_data, bottom[1]->num(), bot_c_, bot_h_, bot_w_,
        top_c_, top_h_, top_w_, bot_diff, roi_data);
      CUDA_POST_KERNEL_CHECK;
      break;
    case ROIPoolingParameter_PoolMethod_AVE:
      NOT_IMPLEMENTED;
      break;
    default:
      NOT_IMPLEMENTED;
      break;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialPoolLayer);

}  // namespace caffe
