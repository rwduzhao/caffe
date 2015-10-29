#include <cfloat>
#include "caffe/layers/cross_regions_pooling_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void CalculateRegionScaleIndice(
  const int num_rois, const Dtype* bottom_rois,
  const int num_scale, const Dtype* scale_levels_data,
  const Dtype *image_areas_data, int *roi_scale_ids_data) {

  CUDA_KERNEL_LOOP(roi_id, num_rois) {
    const int roi_offset = roi_id * 5;

    const Dtype roi_x0 = Dtype(bottom_rois[roi_offset + 1]);
    const Dtype roi_y0 = Dtype(bottom_rois[roi_offset + 2]);
    const Dtype roi_x1 = Dtype(bottom_rois[roi_offset + 3]);
    const Dtype roi_y1 = Dtype(bottom_rois[roi_offset + 4]);
    const Dtype roi_w = roi_x1 - roi_x0 + 1;
    const Dtype roi_h = roi_y1 - roi_y0 + 1;
    const Dtype roi_area = roi_w * roi_h;

    const int roi_image_id = bottom_rois[roi_offset + 0];
    const Dtype image_area = image_areas_data[roi_image_id];

    const Dtype area_ratio = roi_area / image_area;
    int roi_scale_id = num_scale - 1;  // pre-assign to largest scale
    for (int scale_id = 0; scale_id < num_scale; ++scale_id) {
      if (area_ratio >= scale_levels_data[scale_id] &&
          area_ratio < scale_levels_data[scale_id + 1]) {
        roi_scale_id = scale_id;
      }
    }
    roi_scale_ids_data[roi_id] = roi_scale_id;
  }
}

template <typename Dtype>
__global__ void MultiScaledCrossRegionsMaximumPoolForward(
  const int top_count, const int top_channels,
  const int bottom_channels, const Dtype* bottom_data,
  const int num_rois, const Dtype* bottom_rois,
  const int *roi_scale_ids_data,
  Dtype *top_data, int *argmax_data) {

  CUDA_KERNEL_LOOP(top_index, top_count) {
    const int top_image_id = top_index / top_channels;
    const int top_c = top_index % top_channels;
    const int top_scale_id = (top_c - (top_c % bottom_channels)) / bottom_channels;
    const int bottom_c = top_index % bottom_channels;

    int maxidx = -1;
    Dtype maxval = -FLT_MAX;
	for (int roi_id = 0; roi_id < num_rois; ++roi_id) {  // loop for each roi
      const int roi_offset = roi_id * 5;
      const int roi_image_id = bottom_rois[roi_offset + 0];
      if (roi_image_id == top_image_id) {
        const int roi_scale_id = roi_scale_ids_data[roi_id];
        if (roi_scale_id == top_scale_id) {
          const int bottom_index = roi_id * bottom_channels + bottom_c;
          if (bottom_data[bottom_index] > maxval) {
            maxval = bottom_data[bottom_index];
            maxidx = bottom_index;
          }
        }
      }
	}

    if (maxidx == -1 && maxval == -FLT_MAX)
      maxval = 0.0;
	top_data[top_index] = maxval;
	argmax_data[top_index] = maxidx;
  }
}

template <typename Dtype>
__global__ void MultiScaledCrossRegionsAveragePoolForward(
  const int top_count, const int top_channels,
  const int bottom_channels, const Dtype* bottom_data,
  const int num_rois, const Dtype* bottom_rois,
  const int *roi_scale_ids_data,
  Dtype *top_data, int *pool_counts) {

  CUDA_KERNEL_LOOP(top_index, top_count) {
    const int top_image_id = top_index / top_channels;
    const int top_c = top_index % top_channels;
    const int top_scale_id = (top_c - (top_c % bottom_channels)) / bottom_channels;
    const int bottom_c = top_index % bottom_channels;

	top_data[top_index] = 0.0;
    int pool_count = 0;
	for (int roi_id = 0; roi_id < num_rois; ++roi_id) {  // loop for each roi
      const int roi_offset = roi_id * 5;
      const int roi_image_id = bottom_rois[roi_offset + 0];
      if (roi_image_id == top_image_id) {
        const int roi_scale_id = roi_scale_ids_data[roi_id];
        if (roi_scale_id == top_scale_id) {
          const int bottom_index = roi_id * bottom_channels + bottom_c;
          top_data[top_index] += bottom_data[bottom_index];
          ++pool_count;
        }
      }
	}
    if (pool_count > 0)
      top_data[top_index] /= Dtype(pool_count);
    pool_counts[top_index] = pool_count;
  }
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // fixed variables
  const int top_count = top[0]->count();
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const int num_rois = bottom[1]->num();
  const Dtype *bottom_rois = bottom[1]->gpu_data();
  const Dtype *scale_levels_data = scale_levels_.gpu_data();
  const Dtype *image_areas_data = image_areas_.gpu_data();

  // mutable variables
  Dtype *top_data = top[0]->mutable_gpu_data();
  int *argmax_data = max_idx_.mutable_gpu_data();
  int *pool_counts = argmax_data;
  int *roi_scale_ids_data = roi_scale_ids_.mutable_gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  CalculateRegionScaleIndice<Dtype><<<CAFFE_GET_BLOCKS(num_rois), CAFFE_CUDA_NUM_THREADS>>>(
    num_rois, bottom_rois,
    num_scale_, scale_levels_data,
    image_areas_data, roi_scale_ids_data);
  CUDA_POST_KERNEL_CHECK;

  switch (pool_method_) {
    case CrossRegionsPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      MultiScaledCrossRegionsMaximumPoolForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, top_channels_,
        bottom_channels_, bottom_data,
        num_rois, bottom_rois, roi_scale_ids_data,
        top_data, argmax_data);
      CUDA_POST_KERNEL_CHECK;
      break;
    case CrossRegionsPoolingParameter_PoolMethod_AVE:
      // NOLINT_NEXT_LINE(whitespace/operators)
      MultiScaledCrossRegionsAveragePoolForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, top_channels_,
        bottom_channels_, bottom_data,
        num_rois, bottom_rois,
        roi_scale_ids_data,
        top_data, pool_counts);
      CUDA_POST_KERNEL_CHECK;
      break;
    default:
      break;
  }

  // set top[1]
  if (top.size() >= 2) {
    const Dtype *bottom_rois = bottom[1]->cpu_data();
    const int num_roi = top[1]->num();
    for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
      const int roi_offset = roi_id * 5;
      const int roi_image_id = bottom_rois[roi_offset + 0];
      Dtype *bottom_data = top[0]->mutable_gpu_data() + top[0]->offset(roi_image_id);
      Dtype *top_data = top[1]->mutable_gpu_data() + top[1]->offset(roi_id);
      const int top_count = top[1]->channels();
      caffe_copy(top_count, bottom_data, top_data);
    }
  }

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    caffe_gpu_set(top[top_id]->count(), Dtype(0.), top[top_id]->mutable_gpu_diff());
  }
}

template <typename Dtype>
__global__ void MultiScaledCrossRegionsMaximumPoolBackward(
  const int bottom_count, const int bottom_channels, const int top_channels,
  const Dtype *bottom_rois, const int *roi_scale_ids_data, const int* argmax_data,
  const Dtype* top_diff, Dtype *bottom_diff) {

  CUDA_KERNEL_LOOP(bottom_index, bottom_count) {
    const int roi_id = bottom_index / bottom_channels;
    const int bottom_c = bottom_index % bottom_channels;

	const int roi_image_id = bottom_rois[5 * roi_id + 0];
    const int roi_scale_id = roi_scale_ids_data[roi_id];
    const int top_index = roi_image_id * top_channels + roi_scale_id * bottom_channels + bottom_c;
    if (argmax_data[top_index] == bottom_index)
      bottom_diff[bottom_index] = top_diff[top_index];
  }
}

template <typename Dtype>
__global__ void MultiScaledCrossRegionsAveragePoolBackward(
  const int bottom_count, const int bottom_channels, const int top_channels,
  const Dtype *bottom_rois, const int *roi_scale_ids_data, const int *pool_counts,
  const Dtype* top_diff, Dtype *bottom_diff) {

  CUDA_KERNEL_LOOP(bottom_index, bottom_count) {
    const int roi_id = bottom_index / bottom_channels;
    const int bottom_c = bottom_index % bottom_channels;

	const int roi_image_id = bottom_rois[5 * roi_id + 0];
    const int roi_scale_id = roi_scale_ids_data[roi_id];
    const int top_index = roi_image_id * top_channels + roi_scale_id * bottom_channels + bottom_c;
    if (pool_counts[top_index] > 0)
      bottom_diff[bottom_index] = top_diff[top_index] / Dtype(pool_counts[top_index]);
  }
}

template <typename Dtype>
void CrossRegionsPoolingLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0])
    return;

  const int bottom_count = bottom[0]->count();
  const Dtype *bottom_rois = bottom[1]->gpu_data();
  const int *roi_scale_ids_data = roi_scale_ids_.gpu_data();
  const int *argmax_data = max_idx_.gpu_data();
  const int *pool_counts = argmax_data;
  const Dtype *top_diff = top[0]->gpu_diff();

  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);

  const int count = bottom[0]->count();
  const int num_rois = bottom[1]->num();

  switch (pool_method_) {
    case CrossRegionsPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      MultiScaledCrossRegionsMaximumPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, bottom_channels_, top_channels_,
        bottom_rois, roi_scale_ids_data, argmax_data,
        top_diff, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
      break;
    case CrossRegionsPoolingParameter_PoolMethod_AVE:
      // NOLINT_NEXT_LINE(whitespace/operators)
      MultiScaledCrossRegionsAveragePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, bottom_channels_, top_channels_,
        bottom_rois, roi_scale_ids_data, pool_counts,
        top_diff, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
      break;
    default:
      break;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossRegionsPoolingLayer);

}  // namespace caffe
