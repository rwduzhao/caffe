// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Rui-Wei Zhao
// ------------------------------------------------------------------

#include <cfloat>
#include "caffe/layers/roi_pooling_layer.hpp"

#define CUDA_RETURN return;

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIPoolForward_MAX(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    // start and end range on the bottom blob
    bottom_rois += n * 5;
    const int roi_batch_ind = bottom_rois[0];
    const int roi_start_w = round(bottom_rois[1] * spatial_scale);
    const int roi_start_h = round(bottom_rois[2] * spatial_scale);
    const int roi_end_w = round(bottom_rois[3] * spatial_scale);
    const int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // force rois to be at least 1x1
    const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // scale coefficients from pooled (top) to rois (bottom)
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    // calculate scan regions on the bottom for the mapped pixel on the top
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void ROIPoolForward_AVE(
  const int nthreads, const Dtype* bottom_data,
  const Dtype spatial_scale, const int channels, const int height,
  const int width, const int pooled_height, const int pooled_width,
  const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int top_index = index;
    // (n, c, ph, pw) is an element in the pooled output
    const int pw = top_index % pooled_width;
    const int ph = (top_index / pooled_width) % pooled_height;
    const int pc = (top_index / pooled_width / pooled_height) % channels;
    const int pn = top_index / pooled_width / pooled_height / channels;

    // scaled regions in the bottom data {{{
    bottom_rois += pn * 5;
    const int roi_batch_ind = bottom_rois[0];
    const int roi_start_w = round(bottom_rois[1] * spatial_scale);
    const int roi_start_h = round(bottom_rois[2] * spatial_scale);
    const int roi_end_w = round(bottom_rois[3] * spatial_scale);
    const int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // force malformed ROIs to be 1x1
    const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    const Dtype bin_size_h =
      static_cast<Dtype>(roi_height) /
      static_cast<Dtype>(pooled_height);
    const Dtype bin_size_w =
      static_cast<Dtype>(roi_width) /
      static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(
        static_cast<Dtype>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(
        static_cast<Dtype>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(
        static_cast<Dtype>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(
        static_cast<Dtype>(pw + 1) * bin_size_w));
    // add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    // }}} end scaled regions in the bottom data

    const bool is_empty = (hend <= hstart) || (wend <= wstart);
    Dtype maxval = 0.0;

    if (!is_empty) {
      const int pool_size = (hend - hstart) * (wend - wstart);
      bottom_data += (roi_batch_ind * channels + pc) * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          const int bottom_index = h * width + w;
          maxval += bottom_data[bottom_index];
        }
      }
      maxval /= pool_size;
    }

    top_data[top_index] = maxval;
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // CheckRoisSanity(bottom[0], bottom[1]);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois;
  ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
  switch (roi_pooling_param.region_type()) {
    case ROIPoolingParameter_RegionType_ROI:
      bottom_rois = bottom[1]->gpu_data();
      break;
    case ROIPoolingParameter_RegionType_FULL:
      bottom_rois = this->temp_blobs_[0].get()->mutable_gpu_data();
      break;
    default:
      break;
  }
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  switch (roi_pooling_param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForward_MAX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, spatial_scale_, channels_, height_, width_,
        pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
      CUDA_POST_KERNEL_CHECK;
      break;
    case ROIPoolingParameter_PoolMethod_AVE:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForward_AVE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, spatial_scale_, channels_, height_, width_,
        pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
      CUDA_POST_KERNEL_CHECK;
      break;
    case ROIPoolingParameter_PoolMethod_INT:
      break;
    default:
      break;
  }
}

template <typename Dtype>
__global__ void ROIPoolBackward_MAX(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // accumulate gradient over all rois that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void ROIPoolBackward_AVE(
  const int nthreads, const Dtype* top_diff,
  const int *argmax_data, const int num_rois, const Dtype spatial_scale,
  const int channels, const int height, const int width,
  const int pooled_height, const int pooled_width, Dtype *bottom_diff,
  const Dtype* bottom_rois) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      // const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

      // pooled regions of current bottom pixel
      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          gradient += offset_top_diff[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois;
  ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
  switch (roi_pooling_param.region_type()) {
    case ROIPoolingParameter_RegionType_ROI:
      bottom_rois = bottom[1]->gpu_data();
      break;
    case ROIPoolingParameter_RegionType_FULL:
      bottom_rois = this->temp_blobs_[0].get()->gpu_data();
      break;
    default:
      break;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();

  switch (roi_pooling_param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackward_MAX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
        height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
      CUDA_POST_KERNEL_CHECK;
      break;
    case ROIPoolingParameter_PoolMethod_AVE:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackward_AVE<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
        height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
      CUDA_POST_KERNEL_CHECK;
      break;
    case ROIPoolingParameter_PoolMethod_INT:
      break;
    default:
      break;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);

}  // namespace caffe
