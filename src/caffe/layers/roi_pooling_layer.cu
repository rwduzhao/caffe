// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include "caffe/layers/roi_pooling_layer.hpp"

#define CUDA_RETURN return;

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void MakeRoiPositionMap(const int nthreads,
  const int map_height, const int map_width,
  const int bottom_height, const int bottom_width,
  const Dtype* bottom_rois, Dtype* map_data) {
  // map size scales
  const Dtype map_height_scale = static_cast<Dtype>(map_height) / static_cast<Dtype>(bottom_height);
  const Dtype map_width_scale = static_cast<Dtype>(map_width) / static_cast<Dtype>(bottom_width);

  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_rois += index * 5;
    const int batch_ind = bottom_rois[0];
    const int map_start_w = max(0, int(round(bottom_rois[1] * map_width_scale)));
    const int map_start_h = max(0, int(round(bottom_rois[2] * map_height_scale)));
    const int map_end_w = min(int(round(bottom_rois[3] * map_width_scale)), map_width);
    const int map_end_h = min(int(round(bottom_rois[4] * map_height_scale)), map_height);

    map_data += batch_ind * map_height * map_width;
    for (int h = map_start_h; h <= map_end_h; ++h) {
      for (int w = map_start_w; w <= map_end_w; ++w) {
        const int map_index = h * map_width + w;
        map_data[map_index] = Dtype(1.);
      }
    }
  }
}

template <typename Dtype>
__global__ void MakeRoiShapeMap(const int nthreads,
  const int map_height, const int map_width,
  const Dtype* bottom_rois, Dtype* map_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_rois += index * 5;
    const int batch_ind = static_cast<int>(bottom_rois[0]);
    const int roi_height = static_cast<int>(bottom_rois[4] - bottom_rois[2] + 1);
    const int roi_width = static_cast<int>(bottom_rois[3] - bottom_rois[1] + 1);

    const int max_roi_side = max(roi_height, roi_width);
    const Dtype height_ratio = static_cast<Dtype>(roi_height) / static_cast<Dtype>(max_roi_side);
    const Dtype width_ratio = static_cast<Dtype>(roi_width) / static_cast<Dtype>(max_roi_side);
    const int mapped_height = max(1, static_cast<int>(height_ratio * static_cast<Dtype>(map_height)));
    const int mapped_width = max(1, static_cast<int>(width_ratio * static_cast<Dtype>(map_width)));

    const int mapped_h_start = (map_height - mapped_height) / 2;
    const int mapped_h_end = mapped_h_start + mapped_height - 1;
    const int mapped_w_start = (map_width - mapped_width) / 2;
    const int mapped_w_end = mapped_w_start + mapped_width - 1;

    map_data += batch_ind * map_height * map_width;
    for (int h = mapped_h_start; h <= mapped_h_end; ++h) {
      for (int w = mapped_w_start; w <= mapped_w_end; ++w) {
        const int map_index = h * map_width + w;
        map_data[map_index] = Dtype(1.);
      }
    }
  }
}

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
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
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
  switch (roi_pooling_param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
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

  const int num_top = top.size();
  if (num_top > 1) {
    const int bottom_height = bottom[0]->height();
    const int bottom_width = bottom[0]->width();
    const int num_roi = bottom[1]->num();
    Dtype* position_map_data = top[1]->mutable_gpu_data();
    caffe_gpu_set(top[1]->count(), Dtype(0.), position_map_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    MakeRoiPositionMap<Dtype><<<CAFFE_GET_BLOCKS(num_roi), CAFFE_CUDA_NUM_THREADS>>>(
      num_roi, position_map_height_, position_map_width_, bottom_height, bottom_width,
      bottom_rois, position_map_data);
    CUDA_POST_KERNEL_CHECK;
  }
  if (num_top > 2) {
    const int num_roi = bottom[1]->num();
    Dtype* shape_map_data = top[2]->mutable_gpu_data();
    caffe_gpu_set(top[2]->count(), Dtype(0.), shape_map_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    MakeRoiShapeMap<Dtype><<<CAFFE_GET_BLOCKS(num_roi), CAFFE_CUDA_NUM_THREADS>>>(
      num_roi, shape_map_height_, shape_map_width_, bottom_rois, shape_map_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
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
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();

  ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
  switch (roi_pooling_param.pool()) {
    case ROIPoolingParameter_PoolMethod_MAX:
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
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
