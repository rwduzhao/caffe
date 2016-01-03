// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Rui-Wei Zhao
// ------------------------------------------------------------------

#include <cfloat>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/roi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // layer parameters
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();

  // pooled sizes
  pooled_height_ = roi_pool_param.pooled_h();
  CHECK_GT(pooled_height_, 0) << "pooled height must be greater 0";
  pooled_width_ = roi_pool_param.pooled_w();
  CHECK_GT(pooled_width_, 0) << "pooled width must be greater 0";

  // spatial scale
  spatial_scale_ = roi_pool_param.spatial_scale();
  CHECK_GT(spatial_scale_, 0) << "spatial scale must be greater 0";

  // position map sizes
  position_map_height_ = roi_pool_param.position_map_height();
  CHECK_GE(position_map_height_, 0) << "position map height must be greater or equal to 0";
  position_map_width_ = roi_pool_param.position_map_width();
  CHECK_GE(position_map_width_, 0) << "position map width must be greater or equal to 0";

  // shape map sizes
  shape_map_height_ = roi_pool_param.shape_map_height();
  CHECK_GE(shape_map_height_, 0) << "shape map height must be greater or equal to 0";
  shape_map_width_ = roi_pool_param.shape_map_width();
  CHECK_GE(shape_map_width_, 0) << "shape map width must be greater or equal to 0";
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // bottom feature map sizes
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  pooled_channels_ = channels_;
  top[0]->Reshape(bottom[1]->num(), pooled_channels_, pooled_height_, pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), pooled_channels_, pooled_height_, pooled_width_);

  const int num_top = top.size();
  if (num_top > 1) {
    top[1]->Reshape(bottom[1]->num(), 1, position_map_height_, position_map_width_);
  }
  if (num_top > 2) {
    top[2]->Reshape(bottom[1]->num(), 1, shape_map_height_, shape_map_width_);
  }
}

template <typename Dtype>
cv::Mat Roi2Image(const Dtype *data, const int data_height, const int data_width,
                  const int hstart, const int hend, const int wstart, const int wend) {
  const int height = hend - hstart;
  const int width = wend - wstart;
  bool is_empty = height <= 0 || width <= 0;

  cv::Mat image;
  if (is_empty) {
    image = cv::Mat(1, 1, CV_32FC1, cvScalar(0.));
  } else {
    image = cv::Mat(height, width, CV_32FC1, cvScalar(0.));
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int index = h * data_width + w;
        image.at<float>(h - hstart, w - wstart) = static_cast<float>(data[index]);
      }
    }
  }

  return image;
}

template <typename Dtype>
void Image2Data(const cv::Mat image, Dtype *data) {
  for (int h = 0; h < image.rows; ++h) {
    for (int w = 0; w < image.cols; ++w) {
      const int index = h * image.cols + w;
      data[index] = static_cast<Dtype>(image.at<float>(h, w));
    }
  }
}

template <typename Dtype>
bool ROIPoolingLayer<Dtype>::CheckRoisSanity(const Blob<Dtype> *image_blob,
                                             const Blob<Dtype> *roi_blob) {
  LOG(INFO) << "image blob size: "
    << image_blob->num() << ", " << image_blob->channels() << ", "
    << image_blob->height() << ", " << image_blob->width();
  const int image_height_limit = image_blob->height() * 16;
  const int image_width_limit = image_blob->width() * 16;

  const Dtype* roi_data = roi_blob->cpu_data();
  const int num_roi = roi_blob->num();
  for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
    const int roi_data_offset = roi_id * 5;

    const int roi_x0 = round(roi_data[roi_data_offset + 1]);
    const int roi_y0 = round(roi_data[roi_data_offset + 2]);
    const int roi_x1 = round(roi_data[roi_data_offset + 3]);
    const int roi_y1 = round(roi_data[roi_data_offset + 4]);

    LOG(INFO) << "roi data: ("
      << roi_x0 << ", " << roi_y0 << ", "
      << roi_x1 << ", " << roi_y1 << ")";
    CHECK_GT(roi_x0, 0 - 1);
    CHECK_GT(roi_y0, 0 - 1);
    CHECK_LT(roi_x1, image_width_limit + 2);
    CHECK_LT(roi_y1, image_height_limit + 2);
  }
  return true;
};

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CheckRoisSanity(bottom[0], bottom[1]);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    // printf("roi map size: (%d, %d)\n",
    //        roi_end_h - roi_start_h,
    //        roi_end_w - roi_start_w);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    ROIPoolingParameter roi_pooling_param = this->layer_param_.roi_pooling_param();
    switch (roi_pooling_param.pool()) {
      case ROIPoolingParameter_PoolMethod_MAX:  // {{{
        for (int c = 0; c < channels_; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Compute pooling region for this output unit:
              //  start (included) = floor(ph * roi_height / pooled_height_)
              //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
              int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
              int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
              int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
              int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

              hstart = min(max(hstart + roi_start_h, 0), height_);
              hend = min(max(hend + roi_start_h, 0), height_);
              wstart = min(max(wstart + roi_start_w, 0), width_);
              wend = min(max(wend + roi_start_w, 0), width_);

              bool is_empty = (hend <= hstart) || (wend <= wstart);

              const int pool_index = ph * pooled_width_ + pw;
              if (is_empty) {
                top_data[pool_index] = 0;
                argmax_data[pool_index] = -1;
              }

              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width_ + w;
                  if (batch_data[index] > top_data[pool_index]) {
                    top_data[pool_index] = batch_data[index];
                    argmax_data[pool_index] = index;
                  }
                }
              }
            }
          }
          // Increment all data pointers by one channel
          batch_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
          argmax_data += max_idx_.offset(0, 1);
        }
        break;
        // }}}
      case ROIPoolingParameter_PoolMethod_AVE:
        break;
      case ROIPoolingParameter_PoolMethod_INT:
        for (int c = 0; c < channels_; ++c) {
          int hstart = static_cast<int>(floor(static_cast<Dtype>(0) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(0) * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(pooled_height_) * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pooled_width_) * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          cv::Mat image = Roi2Image(batch_data, height_, width_, hstart, hend, wstart, wend);
          cv::Mat resized_image;
          cv::resize(image, resized_image, cv::Size(pooled_width_, pooled_height_),
                     1.0, 1.0, cv::INTER_LINEAR);
          Image2Data(resized_image, top_data);

          // increment all data pointers by one channel
          batch_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }
        break;
      default:
        break;
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe
