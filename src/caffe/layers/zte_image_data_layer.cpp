#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <random>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

#include "caffe/data_layers_extra.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/io_extra.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ZteImageDataLayer<Dtype>::~ZteImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ZteImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.zte_image_data_param().new_height();
  const int new_width  = this->layer_param_.zte_image_data_param().new_width();
  const bool is_color  = this->layer_param_.zte_image_data_param().is_color();
  string root_folder = this->layer_param_.zte_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.zte_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  const int phase = this->layer_param().phase();
  while (infile >> filename >> label) {
    if (label != 0 || phase == caffe::TEST)
      lines_.push_back(std::make_pair(filename, label));
    else
      bg_lines_.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines_.size() << " foreground images.";
  LOG(INFO) << "A total of " << bg_lines_.size() << " background images.";

  if (this->layer_param_.zte_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.zte_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.zte_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.zte_image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void ZteImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(bg_lines_.begin(), bg_lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ZteImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  ZteImageDataParameter zte_image_data_param = this->layer_param_.zte_image_data_param();
  const int batch_size = zte_image_data_param.batch_size();
  const int new_height = zte_image_data_param.new_height();
  const int new_width = zte_image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = zte_image_data_param.is_color();
  const string root_folder = zte_image_data_param.root_folder();
  const bool background_crop_square = zte_image_data_param.background_crop_square();

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, 0, 0, is_color);
    this->prefetch_data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  shared_ptr<Caffe::RNG> rng_ptr;
  const unsigned int rng_seed = caffe_rng_rand();
  rng_ptr.reset(new Caffe::RNG(rng_seed));
  caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_ptr->generator());

  // datum scales
  const int num_line = lines_.size();
  const int num_bg_line = bg_lines_.size();
  int read_lines_id = lines_id_;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(num_line, lines_id_);

    cv::Mat cv_img;
    const int phase = this->layer_param().phase();
    if (phase == caffe::TRAIN) {
      if (read_lines_id % 5 != 0 || num_bg_line == 0) {  // foreground
        DLOG(INFO) << "foreground " << lines_id_;
        const string fg_image_filename = root_folder + lines_[lines_id_].first;
        cv_img = ReadImageToCVMat(fg_image_filename, new_height, new_width, is_color);
        prefetch_label[item_id] = std::max(0, lines_[lines_id_].second);
        ++lines_id_;
      } else if (num_bg_line > 0) {  // background
        const int bg_lines_id = ((*rng)() % num_bg_line);
        DLOG(INFO) << "background " << bg_lines_id;
        const string bg_image_filename = root_folder + bg_lines_[bg_lines_id].first;
        if ((*rng)() % 4 == 0) {
          int x0 = -1;
          int y0 = -1;
          int diff_x = 1E6;
          int diff_y = 1E6;
          double scale = 0.0;
          ZteCropBackground(bg_image_filename, cv_img, 360, 360, x0, y0, diff_x, diff_y, scale);
        } else {
          const int location_id = ((*rng)() % num_line);
          vector<int> crop_location = GetCropLocationFromZteImageFilename(lines_[location_id].first);
          const int crop_x_center = crop_location[0];
          const int crop_y_center = crop_location[1];
          const int crop_x0 = std::max(0, crop_x_center - crop_size / 2);
          const int crop_y0 = std::max(0, crop_y_center - crop_size / 2);
          const int crop_width = crop_location[2];
          const int crop_height = crop_location[3];
          const int crop_size = crop_location[4];
          if (background_crop_square) {
            cv_img = ZteCropBackground(bg_image_filename, new_height, new_width, is_color,
                                       crop_x0, crop_y0, crop_size, crop_size);
          } else {
            cv_img = ZteCropBackground(bg_image_filename, new_height, new_width, is_color,
                                       crop_x0, crop_y0, crop_width, crop_height);
          }
        }
        prefetch_label[item_id] = std::max(0, bg_lines_[bg_lines_id].second);
      }
    } else if (phase == caffe::TEST) {
      const string fg_image_filename = root_folder + lines_[lines_id_].first;
      cv_img = ReadImageToCVMat(fg_image_filename, new_height, new_width, is_color);
      prefetch_label[item_id] = std::max(0, lines_[lines_id_].second);
      ++lines_id_;
    }

    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    // go to the next iter
    ++read_lines_id;
    if (lines_id_ >= num_line) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      read_lines_id = 0;
      if (this->layer_param_.zte_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ZteImageDataLayer);
REGISTER_LAYER_CLASS(ZteImageData);

}  // namespace caffe
