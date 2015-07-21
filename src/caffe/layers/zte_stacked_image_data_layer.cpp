#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/data_layers_extra.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/io_extra.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <boost/random.hpp>
#include <time.h>

using std::ifstream;

namespace caffe {

template <typename Dtype>
ZteStackedImageDataLayer<Dtype>::~ZteStackedImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ZteStackedImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReadSourceListToLines();

  SetDatumSize();

  // set top data size
  const int batch_size = this->layer_param_.zte_stacked_image_data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, this->datum_channels_, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, this->datum_channels_, crop_size, crop_size);
    this->transformed_data_.Reshape(1, this->datum_channels_, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, this->datum_channels_, this->datum_height_, this->datum_width_);
    this->prefetch_data_.Reshape(batch_size, this->datum_channels_, this->datum_height_, this->datum_width_);
    this->transformed_data_.Reshape(1, this->datum_channels_, this->datum_height_, this->datum_width_);
  }

  // set top label size
  const vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);

  // shuffle data order
  if (this->layer_param_.zte_stacked_image_data_param().shuffle()) {
    LOG(INFO) << "now shuffling data ...";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLines();
  }

  lines_id_ = 0;
}

// this function is used to create a thread that prefetches the data.
template <typename Dtype>
void ZteStackedImageDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  const int batch_size = this->layer_param_.zte_stacked_image_data_param().batch_size();
  const int num_line = lines_.size();
  for (int batch_item_id = 0; batch_item_id < batch_size; ++batch_item_id) {
    CHECK_GT(num_line, lines_id_);
    ReadSourceToTop(lines_id_, batch_item_id);
    ++lines_id_;
    if (lines_id_ >= num_line) {
      LOG(INFO) << "restarting data prefetching from start";
      lines_id_ = 0;
      if (this->layer_param_.zte_stacked_image_data_param().shuffle())
        ShuffleLines();
    }
  }
}

template <typename Dtype>
void ZteStackedImageDataLayer<Dtype>::ReadSourceListToLines() {
  const string & source = this->layer_param_.zte_stacked_image_data_param().source();
  LOG(INFO) << "opening souce file (filename + label): " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "could not open source file (filename + label): " << source;

  lines_.clear();
  string filename;
  int label;
  while (infile >> filename >> label)
    lines_.push_back(std::make_pair(filename, label));

  CHECK(!lines_.empty()) << "file list is empty (filename: \"" + source + "\")";
  LOG(INFO) << "a total number of " << lines_.size() << " items in the source file";
}

template <typename Dtype>
void ZteStackedImageDataLayer<Dtype>::SetDatumSize() {
  const int num_stack = this->layer_param_.zte_stacked_image_data_param().num_stack();
  const int image_color = this->layer_param_.zte_stacked_image_data_param().image_color();
  if (image_color == ZteStackedImageDataParameter_ImageColor_GRAY)
    this->datum_channels_ = 1 * num_stack;
  else if (image_color == ZteStackedImageDataParameter_ImageColor_RGB)
    this->datum_channels_ = 3 * num_stack;
  else
    LOG(FATAL) << "unsupported image color";

  this->datum_height_ = this->layer_param_.zte_stacked_image_data_param().new_height();
  this->datum_width_ = this->layer_param_.zte_stacked_image_data_param().new_width();

  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;
}

template <typename Dtype>
void ZteStackedImageDataLayer<Dtype>::ShuffleLines() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
bool ZteStackedImageDataLayer<Dtype>::ReadSourceToTop(const int lines_id, const int batch_item_id) {
  // label
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int label = lines_[lines_id].second;
  top_label[batch_item_id] = std::max(0, label);

  // read stacked image files list from source line
  const string source_item_prefix = this->layer_param_.zte_stacked_image_data_param().source_item_prefix();
  const string source_item_file = this->lines_[lines_id].first;
  const string stacking_list_file_path = source_item_prefix + source_item_file;
  std::ifstream infile(stacking_list_file_path.c_str());
  if (!infile.good()) {
    LOG(ERROR) << "could not open file : " << stacking_list_file_path;
    return false;
  }
  std::vector<string> stacked_image_files;
  string filename;
  while (infile >> filename)
    stacked_image_files.push_back(filename);

  // init datum for stacked images
  Datum datum;
  datum.set_channels(this->datum_channels_);
  datum.set_height(this->datum_height_);
  datum.set_width(this->datum_width_);
  datum.set_label(label);
  datum.clear_data();
  datum.clear_float_data();

  // read images into datum
  const int num_stack = this->layer_param_.zte_stacked_image_data_param().num_stack();
  int x0 = -1;
  int y0 = -1;
  int diff_x = 1E6;
  int diff_y = 1E6;
  double scale = 0.0;
  for (int image_ix = 0; image_ix < std::min((int)stacked_image_files.size(), num_stack); ++image_ix) {
    const string stacked_image_prefix = this->layer_param_.zte_stacked_image_data_param().stacked_image_prefix();
    const string image_path = stacked_image_prefix + stacked_image_files[image_ix];
    const int new_height = this->layer_param_.zte_stacked_image_data_param().new_height();
    const int new_width = this->layer_param_.zte_stacked_image_data_param().new_width();
    const int image_color = this->layer_param_.zte_stacked_image_data_param().image_color();
    const bool is_color = image_color != ZteStackedImageDataParameter_ImageColor_GRAY;
    if (!ReadZteImageToPrespecifiedDatum(image_path, label, new_height, new_width, is_color, &datum,
                                         x0, y0, diff_x, diff_y, scale)) {
      const int num_channels = (is_color ? 3 : 3);  // TODO 3 : 1
      string *datum_string = datum.mutable_data();
      for (int ix = 0; ix < num_channels * new_height * new_width; ++ix)
        datum_string->push_back(static_cast<char>(0));
    }
  }
  string *datum_string = datum.mutable_data();
  DLOG(INFO) << "old datum_string length: " << datum_string->length();
  const int num_append = datum.channels() * datum.height() * datum.width() - datum_string->length();
  for (int id = 0; id < num_append; ++id)
    datum_string->push_back(static_cast<char>(0));  // append unfilled
  DLOG(INFO) << "new datum_string length: " << datum_string->length();
  CHECK(datum_string->length() == datum.channels() * datum.height() * datum.width())
    << datum_string->length() << " vs " << datum.channels() << " " << datum.height() << " " << datum.width()
    << " = " << datum.channels() * datum.height() * datum.width();
  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  const int offset = this->prefetch_data_.offset(batch_item_id);
  this->transformed_data_.set_cpu_data(prefetch_data + offset);
  this->data_transformer_->Transform(datum, &(this->transformed_data_));

  return true;
}

INSTANTIATE_CLASS(ZteStackedImageDataLayer);
REGISTER_LAYER_CLASS(ZteStackedImageData);

}  // namespace caffe
