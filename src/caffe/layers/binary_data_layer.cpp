#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureChannels(const int ix) {
  return this->layer_param_.binary_data_param().binary_feature(ix).channels();
}

template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureHeight(const int ix) {
  return this->layer_param_.binary_data_param().binary_feature(ix).height();
}
template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureWidth(const int ix) {
  return this->layer_param_.binary_data_param().binary_feature(ix).width();
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::CheckFeatureSizeIntegrity() {
  for (int ix = 1; ix < this->layer_param_.binary_data_param().binary_feature_size(); ++ix) {
    if (this->layer_param_.binary_data_param().merge_direction() == BinaryDataParameter_MergeDirection_WIDTH) {
      CHECK(GetFeatureChannels(0) == GetFeatureChannels(ix));
      CHECK(GetFeatureHeight(0) == GetFeatureHeight(ix));
    } else
      LOG(FATAL) << "unrecgnized merge direction";
  }
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::ReadSourceListToLines() {
  const string & source = this->layer_param_.binary_data_param().source();
  lines_.clear();

  LOG(INFO) << "opening souce file (filename + label): " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good()) << "could not open source file (filename + label): " << source;

  string filename;
  int label;
  while (infile >> filename >> label)
    lines_.push_back(std::make_pair(filename, label));

  LOG(INFO) << "a total number of " << lines_.size() << " files.";
  CHECK(!lines_.empty()) << "file list is empty (filename: \"" + source + "\")";
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::ReadFeatureFiles() {
  feature_files_.clear();
  for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_feature_size(); ++ix) {
    const string feature_list_file = this->layer_param_.binary_data_param().binary_feature(ix).list_file();
    std::ifstream feature_infile(feature_list_file.c_str());
    CHECK(feature_infile.good()) << "could not open feature list file: " << feature_list_file;
    string feature_file_name;
    vector<std::string> feature_file_names;
    while (feature_infile >> feature_file_name)
      feature_file_names.push_back(feature_file_name);
    CHECK(feature_file_names.size() == lines_.size()) << "number of files and feature files not match.";
    feature_files_.push_back(feature_file_names);
  }
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::ReadFeatureMeans() {
  feature_means_.clear();
  for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_feature_size(); ++ix) {
    const int feature_dim = GetFeatureChannels(ix) * GetFeatureHeight(ix) * GetFeatureWidth(ix);
    Dtype * feature_mean = new Dtype[feature_dim]();
    feature_means_.push_back(feature_mean);
    const string feature_mean_file = this->layer_param_.binary_data_param().binary_feature(ix).mean_file();
    if (!feature_mean_file.empty()) {
      FILE * fd = NULL;
      fd = fopen(feature_mean_file.c_str(), "rb");
      CHECK(fd != NULL) << "could not open mean file: " << feature_mean_file;
      Dtype value;
      int mean_id = 0;
      while (fread(&value, sizeof(Dtype), 1, fd) > 0) {
        feature_mean[mean_id++] = value;
      }
      CHECK(mean_id == feature_dim) << "feature dim and mean dim mismatch.";
      fclose(fd);
      fd = NULL;
    } else
      LOG(INFO) << "no mean file (means set to zeros).";
  }
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::SetDatumSize() {
  if (this->layer_param_.binary_data_param().merge_direction() == BinaryDataParameter_MergeDirection_WIDTH) {
    this->datum_channels_ = GetFeatureChannels(0);
    this->datum_height_ = this->layer_param_.binary_data_param().max_length();
    this->datum_width_ = 0;
    for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_feature_size(); ++ix)
      this->datum_width_ += GetFeatureWidth(ix);
  } else
    LOG(FATAL) << "unrecgnized merge direction";

  this->datum_size_ = this->datum_channels_ * this->datum_height_ * this->datum_width_;
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::ShuffleBinarys() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  caffe::rng_t* dup_rngs = (caffe::rng_t*)malloc(sizeof(caffe::rng_t) * feature_files_.size());
  for (int ix = 0; ix < feature_files_.size(); ++ix)
    memcpy(&dup_rngs[ix], prefetch_rng, sizeof(caffe::rng_t));

  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  for (int ix = 0; ix < feature_files_.size(); ++ix)
    shuffle(feature_files_[ix].begin(), feature_files_[ix].end(), &dup_rngs[ix]);

  free(dup_rngs);
}

template <typename Dtype>
bool BinaryDataLayer<Dtype>::ReadBinariesToTop(const int lines_id, const int batch_item_id) {
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  if (this->layer_param_.binary_data_param().merge_direction() == BinaryDataParameter_MergeDirection_WIDTH) {
    /* open feature files */
    vector<FILE *> p_feature_files;
    for (int ix = 0; ix < feature_files_.size(); ++ix) {
      FILE * fd = NULL;
      const string feature_file = this->layer_param_.binary_data_param().binary_feature(ix).root_dir() + "/" + feature_files_[ix][lines_id];
      fd = fopen(feature_file.c_str(), "rb");
      CHECK(fd != NULL) << "could not open feature file: " << feature_file;
      p_feature_files.push_back(fd);
    }

    /* fill into top data */
    top_data += batch_item_id * this->datum_size_;
    for (int c = 0; c < this->datum_channels_; ++c) {
      for (int h = 0; h < this->datum_height_; ++h) {
        for (int ix = 0; ix < feature_files_.size(); ++ix) {
          const int feature_width = GetFeatureWidth(ix);

          if (p_feature_files[ix] != NULL) {
            const size_t read_size = fread(top_data, sizeof(Dtype), feature_width, p_feature_files[ix]);
            if (read_size == feature_width) {
              for (int w = 0; w < feature_width; ++w)
                top_data[w] -= feature_means_[ix][w];
            } else if (read_size == 0) {
              fclose(p_feature_files[ix]);
              p_feature_files[ix] = NULL;
            } else
              LOG(FATAL) << "invalid feature width in " << lines_[lines_id].first
                << " (" << read_size << " against " << feature_width << ")";
          }

          if (p_feature_files[ix] == NULL) {
            for (int w = 0; w < feature_width; ++w)
              top_data[w] = 0 - feature_means_[ix][w];
          }

          top_data += feature_width;
        }
      }
    }
  } else
    LOG(FATAL) << "unsupported merge direction";

  /* fill into top data */
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  top_label[batch_item_id] = lines_[lines_id].second;

  return true;
}

template <typename Dtype>
BinaryDataLayer<Dtype>::~BinaryDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  for (int ix = 0; ix < feature_means_.size(); ++ix)
    delete[] feature_means_[ix];
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CheckFeatureSizeIntegrity();
  ReadSourceListToLines();
  ReadFeatureFiles();
  ReadFeatureMeans();

  SetDatumSize();
  const int batch_size = this->layer_param_.binary_data_param().batch_size();

  (*top)[0]->Reshape(batch_size, this->datum_channels_, this->datum_height_, this->datum_width_);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << "," << (*top)[0]->channels() << "," << (*top)[0]->height() << "," << (*top)[0]->width();
  this->prefetch_data_.Reshape(batch_size, this->datum_channels_, this->datum_height_, this->datum_width_);

  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

  if (this->layer_param_.binary_data_param().shuffle()) {
    LOG(INFO) << "shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleBinarys();
  }

  lines_id_ = 0;
}

// this function is used to create a thread that prefetches the data.
template <typename Dtype>
void BinaryDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  const int batch_size = this->layer_param_.binary_data_param().batch_size();
  const int lines_size = lines_.size();
  for (int batch_item_id = 0; batch_item_id < batch_size; ++batch_item_id) {
    CHECK_GT(lines_size, lines_id_);
    if (!ReadBinariesToTop(lines_id_, batch_item_id))
      continue;

    ++lines_id_;

    if (lines_id_ >= lines_size) {
      DLOG(INFO) << "restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.binary_data_param().shuffle())
        ShuffleBinarys();
    }
  }
}

INSTANTIATE_CLASS(BinaryDataLayer);

}  // namespace caffe
