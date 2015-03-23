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
#include <boost/random.hpp>
#include <time.h>

using std::ifstream;

namespace caffe {

template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureChannels(const int ix) {
  return this->layer_param_.binary_data_param().binary_features(ix).channels();
}

template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureHeight(const int ix) {
  return this->layer_param_.binary_data_param().binary_features(ix).height();
}
template <typename Dtype>
int BinaryDataLayer<Dtype>::GetFeatureWidth(const int ix) {
  return this->layer_param_.binary_data_param().binary_features(ix).width();
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::CheckFeatureSizeIntegrity() {
  for (int ix = 1; ix < this->layer_param_.binary_data_param().binary_features_size(); ++ix) {
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
  for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_features_size(); ++ix) {
    const string feature_list_file = this->layer_param_.binary_data_param().binary_features(ix).list_file();
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
  for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_features_size(); ++ix) {
    const int feature_dim = GetFeatureChannels(ix) * GetFeatureHeight(ix) * GetFeatureWidth(ix);
    Dtype * feature_mean = new Dtype[feature_dim]();
    feature_means_.push_back(feature_mean);
    const string feature_mean_file = this->layer_param_.binary_data_param().binary_features(ix).mean_file();
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
void BinaryDataLayer<Dtype>::SetSkipSize() {
  const int base_skip_size = this->layer_param_.binary_data_param().skip_size();
  for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_features_size(); ++ix) {
    if (!this->layer_param_.binary_data_param().binary_features(ix).has_skip_size())
      this->layer_param_.mutable_binary_data_param()->mutable_binary_features(ix)->set_skip_size(base_skip_size);
  }
}

template <typename Dtype>
void BinaryDataLayer<Dtype>::SetDatumSize() {
  if (this->layer_param_.binary_data_param().merge_direction() == BinaryDataParameter_MergeDirection_WIDTH) {
    this->datum_channels_ = GetFeatureChannels(0);
    this->datum_height_ = this->layer_param_.binary_data_param().max_length();
    this->datum_width_ = 0;
    for (int ix = 0; ix < this->layer_param_.binary_data_param().binary_features_size(); ++ix)
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
  if (this->layer_param_.binary_data_param().merge_direction() == BinaryDataParameter_MergeDirection_WIDTH) {
    /* open feature files */
    int num_open_file = 0;
    std::ifstream *feature_streams = new std::ifstream[feature_files_.size()];
    for (int ix = 0; ix < feature_files_.size(); ++ix) {
      const string file = this->layer_param_.binary_data_param().binary_features(ix).root_dir() + "/" + feature_files_[ix][lines_id];
      feature_streams[ix].open(file.c_str(), std::ios::in | std::ios::binary);
      if (feature_streams[ix].is_open()) {
        ++num_open_file;
      } else {
        for (int file_id = 0; file_id < feature_files_.size(); ++file_id) {
          if (feature_streams[ix].is_open()) {
            feature_streams[ix].close();
            --num_open_file;
          }
        }
        if (Caffe::phase() == Caffe::TRAIN) {
          LOG(ERROR) << "file " << file << " was skipped because it could not be opened";
          return false;
        } else if (Caffe::phase() == Caffe::TEST) {
          LOG(FATAL) << "errno: " << errno << "; could not open feature file : " << file;
        }
      }

      /* sample at random start */
      const float random_start_ratio = this->layer_param_.binary_data_param().random_start_ratio();
      CHECK(random_start_ratio >= 0 && random_start_ratio <= 1);
      if (random_start_ratio > 0) {
        feature_streams[ix].seekg(0, std::ios::end);
        const std::streampos file_size = feature_streams[ix].tellg();
        const std::streampos feature_size = sizeof(Dtype) * GetFeatureChannels(ix) * GetFeatureHeight(ix) * GetFeatureWidth(ix);
        const unsigned long int num_instance = file_size / feature_size;

        boost::mt19937 rng = boost::mt19937();
        const unsigned long t = clock();
        rng.seed(t);
        const unsigned long int max_num = (long int)((num_instance - 1) * random_start_ratio);
        boost::uniform_int<> range(0, max_num);
        boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(rng, range);
        const std::streampos offset_size = die() * feature_size;

        DLOG(INFO) << "num_instacne: " << num_instance << ", offset: " << offset_size / feature_size << " / " << offset_size;

        feature_streams[ix].seekg(offset_size, std::ios::beg);
      }
    }

    /* fill into top data */
    Dtype* top_data = this->prefetch_data_.mutable_cpu_data() + this->prefetch_data_.offset(batch_item_id);
    for (int c = 0; c < this->datum_channels_; ++c) {
      for (int h = 0; h < this->datum_height_; ++h) {
        for (int ix = 0; ix < feature_files_.size(); ++ix) {
          const int feature_width = GetFeatureWidth(ix);
          bool read = false;
          /* for the read binary feature height within datum_height */
          if (feature_streams[ix].is_open()) {
            const unsigned long int feature_size = sizeof(Dtype) * feature_width;
            const size_t read_size = feature_streams[ix].read(reinterpret_cast<char *>(top_data), feature_size).gcount();

            if (read_size == feature_size) {  /* successful read from file */
              for (int w = 0; w < feature_width; ++w)
                top_data[w] -= feature_means_[ix][w];
              read = true;
              const int skip_size = this->layer_param_.binary_data_param().binary_features(ix).skip_size();
              feature_streams[ix].seekg(sizeof(Dtype) * feature_width * skip_size, std::ios::cur);
            } else if (!feature_streams[ix].eof()) {  /* less than feature width was read */
              for (int file_id = 0; file_id < feature_files_.size(); ++file_id) {
                if (feature_streams[ix].is_open()) {
                  feature_streams[ix].close();
                  --num_open_file;
                }
              }
              const string file = this->layer_param_.binary_data_param().binary_features(ix).root_dir() + "/" + feature_files_[ix][lines_id];
              if (Caffe::phase() == Caffe::TRAIN) {
                LOG(ERROR) << "file " << file << " was skipped because of incorrect feature dimension";
                return false;
              } else if (Caffe::phase() == Caffe::TEST) {
                LOG(FATAL) << "encounter incorrect feature file dimension in file: " << file;
              }
            }
            if (feature_streams[ix].eof() || h == this->datum_height_ - 1) {
              feature_streams[ix].close();
              --num_open_file;
            }
          }

          /* for the remaining datum_height greater than binary feature height */
          if (!read) {
            for (int w = 0; w < feature_width; ++w)
              top_data[w] = -feature_means_[ix][w];
          }

          top_data += feature_width;
        }
      }
    }

    CHECK(num_open_file == 0) << num_open_file << " files not closed.";
    delete[] feature_streams;
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

  SetSkipSize();

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
