#ifndef CAFFE_DATA_LAYERS_EXTRA_HPP_
#define CAFFE_DATA_LAYERS_EXTRA_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class BinaryDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BinaryDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~BinaryDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinaryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;

  virtual void ShuffleBinarys();
  virtual void InternalThreadEntry();

  virtual int GetFeatureChannels(const int id);
  virtual int GetFeatureHeight(const int id);
  virtual int GetFeatureWidth(const int id);
  virtual void CheckFeatureSizeIntegrity();
  virtual void ReadSourceListToLines();
  virtual void ReadFeatureFiles();
  virtual void ReadFeatureMeans();
  virtual void SetSkipSize();
  virtual void SetDatumSize();
  virtual bool ReadBinariesToTop(const int lines_id, const int batch_item_id);

  vector<std::pair<std::string, int> > lines_;  /*  data name, label  */
  vector<vector<std::string> > feature_files_;  /*  feature sizes, feature file names  */
  vector<Dtype *> feature_means_;
  int lines_id_;
};

template <typename Dtype>
class OmniDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit OmniDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~OmniDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OmniData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, int> > lines_;  /*  data name, label  */
  vector<vector<std::string> > feature_files_;  /*  feature sizes, feature file names  */
  vector<Dtype *> feature_means_;
  int lines_id_;
};

template <typename Dtype>
class StackedImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit StackedImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~StackedImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StackedImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;

  virtual void ShuffleLines();
  virtual void InternalThreadEntry();

  virtual void ReadSourceListToLines();
  virtual void SetDatumSize();
  virtual bool ReadSourceToTop(const int lines_id, const int batch_item_id);

  vector<std::pair<std::string, int> > lines_;  // data name, label
  vector<vector<std::string> > feature_files_;  // feature sizes, feature file names
  vector<Dtype *> feature_means_;
  int lines_id_;
};

template <typename Dtype>
class ZteImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ZteImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ZteImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ZteImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};

template <typename Dtype>
class ZteStackedImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ZteStackedImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ZteStackedImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ZteStackedImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;

  virtual void ShuffleLines();
  virtual void InternalThreadEntry();

  virtual void ReadSourceListToLines();
  virtual void SetDatumSize();
  virtual bool ReadSourceToTop(const int lines_id, const int batch_item_id);

  vector<std::pair<std::string, int> > lines_;  // data name, label
  vector<vector<std::string> > feature_files_;  // feature sizes, feature file names
  vector<Dtype *> feature_means_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_EXTRA_HPP_
