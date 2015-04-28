#ifndef CAFFE_DATA_LAYERS_EXTRA_HPP_
#define CAFFE_DATA_LAYERS_EXTRA_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class BinaryDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BinaryDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~BinaryDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_BINARY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
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
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const { return LayerParameter_LayerType_OMNI_DATA; }
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
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_BINARY_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
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
