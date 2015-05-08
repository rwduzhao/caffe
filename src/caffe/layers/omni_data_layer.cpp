#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/data_layers_extra.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
OmniDataLayer<Dtype>::~OmniDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void OmniDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

// this function is used to create a thread that prefetches the data.
template <typename Dtype>
void OmniDataLayer<Dtype>::InternalThreadEntry() {
}

INSTANTIATE_CLASS(OmniDataLayer);
REGISTER_LAYER_CLASS(OmniData);

}  // namespace caffe
