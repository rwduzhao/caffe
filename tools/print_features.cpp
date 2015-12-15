#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  const int num_required_args = 5;
  int arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "using Device_id = " << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;
  string pretrained_binary_proto(argv[++arg_pos]);
  string feature_extraction_proto(argv[++arg_pos]);
  Net<float> feature_extraction_net(feature_extraction_proto, caffe::TEST);
  feature_extraction_net.CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_names(argv[++arg_pos]);
  vector<string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net.has_blob(blob_names[i]))
      << "unknown feature blob name " << blob_names[i]
      << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "extracting features ...";
  vector<Blob<float>*> input_vec;
  vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net.Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net.blob_by_name(blob_names[i]);
      const int batch_size = feature_blob->num();
      const int dim_features = feature_blob->count() / batch_size;
      Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(n);
        int data_index = 0;
        for (int c = 0; c < feature_blob->channels(); ++c) {
          std::cout << "layer:" << blob_names[i] << ", c:" << c << std::endl;
          for (int h = 0; h < feature_blob->height(); ++h) {
            for (int w = 0; w < feature_blob->width(); ++w) {
              std::cout << feature_blob_data[data_index++];
              if (w + 1 != feature_blob->width())
                std::cout << " ";
              else
                std::cout << std::endl;
            }
          }
        }
        CHECK(dim_features == data_index) << "feature dimensions mismatch";
        std::cout << std::endl;
        ++image_indices[i];
      }
    }
  }
  LOG(ERROR)<< "successfully extracted the features";

  return 0;
}
