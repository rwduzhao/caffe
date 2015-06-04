/*=============================================================================
#     FileName: extract_blobs.cpp
#   Desciption: extract blob content in the test data into binary files.
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-03-24 23:44:23
#      History:
=============================================================================*/
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <gflags/gflags.h>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"

DEFINE_int32(gpu, 0, "gpu device");
DEFINE_string(blob, "", "blob names to be extracted");
DEFINE_string(model, "", "trained model file");
DEFINE_string(net, "", "net prototxt file");
DEFINE_string(outfile, "", "output file");
DEFINE_string(prefix, "", "output prefix");
DEFINE_uint64(disp, 0, "display interval");
DEFINE_uint64(num, 0, "number of instances");

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /* configure caffe */
  const int device_id = FLAGS_gpu;
  if (device_id >= 0) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(device_id);
  } else {
    Caffe::set_mode(Caffe::CPU);
  }

  /* net param and model */
  const string net_param_file = FLAGS_net;
  Net<float> net(net_param_file, caffe::TEST);
  const string model_file = FLAGS_model;
  net.CopyTrainedLayersFrom(model_file);

  /* blob names */
  vector<string> blob_names;
  boost::split(blob_names, FLAGS_blob, boost::is_any_of(","));
  for (int ix = 0; ix < blob_names.size(); ++ix)
    CHECK(net.has_blob(blob_names[ix])) << "blob name not found: " << blob_names[ix];

  /* output files */
  vector<string> outfile_names;
  if (FLAGS_outfile == "") {
    for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id)
      outfile_names.push_back(blob_names[blob_id] + ".bdt");
  } else {
    boost::split(outfile_names, FLAGS_outfile, boost::is_any_of(","));
    CHECK(blob_names.size() == outfile_names.size());
  }

  const unsigned long int num_instance = FLAGS_num;
  const std::string outfile_prefix = FLAGS_prefix;
  std::ofstream * outfiles = new std::ofstream[outfile_names.size()];
  for (int file_id = 0; file_id < outfile_names.size(); ++file_id) {
    const unsigned int blob_feature_size = net.blob_by_name(blob_names[file_id])->offset(1) * sizeof(float);
    const string outfile_name = outfile_prefix + outfile_names[file_id];
    LOG(INFO) << "extracting blob [" << blob_names[file_id] << "] into " << outfile_name;
    LOG(INFO) << "  this blob has a feature size of " << net.blob_by_name(blob_names[file_id])->offset(1) << " dimesion";
    LOG(INFO) << "  estimated output file size: "
      << num_instance * blob_feature_size << " B" << " or "
      << num_instance * blob_feature_size / 1024.0 / 1024.0 << " M";
    outfiles[file_id].open(outfile_name.c_str(), std::ios::binary);
  }

  /* forward net and extract blob data */
  const unsigned long int display_interval = FLAGS_disp;
  unsigned long int instance_count = 0;
  unsigned long int batch_size = net.blob_by_name(blob_names[0])->num();
  vector<Blob<float>*> input_vec;
  while (instance_count < num_instance) {
    net.Forward(input_vec);
    if (num_instance - instance_count < batch_size)
      batch_size = num_instance - instance_count;
    for (long unsigned int item_id = 0; item_id < batch_size; ++item_id) {
      for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
        const shared_ptr<Blob<float> > feature_blob = net.blob_by_name(blob_names[blob_id]);
        float * feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(item_id);
        const unsigned int blob_feature_size = feature_blob->offset(1) * sizeof(float);
        outfiles[blob_id].write(reinterpret_cast<char *>(feature_blob_data), blob_feature_size);
      }
    }
    instance_count += batch_size;
    if (display_interval != 0 &&
        (instance_count % display_interval == 0 ||
         instance_count == num_instance))
    LOG(INFO) << "num " << instance_count << " / " << num_instance << " processed";
  }

  for (int file_id = 0; file_id < outfile_names.size(); ++file_id) {
    outfiles[file_id].close();
    const string outfile_name = outfile_prefix + outfile_names[file_id];
    LOG(INFO) << "blob [" << blob_names[file_id] << "] extracted into " << outfile_name;
  }
  delete[] outfiles;


  return 0;
}
