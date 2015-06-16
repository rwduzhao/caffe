/*=============================================================================
#     FileName: extract_blob_data.cpp
#   Desciption: extract blob data
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-06-16 21:11:36
#      History:
=============================================================================*/

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "caffe/caffe.hpp"

DEFINE_int32(device_id, 0, "device id (for gpu only)");
DEFINE_int32(image_channels, 3, "image channels");
DEFINE_int32(image_height, 256, "image height");
DEFINE_int32(image_width, 256, "image width");
DEFINE_string(blobs, "", "blob names to be extracted");
DEFINE_string(channel_means, "0,0,0", "image channel means");
DEFINE_string(device_type, "gpu", "compute using cpu or gpu");
DEFINE_string(image_list, "", "image list file");
DEFINE_string(image_prefix, "", "image prefix name");
DEFINE_string(mean_blob, "", "image mean blob");
DEFINE_string(model, "", "trained model file");
DEFINE_string(net, "", "net prototxt file");
DEFINE_string(outfile_format, "txt", "output format: txt or bdt");
DEFINE_string(outfile_prefix, "", "output file prefix");
DEFINE_string(source_type, "image_list", "data source type");

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

bool ReadMeanBlobSize(const string mean_blob_file, int &num, int &channels, int &height, int &width) {
  if (mean_blob_file.empty())
    return false;

  BlobProto mean_blobproto;
  ReadProtoFromBinaryFileOrDie(mean_blob_file, &mean_blobproto);

  Blob<float> mean_blob;
  mean_blob.FromProto(mean_blobproto);

  num = mean_blob.num();
  channels = mean_blob.channels();
  height = mean_blob.height();
  width = mean_blob.width();

  return true;
}

bool ReadMeanBlobData(const string mean_blob_file, float *mean_data, const size_t data_size) {
  if (mean_blob_file.empty())
    return false;

  BlobProto mean_blobproto;
  ReadProtoFromBinaryFileOrDie(mean_blob_file, &mean_blobproto);

  Blob<float> mean_blob;
  mean_blob.FromProto(mean_blobproto);

  const float *src_mean_data = mean_blob.cpu_data();
  memcpy(mean_data, src_mean_data, data_size);

  return true;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // model parameters
  const string net_file = FLAGS_net;
  LOG(INFO) << "reading net structure from: " << net_file;
  Net<float> net(net_file.c_str(), caffe::TEST);
  const string model_file = FLAGS_model;
  LOG(INFO) << "reading net parameter from: " << model_file;
  net.CopyTrainedLayersFrom(model_file.c_str());

  // device type and id
  const string device_type = FLAGS_device_type;
  const int device_id = FLAGS_device_id;
  LOG(INFO) << "using " << device_type << ": " << device_id;
  if (strcmp(device_type.c_str(), "gpu") == 0) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(device_id);
  } else if (strcmp(device_type.c_str(), "cpu") == 0) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    LOG(FATAL) << "unsupported device type";
  }

  // mean file and size
  const string mean_blob_file = FLAGS_mean_blob;
  int image_num = -1;
  int image_channels = -1;
  int image_height = -1;
  int image_width = -1;
  if (!ReadMeanBlobSize(mean_blob_file, image_num, image_channels, image_height, image_width)) {
    image_channels = FLAGS_image_channels;
    image_height = FLAGS_image_height;
    image_width = FLAGS_image_width;
  }
  LOG(INFO) << "image size: " << image_channels << ", " << image_height << ", " << image_width;
  float *mean_data = NULL;
  const int image_blob_size = image_num * image_channels * image_height * image_width;
  if (image_blob_size > 0) {
    const size_t blob_data_size = image_blob_size * sizeof(float);
    mean_data = (float *)malloc(blob_data_size);
    ReadMeanBlobData(mean_blob_file, mean_data, blob_data_size);
  } else {
    const size_t blob_data_size = image_channels * image_height * image_width * sizeof(float);
    mean_data = (float *)malloc(blob_data_size);
    vector<string> channel_means;
    boost::split(channel_means, FLAGS_channel_means, boost::is_any_of(","));
    CHECK_EQ(image_channels, channel_means.size());
    for (int c = 0; c < image_channels; ++c) {
      const int start = c * image_height * image_width;
      const int end = start + image_height * image_width;
      const float channel_mean = atof(channel_means[c].c_str());
      for (int id = start; id < end; ++id) {
        mean_data[id] = channel_mean;
      }
    }
  }

  // image crop size
  const int image_crop_width = net.blob_by_name("data")->height();
  const int image_crop_height = net.blob_by_name("data")->width();
  LOG(INFO) << "image crop size: " << image_channels << ", " << image_crop_height << ", " << image_crop_width;
  const int image_data_size = image_channels * image_crop_width * image_crop_height;
  const int image_height_offset = (image_height - image_crop_height) / 2;
  const int image_width_offset = (image_width - image_crop_width) / 2;

  // prepare blobs
  const string outfile_prefix = FLAGS_outfile_prefix;
  std::vector<std::string> blob_names;
  boost::split(blob_names, FLAGS_blobs, boost::is_any_of(","));
  const int num_blob = blob_names.size();
  const string outfile_format = FLAGS_outfile_format;
  string outfile_extension;
  bool is_binary_output = false;
  if (outfile_format.compare("txt") == 0) {
    outfile_extension = "txt";
  } else if  (outfile_format.compare("bdt") == 0) {
    outfile_extension = "bdt";
    is_binary_output = true;
  } else {
    LOG(FATAL) << "unsupported output file format";
  }
  FILE *outfiles[num_blob];
  for (int blob_id = 0; blob_id < num_blob; blob_id++) {
    const string blob_name = blob_names[blob_id];
    const string outfile_name = outfile_prefix + "." + blob_name + "." + outfile_extension;
    LOG(INFO) << "output blob data in " << blob_name << " into " << outfile_name;
    outfiles[blob_id] = fopen(outfile_name.c_str(), "w");
  }

  const string source_type = FLAGS_source_type;
  if (strcmp(source_type.c_str(), "image_list") == 0) {
    // read image list
    const string image_list_file = FLAGS_image_list;
    LOG(INFO) << "reading from image list file: " << image_list_file;
    std::ifstream infile(image_list_file.c_str());
    std::vector<string> image_names;
    string image_name;
    while (getline(infile, image_name)) {
      image_names.push_back(image_name);
    }
    LOG(INFO) << "a total number of " << image_names.size() << " images in the image list file";
    const string image_prefix = FLAGS_image_prefix;

    const int batch_size = net.blob_by_name("data")->num();
    std::vector<string> batch_image_names;
    std::vector<Blob<float>*> batch_image_blobs;
    Blob<float> *image_blob = new Blob<float>(batch_size, image_channels, image_crop_height, image_crop_width);

    for (int image_id = 0; image_id < image_names.size(); ++image_id) {
      const string image_name = image_names[image_id];
      batch_image_names.push_back(image_name);

      // read image
      bool is_good_image_data = true;
      const string image_filename = image_prefix + image_name;
      cv::Mat image = cv::imread(image_filename, CV_LOAD_IMAGE_COLOR);
      if (!image.data) {
        is_good_image_data = false;
        LOG(ERROR) << "failed to open image: " << image_filename;
      } else {
        try {
          cv::resize(image, image, cv::Size(image_width, image_height), 0, 0, 3);
        } catch (cv::Exception& e) {
          is_good_image_data = false;
          LOG(ERROR) << "failed to resize image: " << image_filename;
        }
      }
      if (!is_good_image_data) {
        LOG(INFO) << "creating blank image data for image: " << image_filename;
        image.create(image_width, image_height, CV_8UC3);
        image.setTo(cv::Scalar(0.0, 0.0, 0.0));
      }

      // preprocess image
      float *image_blob_data = image_blob->mutable_cpu_data();
      for (int c = 0; c < image_channels; ++c) {
        for (int h = 0; h < image_crop_height; ++h) {
          for (int w = 0; w < image_crop_width; ++w) {
            const int data_id = (image_id % batch_size) * image_data_size + (c * image_crop_height + h) * image_crop_width + w;
            const float pixel_value = static_cast<float>(image.at<cv::Vec3b>(h + image_height_offset, w + image_width_offset)[c]);

            const int mean_id = (c * image_height + h + image_height_offset) * image_width + w + image_width_offset;
            const float mean_value = mean_data[mean_id];

            image_blob_data[data_id] = pixel_value - mean_value;
          }
        }
      }

      // go through the network
      if ((image_id + 1) % batch_size == 0 || (image_id + 1) == image_names.size()) {
        batch_image_blobs.push_back(image_blob);
        net.Forward(batch_image_blobs);

        const int batch_image_size = batch_image_names.size();
        for (int blob_id = 0; blob_id < num_blob; ++blob_id) {
          const string blob_name = blob_names[blob_id];
          const shared_ptr<Blob<float> > blob = net.blob_by_name(blob_name);
          const float *blob_data = blob->cpu_data();

          if (is_binary_output) {
            fwrite(blob_data, batch_image_size * blob->channels(), sizeof(float), outfiles[blob_id]);
          } else {
            int ix = 0;
            for (int batch_image_id = 0; batch_image_id < batch_image_size; batch_image_id++) {
              for (int c = 0; c < blob->channels() - 1; ++c)
                fprintf(outfiles[blob_id], "%.9g ", blob_data[ix++]);
              fprintf(outfiles[blob_id], "%.9g\n", blob_data[ix++]);
            }
          }
        }

        LOG(INFO) << image_id + 1 << " (+" << batch_image_size << ") out of " << image_names.size() << " images processed";

        batch_image_names.clear();
        batch_image_blobs.clear();
      }
    }

    delete image_blob;
  }

  for (int blob_id = 0; blob_id < blob_names.size(); blob_id++)
    fclose(outfiles[blob_id]);

  if (mean_data != NULL)
    free(mean_data);

  return 0;
}
