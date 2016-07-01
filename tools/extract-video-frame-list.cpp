#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <gflags/gflags.h>
#include <ctime>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"

DEFINE_string(devices, "0", "device ids");
DEFINE_string(net, "", "net prototxt file");
DEFINE_string(model, "", "trained model file");
DEFINE_int32(batch_size, 0, "");
DEFINE_string(mean_blob, "", "image mean blob");
DEFINE_string(channel_means, "0,0,0", "image channel means");
DEFINE_int32(image_height, 256, "image height");
DEFINE_int32(image_width, 256, "image width");
DEFINE_int32(flip, 0, "augment flip");
DEFINE_int32(corners, 0, "augment corners");
DEFINE_string(blobs, "", "blob names to be extracted");
DEFINE_string(frame_prefix, "", "frame prefix");
DEFINE_string(frame_list, "", "frame list");
DEFINE_int32(write_frame_features, 1, "write frame features");
DEFINE_int32(write_average_pool, 1, "write average pool");
DEFINE_int32(write_maximum_pool, 1, "write maximum pool");
DEFINE_string(outfile_prefix, "", "output file prefix");

using std::ifstream;
using std::ofstream;
using std::string;
using std::map;
using boost::thread;
using boost::thread_group;
using namespace caffe;

const string GetDirName(const string file_name) {
  const int pos_e = file_name.rfind("/");
  return file_name.substr(0, pos_e);
}

void WriteBlobToText(const string filename, const Blob<float>* blob) {
  const float* data = blob->cpu_data();
  const int line_word_count = blob->offset(0, 1, 0, 0);
  FILE *fp = fopen(filename.c_str(), "w");
  for (int n = 0; n < blob->num(); ++n) {
    for (int c = 0; c < blob->channels(); ++c) {
      for (int h = 0; h < blob->height(); ++h) {
        for (int w = 0; w < blob->width(); ++w) {
          const int index = blob->offset(n, c, h, w);
          const float value = data[index];
          const char sep = (index + 1) % line_word_count == 0 ? '\n' : ' ';
          fprintf(fp, "%f%c", value, sep);
        }
      }
    }
  }
  fclose(fp);
}

void WriteBlobToBin(const string filename, const Blob<float>* blob) {
  ofstream output;
  output.open(filename.c_str(), std::ios::binary | std::ios::out);
  output.write((char*)blob->cpu_data(), blob->count() * sizeof(float));
  output.close();
}

vector<int> GetDeviceIds(const string &devices) {
  vector<string> device_id_names;
  boost::split(device_id_names, devices, boost::is_any_of(","));
  vector<int> device_ids;
  for(int id = 0; id < device_id_names.size(); ++id) {
    const int device_id = atoi(device_id_names[id].c_str());
    device_ids.push_back(device_id);
  }
  return device_ids;
}

const string GetVideoName(const string file_name, const string split = "-") {
  const int pos_b = file_name.rfind("/");
  const int pos_e = file_name.rfind(split);
  return file_name.substr(pos_b + 1, pos_e - pos_b - 1);
}

void ReadVideoFramesTable(const string &frame_list_filename,
                          vector<string> &video_list,
                          map<string, vector<string> > &video_frames_table) {
  ifstream infile(frame_list_filename);
  string frame_filename;
  while (infile >> frame_filename) {
    const string video_name = GetVideoName(frame_filename);
    map<string, vector<string> >::iterator iter = video_frames_table.find(video_name);
    if (iter != video_frames_table.end()) {
      iter->second.push_back(frame_filename);
    } else {
      vector<string> frame_list;
      frame_list.push_back(frame_filename);
      video_frames_table[video_name] = frame_list;
      video_list.push_back(video_name);
    }
  }
  infile.close();
}

void CreateMeanBlobFromChannels(const vector<float> channel_means,
                                const int image_height, const int image_width,
                                Blob<float> *mean_blob) {
  const int image_channels = channel_means.size();
  mean_blob->Reshape(1, image_channels, image_height, image_width);
  float* mean_blob_data = mean_blob->mutable_cpu_data();
  for (int c = 0; c < image_channels; ++c) {
    const int start = mean_blob->offset(0, c, 0, 0);
    const int end = mean_blob->offset(0, c + 1, 0, 0);
    const float channel_mean = channel_means[c];
    for (int id = start; id < end; ++id) {
      mean_blob_data[id] = channel_mean;
    }
  }
}

void ReadMeanFromProtoFile(const string mean_blob_filename,
                           Blob<float> *mean_blob) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(FLAGS_mean_blob.c_str(), &blob_proto);
  mean_blob->FromProto(blob_proto);
}

void SetImageOffsets(const int image_height, const int image_width,
                     const int image_crop_height, const int image_crop_width,
                     vector<int> &image_height_offsets, vector<int> &image_width_offsets) {
  // center crop offsets
  image_height_offsets.push_back((image_height - image_crop_height) / 2);
  image_width_offsets.push_back((image_width - image_crop_width) / 2);
  // upper left crop offsets
  image_height_offsets.push_back(0);
  image_width_offsets.push_back(0);
  // upper right crop offsets
  image_height_offsets.push_back(0);
  image_width_offsets.push_back(image_width - image_crop_width);
  // lower left crop offsets
  image_height_offsets.push_back(image_height - image_crop_height);
  image_width_offsets.push_back(0);
  // lower right crop offsets
  image_height_offsets.push_back(image_height - image_crop_height);
  image_width_offsets.push_back(image_width - image_crop_width);
}

cv::Mat ReadResizedImage(const string image_filename,
                         const int image_height, const int image_width) {
  cv::Mat image = ReadImageToCVMat(image_filename, image_height, image_width);
  if (!image.data) {
    LOG(ERROR) << "failed to read image: " << image_filename;
    image.create(image_width, image_height, CV_8UC3);
    image.setTo(cv::Scalar(0.0, 0.0, 0.0));
  }
  return image;
}

void FillInImageDataBlob(const cv::Mat &image, const Blob<float> *mean_blob,
                         const int height_offset, const int width_offset,
                         const int batch_id, Blob<float> *image_blob) {
  const float *mean_blob_data = mean_blob->cpu_data();
  float *image_blob_data = image_blob->mutable_cpu_data();
  for (int c = 0; c < image_blob->channels(); ++c) {
    for (int h = 0; h < image_blob->height(); ++h) {
      for (int w = 0; w < image_blob->width(); ++w) {
        const float pixel_value = static_cast<float>(static_cast<uint8_t>(image.at<cv::Vec3b>(h + height_offset, w + width_offset)[c]));
        const int mean_id = mean_blob->offset(0, c, h + height_offset, w + width_offset);
        const float mean_value = mean_blob_data[mean_id];
        const int data_id = image_blob->offset(batch_id, c, h, w);
        image_blob_data[data_id] = pixel_value - mean_value;
      }
    }
  }
}

void AverageBlobByRowCPU(const Blob<float>* blob, const int blob_offset,
                         const int num_row, const int num_col,
                         Blob<float>* feature_blob, const int feature_blob_offset) {
  Blob<float> multiplier_blob;
  multiplier_blob.Reshape(1, 1, num_row, 1);
  float* multiplier_blob_data = multiplier_blob.mutable_cpu_data();
  caffe_set(multiplier_blob.count(), (float)1. / (float)num_row, multiplier_blob_data);

  const float* blob_data = blob->cpu_data() + blob_offset;
  float* feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob_offset;
  caffe_cpu_gemv(CblasTrans, num_row, num_col, (float)1.0, blob_data,
                 multiplier_blob.cpu_data(), (float)0.0, feature_blob_data);
}

void AverageBlobByRowGPU(const Blob<float>* blob, const int blob_offset,
                         const int num_row, const int num_col,
                         Blob<float>* feature_blob, const int feature_blob_offset) {
  Blob<float> multiplier_blob;
  multiplier_blob.Reshape(1, 1, num_row, 1);
  float* multiplier_blob_data = multiplier_blob.mutable_gpu_data();
  caffe_gpu_set(multiplier_blob.count(), (float)1. / (float)num_row, multiplier_blob_data);

  const float* blob_data = blob->gpu_data() + blob_offset;
  float* feature_blob_data = feature_blob->mutable_gpu_data() + feature_blob_offset;
  caffe_gpu_gemv(CblasTrans, num_row, num_col, (float)1.0, blob_data,
                 multiplier_blob.gpu_data(), (float)0.0, feature_blob_data);
}

void PrintRemainTime(const int num_total, const int num_pass, const double used_time) {
  const double mean_time = used_time / (double)num_pass;
  const int num_remain = num_total - num_pass;
  const double remain_time = (double)num_remain * mean_time;
  const double remain_second = fmod(remain_time, 60.);
  const double remain_minute = floor(fmod(remain_time, 60. * 60.) / 60.0);
  const double remain_hour = floor(fmod(remain_time, 60. * 60. * 24.) / (60. * 60.));
  const double remain_day = floor(remain_time / 3600.0 / 24.);
  LOG(INFO) << "Estimated remaining time: "
    << remain_day << " days "
    << remain_hour << " hours "
    << remain_minute << " minutes "
    << remain_second << " seconds.";
}

void ExtractFeature(vector<string> &video_list,map<string,
                    vector<string> > &video_frames_table,
                    const string &frame_prefix,
                    const string &net_proto,
                    const string &net_model,
                    const int device_id,
                    const int batch_size,
                    const int image_height,
                    const int image_width,
                    const Blob<float>* mean_blob,
                    const bool &flip,
                    const int corners,
                    const vector<string> &blob_names,
                    const string &outfile_prefix,
                    const bool write_frame_features,
                    const bool write_average_pool,
                    const bool write_maximum_pool) {
  // setup caffe
  Caffe::SetDevice(device_id);
  Caffe::set_mode(Caffe::GPU);
  shared_ptr<Net<float> > net(new Net<float>(net_proto, caffe::TEST));
  net->CopyTrainedLayersFrom(net_model);
  LOG(INFO) << "Network parameters loaded.";

  // set batch size
  Blob<float>* image_blob = net->input_blobs()[0];
  const int num_crop = ((int)flip + 1) * (1 + corners);
  if (batch_size == 0) {
    CHECK(image_blob->num() % num_crop == 0) << "bad input data dimension";
  } else {
    image_blob->Reshape(batch_size * num_crop,
                        image_blob->channels(),
                        image_blob->height(),
                        image_blob->width());
    net->Reshape();
  }
  LOG(INFO) << "Batch size set to " << batch_size;

  // set crop offsets
  vector<int> image_height_offsets;
  vector<int> image_width_offsets;
  SetImageOffsets(image_height, image_width,
                  image_blob->height(), image_blob->width(),
                  image_height_offsets, image_width_offsets);
  LOG(INFO) << "Image crop offsets set to:";
  for (int index = 0; index < image_height_offsets.size(); ++index)
    LOG(INFO) << index
      << " (" << image_height_offsets[index]
      << ", " << image_width_offsets[index] << ")";

  int num_frame = 0;
  int max_single_frame = 0;
  const int num_video = video_list.size();
  for (int video_index = 0; video_index < num_video; ++video_index) {
    const string video_name = video_list[video_index];
    const int num_single_frame = video_frames_table[video_name].size();
    num_frame += num_single_frame;
    if (num_single_frame > max_single_frame)
      max_single_frame = num_single_frame;
  }
  LOG(INFO) << "Number of frames: " << num_frame;

  // init feature blobs
  vector<shared_ptr<Blob<float> > > feature_blobs;
  vector<shared_ptr<Blob<float> > > summary_feature_blobs;
  for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
    const string blob_name = blob_names[blob_id];
    const shared_ptr<Blob<float> > blob = net->blob_by_name(blob_name);
    const int channels = blob->channels();
    const int height = blob->height();
    const int width = blob->width();
    shared_ptr<Blob<float> > feature_blob(new Blob<float>(max_single_frame, channels, height, width));
    feature_blobs.push_back(feature_blob);
    shared_ptr<Blob<float> > summary_feature_blob(new Blob<float>(1, channels, height, width));
    summary_feature_blobs.push_back(summary_feature_blob);
  }
  LOG(INFO) << "Feature blob sizes initialized.";

  float disk_size = 0.;
  float max_single_size = 0.;
  for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
    const float blob_size = static_cast<float>(summary_feature_blobs[blob_id]->count());
    if (write_frame_features) {
      disk_size += (blob_size * static_cast<float>(num_frame));
      max_single_size += (blob_size * static_cast<float>(max_single_frame));
    }
    if (write_average_pool) {
      disk_size += (blob_size * static_cast<float>(num_video));
      max_single_size += (blob_size * 1);
    }
    if (write_maximum_pool) {
      disk_size += (blob_size * static_cast<float>(num_video));
      max_single_size += (blob_size * 1);
    }
  }
  const float disk_size_kb = disk_size * 4. / 1024.;
  const float disk_size_mb = disk_size_kb / 1024.;
  const float disk_size_gb = disk_size_mb / 1024.;
  LOG(INFO) << "Disk space required: "
    << static_cast<int>(disk_size_gb) << "GB "
    << static_cast<int>(disk_size_mb) << "MB "
    << static_cast<int>(disk_size_kb) << "KB";
  const float data_memory_size_kb = max_single_size * 4. / 1024.;
  const float data_memory_size_mb = data_memory_size_kb / 1024.;
  const float data_memory_size_gb = data_memory_size_mb / 1024.;
  LOG(INFO) << "Data memory space required: "
    << static_cast<int>(data_memory_size_gb) << "GB "
    << static_cast<int>(data_memory_size_mb) << "MB "
    << static_cast<int>(data_memory_size_kb) << "KB";

  std::clock_t start;
  start = std::clock();

  int num_possessed_frame = 0;
  for (int video_index = 0; video_index < num_video; ++video_index) {
    const string video_name = video_list[video_index];
    LOG(INFO) << "processing video [" << video_index + 1 << "/" << num_video << "]: " << video_name;

    vector<string> frame_list = video_frames_table[video_name];

    // alloc features space
    for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
      const int channels = feature_blobs[blob_id]->channels();
      const int height = feature_blobs[blob_id]->height();
      const int width = feature_blobs[blob_id]->width();
      feature_blobs[blob_id]->Reshape(frame_list.size(), channels, height, width);
    }

    int num_crop_added = 0;
    for (int frame_id = 0; frame_id < frame_list.size(); ++frame_id) {
      cv::Mat image = ReadResizedImage(frame_prefix + frame_list[frame_id], image_height, image_width);
      if (true) {  // original non-flip
        for (int crop_id = 0; crop_id < 1 + corners; ++crop_id) {
          const int height_offset = image_height_offsets[crop_id];
          const int width_offset = image_width_offsets[crop_id];
          FillInImageDataBlob(image, mean_blob, height_offset, width_offset, num_crop_added, image_blob);
          ++num_crop_added;
        }
      }
      if (flip) {  // flip
        const int CV_FLIP_LEFT_RIGHT = 1;
        cv::Mat flipped_image;
        cv::flip(image, flipped_image, CV_FLIP_LEFT_RIGHT);
        for (int crop_id = 0; crop_id < 1 + corners; ++crop_id) {
          const int height_offset = image_height_offsets[crop_id];
          const int width_offset = image_width_offsets[crop_id];
          FillInImageDataBlob(flipped_image, mean_blob, height_offset, width_offset, num_crop_added, image_blob);
          ++num_crop_added;
        }
      }
      CHECK(num_crop_added % num_crop == 0);

      if (num_crop_added == image_blob->num() || frame_id + 1 == frame_list.size()) {
        net->ForwardPrefilled();

        for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
          const shared_ptr<Blob<float> > net_blob = net->blob_by_name(blob_names[blob_id]);
          const int num_row = num_crop;
          const int num_col = net_blob->count() / net_blob->num();
          shared_ptr<Blob<float> > feature_blob = feature_blobs[blob_id];

          const int num_batch_frame = num_crop_added / num_crop;
          for (int batch_frame_id = 0; batch_frame_id < num_batch_frame; ++batch_frame_id) {
            const int blob_offset = net_blob->offset(batch_frame_id * num_crop, 0, 0, 0);
            const int feature_blob_offset = feature_blob->offset(frame_id + 1 - num_batch_frame + batch_frame_id , 0, 0, 0);
            AverageBlobByRowGPU(net_blob.get(), blob_offset, num_row, num_col, feature_blob.get(), feature_blob_offset);
          }
        }
        num_crop_added = 0;
      }
    }
    num_possessed_frame += frame_list.size();

    for (int blob_id = 0; blob_id < blob_names.size(); ++blob_id) {
      const string blob_name = blob_names[blob_id];

      if (write_frame_features) {
        const string outfile = outfile_prefix + video_name + "." + blob_name + ".bdt";
        CHECK(system(("mkdir -p " + GetDirName(outfile)).c_str()) != -1);
        WriteBlobToBin(outfile, feature_blobs[blob_id].get());
      }

      if (write_average_pool) {
        AverageBlobByRowGPU(feature_blobs[blob_id].get(), 0,
                            feature_blobs[blob_id]->num(),
                            feature_blobs[blob_id]->count() / feature_blobs[blob_id]->num(),
                            summary_feature_blobs[blob_id].get(), 0);
        const string outfile = outfile_prefix + video_name + "." + blob_name + ".avg.bdt";
        CHECK(system(("mkdir -p " + GetDirName(outfile)).c_str()) != -1);
        WriteBlobToBin(outfile, summary_feature_blobs[blob_id].get());
      }

      if (write_maximum_pool) {
        const string outfile = outfile_prefix + video_name + "." + blob_name + ".max.bdt";
        CHECK(system(("mkdir -p " + GetDirName(outfile)).c_str()) != -1);
      }
    }

    if ((video_index + 1) % 50 == 0)
      PrintRemainTime(num_frame, num_possessed_frame,
                      (std::clock() - start) / (double)CLOCKS_PER_SEC);
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // get video and frame list
  vector<string> video_list;
  map<string, vector<string> > video_frames_table;
  LOG(INFO) << "reading frame list: " << FLAGS_frame_list;
  ReadVideoFramesTable(FLAGS_frame_list, video_list, video_frames_table);
  if (video_list.size() == 0) {
    LOG(INFO) << "number of videos: " << video_list.size();
    return 0;
  }

  // get mean data
  Blob<float> mean_blob;
  if (FLAGS_mean_blob.empty()) {
    vector<string> channel_mean_strings;
    boost::split(channel_mean_strings, FLAGS_channel_means, boost::is_any_of(","));
    vector<float> channel_means;
    for (int c = 0; c < channel_mean_strings.size(); ++c)
      channel_means.push_back(atof(channel_mean_strings[c].c_str()));
    CreateMeanBlobFromChannels(channel_means, FLAGS_image_height, FLAGS_image_width, &mean_blob);
  } else {
    ReadMeanFromProtoFile(FLAGS_mean_blob, &mean_blob);
  }

  // get blob names
  vector<string> blob_names;
  boost::split(blob_names, FLAGS_blobs, boost::is_any_of(","));

  const vector<int> device_ids = GetDeviceIds(FLAGS_devices);
  const int device_id = device_ids[0];
  ExtractFeature(video_list,
                 video_frames_table,
                 FLAGS_frame_prefix,
                 FLAGS_net,
                 FLAGS_model,
                 device_id,
                 FLAGS_batch_size,
                 mean_blob.height(),
                 mean_blob.width(),
                 &mean_blob,
                 FLAGS_flip == 1,
                 FLAGS_corners,
                 blob_names,
                 FLAGS_outfile_prefix,
                 FLAGS_write_frame_features == 1,
                 FLAGS_write_average_pool == 1,
                 FLAGS_write_maximum_pool == 1);
//  thread_group t_group;
//  for (int device_id = 0; device_id < device_ids.size(); ++device_id) {
//    const string frame_prefix = string(FLAGS_frame_prefix);
//    const string net = string(FLAGS_net);
//    const string model = string(FLAGS_model);
//    int batch_size = FLAGS_batch_size;
//    const bool flip = FLAGS_flip == 1;
//    const int corners = FLAGS_corners;
//    const string outfile_prefix = string(FLAGS_outfile_prefix);
//    t_group.add_thread(new thread(ExtractFeature,
//                                  video_list,
//                                  video_frames_table,
//                                  frame_prefix,
//                                  net,
//                                  model,
//                                  device_id,
//                                  batch_size,
//                                  mean_blob.height(),
//                                  mean_blob.width(),
//                                  &mean_blob,
//                                  flip,
//                                  corners,
//                                  blob_names,
//                                  outfile_prefix,
//                                  FLAGS_write_frame_features == 1,
//                                  FLAGS_write_average_pool == 1,
//                                  FLAGS_write_maximum_pool == 1));
//  }
//  t_group.join_all();

  return 0;
}
