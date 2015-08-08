#ifndef CAFFE_UTIL_IO_EXTRA_HPP_
#define CAFFE_UTIL_IO_EXTRA_HPP_

#include <unistd.h>
#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using ::google::protobuf::Message;

bool ZteCropBackground(const string image_file, cv::Mat &image,
                       int crop_width, int crop_height,
                       int &x0, int &y0, int &diff_x, int &diff_y,
                       double &scale);
bool ZteCropPerson(const string image_file, cv::Mat &image);
bool ZteCropCenteredPerson(const string image_file, cv::Mat &image);
bool ReadImageToPrespecifiedDatum(const string& filename, const int label,
                                  const int height, const int width,
                                  const bool is_color, Datum* datum);
bool ReadZteImageToPrespecifiedDatum(const string& filename, const int label,
                                     const int height, const int width,
                                     const bool is_color, Datum* datum,
                                     int &x0, int &y0, int &diff_x, int &diff_y,
                                     double &scale);
cv::Mat ZteCropBackground(const string& filename,
                          const int height, const int width,
                          const bool is_color,
                          const int crop_x0, const int crop_y0,
                          const int crop_width, const int crop_height);


}  // namespace caffe

#endif   // CAFFE_UTIL_IO_EXTRA_HPP_
