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

bool ReadImageToPrespecifiedDatum(const string& filename, const int label, const int height, const int width, const bool is_color, Datum* datum);


}  // namespace caffe

#endif   // CAFFE_UTIL_IO_EXTRA_HPP_
