#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <boost/random.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/io_extra.hpp"

namespace caffe {


cv::Mat ZteCropBackground(const string& filename,
                          const int height, const int width,
                          const bool is_color,
                          const int crop_x0, const int crop_y0,
                          const int crop_width, const int crop_height) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }

  cv_img_origin = cv_img_origin(cv::Rect(std::max(0, crop_x0), std::max(0, crop_y0),
                                         std::min(crop_width, cv_img_origin.cols - crop_x0),
                                         std::min(crop_height, cv_img_origin.rows - crop_y0)));

  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  return cv_img;
}

bool ZteCropBackground(const string image_file, cv::Mat &image,
                       int crop_width, int crop_height,
                       int &x0, int &y0, int &diff_x, int &diff_y,
                       double &scale) {
  // read image
  try {
    image = cv::imread(image_file.c_str());
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    LOG(ERROR) << "exception caught: " << err_msg << std::endl;
    image.create(1280, 960, CV_8UC3);
    image.setTo(cv::Scalar(104.0, 117.0, 124.0));
    return false;
  }

  // random function

  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));

  boost::uniform_int<> width_dist(0, image.cols - 1);
  boost::uniform_int<> height_dist(0, image.rows - 1);
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > dice_x0(*rng, width_dist);
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > dice_y0(*rng, height_dist);

  boost::uniform_int<> diff_x_dist(-15, 15);
  boost::uniform_int<> diff_y_dist(-15, 15);
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > dice_diff_x(*rng, diff_x_dist);
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > dice_diff_y(*rng, diff_y_dist);

  boost::normal_distribution<> distribution(0.67, 0.1);
  boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);

  if (scale == 0.0)
    scale = std::min(std::max(0.30, dist()), 1.5);
  crop_width = (int)(scale * crop_width);
  crop_height = (int)(scale * crop_height);
  const int crop_size_limit = std::min(image.cols, image.rows) - 1;
  if (crop_width > crop_size_limit)
    crop_width = crop_size_limit;
  if (crop_height > crop_size_limit)
    crop_height = crop_size_limit;

  x0 = x0 >= 0 ? x0 : dice_x0();
  y0 = y0 >= 0 ? y0 : dice_y0();
  diff_x = (diff_x > -image.cols && diff_x < image.cols) ? diff_x : dice_diff_x();
  diff_y = (diff_y > -image.rows && diff_y < image.rows) ? diff_y : dice_diff_y();
  x0 += diff_x;
  y0 += diff_y;
  if (x0 < 0)
    x0 = 0;
  if (y0 < 0)
    y0 = 0;

  x0 = std::min(x0, image.cols - crop_width);
  y0 = std::min(y0, image.rows - crop_height);

  image = image(cv::Rect(x0, y0, crop_width, crop_height));

  delete rng;

  return true;
}

bool ZteCropCenteredPerson(const string image_file, cv::Mat &image) {
  try {
    image = cv::imread(image_file.c_str());
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    LOG(ERROR) << "exception caught: " << err_msg << std::endl;
    image.create(360, 360, CV_8UC3);
    image.setTo(cv::Scalar(104.0, 117.0, 124.0));
    return false;
  }

  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));
  boost::normal_distribution<> distribution(0.67, 0.1);
  boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);
  const double scale = 0.67;

  const int height = image.rows;
  const int width = image.cols;
  const int size_limit = std::min(height, width) - 1;

  const int scaled_height = std::min((int)(scale * height), size_limit);
  const int scaled_width = std::min((int)(scale * width), size_limit);

  const int height_center = height / 2;
  const int width_center = width / 2;
  const int x0 = width_center - scaled_width / 2;
  const int y0 = height_center - scaled_height / 2;

  image = image(cv::Rect(x0, y0, scaled_width, scaled_height));

  delete rng;

  return true;
}

bool ZteCropPerson(const string image_file, cv::Mat &image) {
  try {
    image = cv::imread(image_file.c_str());
  } catch(cv::Exception& e) {
    const char* err_msg = e.what();
    LOG(ERROR) << "exception caught: " << err_msg << std::endl;
    image.create(360, 360, CV_8UC3);
    image.setTo(cv::Scalar(104.0, 117.0, 124.0));
    return false;
  }

  boost::mt19937 *rng = new boost::mt19937();
  rng->seed(time(NULL));
  boost::normal_distribution<> distribution(0.67, 0.1);
  boost::variate_generator< boost::mt19937, boost::normal_distribution<> > dist(*rng, distribution);
  const double scale = std::min(std::max(0.60, dist()), 1.0);

  const int height = image.rows;
  const int width = image.cols;
  const int size_limit = std::min(height, width) - 1;

  const int scaled_height = std::min((int)(scale * height), size_limit);
  const int scaled_width = std::min((int)(scale * width), size_limit);

  const int height_center = height / 2;
  const int width_center = width / 2;
  const int x0 = width_center - scaled_width / 2;
  const int y0 = height_center - scaled_height / 2;

  image = image(cv::Rect(x0, y0, scaled_width, scaled_height));

  delete rng;

  return true;
}

bool ReadImageToPrespecifiedDatum(const string& filename, const int label,
                                  const int height, const int width,
                                  const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (is_color ? 3 : 1);
  CHECK(datum->channels() >= cv_img.channels());
  CHECK(datum->height() == cv_img.rows);
  CHECK(datum->width() == cv_img.cols);
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

bool ReadZteImageToPrespecifiedDatum(const string& filename, const int label,
                                     const int height, const int width,
                                     const bool is_color, Datum* datum,
                                     int &x0, int &y0, int &diff_x, int &diff_y,
                                     double &scale) {
  const bool zte_is_color = true;
  // const int cv_read_flag = (zte_is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin;
  if (label == 0)
    ZteCropBackground(filename, cv_img_origin, 360, 360, x0, y0, diff_x, diff_y, scale);
  if (label < 0)
    ZteCropCenteredPerson(filename, cv_img_origin);
  else
    ZteCropPerson(filename, cv_img_origin);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (zte_is_color ? 3 : 1);
  CHECK(datum->channels() >= cv_img.channels());
  CHECK(datum->height() == cv_img.rows);
  CHECK(datum->width() == cv_img.cols);
  string* datum_string = datum->mutable_data();
  if (zte_is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

bool ReadZteImageToPrespecifiedDatum(const string& filename, const int label,
                                     const int height, const int width,
                                     const bool is_color, Datum* datum,
                                     const vector<int> location, double &scale) {
  // const int cv_read_flag = (zte_is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin;
  if (location.size() == 0) {
    DLOG(INFO) << "foreground: " << filename;
    cv_img_origin = ReadImageToCVMat(filename, height, width, is_color);
  } else {
    DLOG(INFO) << "background: " << filename;
    const int crop_x_center = location[0];
    const int crop_y_center = location[1];
    const int crop_size = location[4];
    const int crop_x0 = std::max(0, crop_x_center - crop_size / 2);
    const int crop_y0 = std::max(0, crop_y_center - crop_size / 2);
    cv_img_origin = ZteCropBackground(filename, height, width, is_color,
                                      crop_x0, crop_y0, crop_size, crop_size);

  }

  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  const bool zte_is_color = true;  //TODO
  int num_channels = (zte_is_color ? 3 : 1);
  CHECK(datum->channels() >= cv_img.channels());
  CHECK(datum->height() == cv_img.rows);
  CHECK(datum->width() == cv_img.cols);
  string* datum_string = datum->mutable_data();
  if (zte_is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }

  return true;
}

}  // namespace caffe
