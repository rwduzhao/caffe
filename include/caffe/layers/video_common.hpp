
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth�� Zhaofan Qiu
** mail�� zhaofanqiu@gmail.com
** date�� 2015/12/13
** desc�� Caffe-video common head
*********************************************************************************/

#ifndef CAFFE_VIDEO_COMMON_HPP_
#define CAFFE_VIDEO_COMMON_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/blob.hpp"

using namespace std;
namespace caffe {

	vector<int> video_shape(int num, int channels = 0, int length = 0, int height = 0, int width = 0);

}  // namespace caffe

#endif  // CAFFE_VIDEO_COMMON_HPP_

