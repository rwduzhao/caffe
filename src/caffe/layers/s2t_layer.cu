#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/s2t_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"

namespace caffe {

template <typename Dtype>
void S2TLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  const Dtype* bot_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_gpu_set(top[0]->count(), static_cast<Dtype>(0.), top_data);

  int bot_offset = 0;
  int top_offset = 0;
  const int bot_inc = bot_h_ * bot_w_;
  const int top_inc = 1;
  const S2TParameter_Order order = this->layer_param_.s2t_param().order();
  for (int nid = 0; nid < bot_n_; ++nid) {
    for (int hid = 0; hid < bot_h_; ++hid) {
      for (int wid = 0; wid < bot_w_; ++wid) {
        GetBotTopOffset(bottom[0], top[0],
                        nid, hid, wid, order,
                        bot_offset, top_offset);
        caffe_gpu_axpy(bot_c_, static_cast<Dtype>(1.),
                       bot_data + bot_offset, bot_inc,
                       top_data + top_offset, top_inc);
      }
    }
  }
}

template <typename Dtype>
void S2TLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  Dtype* bot_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bot_diff);

  int bot_offset = 0;
  int top_offset = 0;
  const int bot_inc = bot_h_ * bot_w_;
  const int top_inc = 1;
  const S2TParameter_Order order = this->layer_param_.s2t_param().order();
  for (int nid = 0; nid < bot_n_; ++nid) {
    for (int hid = 0; hid < bot_h_; ++hid) {
      for (int wid = 0; wid < bot_w_; ++wid) {
        GetBotTopOffset(bottom[0], top[0],
                        nid, hid, wid, order,
                        bot_offset, top_offset);
        caffe_gpu_axpy(bot_c_, static_cast<Dtype>(1.),
                       top_diff + top_offset, top_inc,
                       bot_diff + bot_offset, bot_inc);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(S2TLayer);

}  // namespace caffe
