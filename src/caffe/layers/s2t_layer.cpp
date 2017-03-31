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
void S2TLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void S2TLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  bot_n_ = bottom[0]->num();  // num of examples
  bot_c_ = bottom[0]->channels();  // feature dim
  bot_h_ = bottom[0]->height();  // feature map height
  bot_w_ = bottom[0]->width();  // feature map width

  const S2TParameter param = this->layer_param_.s2t_param();
  switch (param.order()) {
    case S2TParameter_Order_H0:
      top_n_ = bot_h_;
      top_c_ = bot_n_ * bot_w_;
      top_h_ = bot_c_;
      top_w_ = 1;
      break;
    case S2TParameter_Order_W0:
      top_n_ = bot_w_;
      top_c_ = bot_n_ * bot_h_;
      top_h_ = bot_c_;
      top_w_ = 1;
      break;
    default:
      top_n_ = bot_h_ * bot_w_ * bot_n_;
      top_c_ = bot_c_;
      top_h_ = 1;
      top_w_ = 1;
      break;
  }
  // top_n_ : num example * time step = bot_n_ * bot_h_ * bot_w_
  // top_c_ : feature dim = bot_c_
  // top_h_ : 1
  // top_w_ : 1
  top[0]->Reshape(top_n_, top_c_, top_h_, top_w_);

  if (top.size() == 2)
    top[1]->Reshape(top_n_, 1, 1, 1);  // clip data : num example * time step - 1 - 1 - 1
}

template <typename Dtype>
void S2TLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bot_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_set(top[0]->count(), static_cast<Dtype>(0.), top_data);

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
        caffe_cpu_axpy(bot_c_, static_cast<Dtype>(1.),
                       bot_data + bot_offset, bot_inc,
                       top_data + top_offset, top_inc);
      }
    }
  }

  // clip data
  if (top.size() == 2) {
    Dtype* clip_data = top[1]->mutable_cpu_data();
    caffe_set(top[1]->count(), Dtype(1.), clip_data);
    for (int nid = 0; nid < bot_n_; ++nid)
      clip_data[top[1]->offset(nid * bot_h_ * bot_w_)] = Dtype(0.);
  }
}

template <typename Dtype>
void S2TLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  Dtype* bot_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0.), bot_diff);

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
        caffe_cpu_axpy(bot_c_, static_cast<Dtype>(1.),
                       top_diff + top_offset, top_inc,
                       bot_diff + bot_offset, bot_inc);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(S2TLayer);
#endif

INSTANTIATE_CLASS(S2TLayer);
REGISTER_LAYER_CLASS(S2T);

}  // namespace caffe
