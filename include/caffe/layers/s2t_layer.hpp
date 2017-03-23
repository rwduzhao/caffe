#ifndef CAFFE_S2T_LAYER_HPP_
#define CAFFE_S2T_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Transform spatial feature maps to temporal feature maps.
// Typically used between Conv layers and RNN/LSTM layers.
template <typename Dtype>
class S2TLayer : public Layer<Dtype> {
 public:
  explicit S2TLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "S2T"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int bot_n_;
  int bot_c_;
  int bot_h_;
  int bot_w_;

  int top_n_;
  int top_c_;
  int top_h_;
  int top_w_;

  void GetBotTopOffset(const Blob<Dtype>* bottom, const Blob<Dtype>* top,
                       const int nid, const int hid, const int wid,
                       const S2TParameter_Order order,
                       int& bot_offset, int& top_offset) {
    const int bot_h = bottom->height();
    const int bot_w = bottom->width();
    switch (order) { case S2TParameter_Order_H0:
      // H-NW-C-1
      bot_offset = bottom->offset(nid, 0, hid, wid);
      top_offset = top->offset(hid, nid * bot_w + wid, 0, 0);
      break;
      case S2TParameter_Order_W0:
      // W-NH-C-1
      bot_offset = bottom->offset(nid, 0, hid, wid);
      top_offset = top->offset(wid, nid * bot_h + hid, 0, 0);
      break;
      case S2TParameter_Order_H0W0:
      // HW-N-C-1
      bot_offset = bottom->offset(nid, 0, hid, wid);
      top_offset = top->offset(wid * bot_h + hid, nid, 0, 0);
      break;
      case S2TParameter_Order_W0H0:
      // WH-N-C-1
      bot_offset = bottom->offset(nid, 0, hid, wid);
      top_offset = top->offset(hid * bot_w + wid, nid, 0, 0);
      break;
      case S2TParameter_Order_H01W0:
      // HW-N-C-1
      bot_offset = hid % 2 == 0 ?
        bottom->offset(nid, 0, hid, wid) :
        bottom->offset(nid, 0, bot_h - hid - 1, wid);
      top_offset = top->offset(wid * bot_h + hid, nid, 0, 0);
      break;
      case S2TParameter_Order_H10W1:
      // HW-N-C-1
      bot_offset = hid % 2 == 0 ?
        bottom->offset(nid, 0, bot_h - hid - 1, bot_w - wid - 1) :
        bottom->offset(nid, 0, hid, bot_w - wid - 1);
      top_offset = top->offset(wid * bot_h + hid, nid, 0, 0);
      break;
      case S2TParameter_Order_W01H0:
      // WH-N-C-1
      bot_offset = wid % 2 == 0 ?
        bottom->offset(nid, 0, hid, wid) :
        bottom->offset(nid, 0, hid, bot_w - wid - 1);
      top_offset = top->offset(hid * bot_w + wid, nid, 0, 0);
      break;
      case S2TParameter_Order_W10H1:
      // WH-N-C-1
      bot_offset = wid % 2 == 0 ?
        bottom->offset(nid, 0, bot_h - hid - 1, bot_w - wid - 1) :
        bottom->offset(nid, 0, bot_h - hid - 1, wid);
      top_offset = top->offset(hid * bot_w + wid, nid, 0, 0);
      break;
      default:
      LOG(FATAL) << "Unsupported S2T order!";
      break;
    }
  }
};

}  // namespace caffe

#endif  // CAFFE_S2T_LAYER_HPP_
