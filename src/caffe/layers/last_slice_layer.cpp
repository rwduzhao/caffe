#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/last_slice_layer.hpp"

namespace caffe {

template <typename Dtype>
void LastSliceLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const LastSliceParameter& layer_param = this->layer_param_.last_slice_param();
  if (top.size() > 1) {
    CHECK(!layer_param.has_num_division());
    num_division_ = top.size();
  } else {
    CHECK(top.size() == 1);
    num_division_ = layer_param.num_division();
  }
  CHECK_GE(num_division_, 1);
}

template <typename Dtype>
void LastSliceLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  if (top.size() == 1) {
    slice_count_ = bottom[0]->count() / num_division_;

    vector<int> blob_shape = bottom[0]->shape();
    blob_shape[0] /= num_division_;
    top[0]->Reshape(blob_shape);
  }
}

template <typename Dtype>
void LastSliceLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LastSliceLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LastSliceLayer);
#endif

INSTANTIATE_CLASS(LastSliceLayer);
REGISTER_LAYER_CLASS(LastSlice);

}  // namespace caffe
