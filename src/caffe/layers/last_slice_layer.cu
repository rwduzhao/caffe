#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/last_slice_layer.hpp"

namespace caffe {

template <typename Dtype>
void LastSliceLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  Dtype *bottom_data = bottom[0]->mutable_gpu_data();
  const int bottom_offset = bottom[0]->count() - slice_count_;
  Dtype *top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();
  caffe_copy(top_count, bottom_data + bottom_offset, top_data);
}

template <typename Dtype>
void LastSliceLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  if (!propagate_down[0]) { return; }

  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_offset = bottom[0]->count() - slice_count_;
  caffe_gpu_set(bottom_offset, Dtype(0.), bottom_diff);

  const Dtype *top_diff = top[0]->gpu_diff();
  const int top_count = top[0]->count();
  caffe_copy(top_count, top_diff, bottom_diff + bottom_offset);
}

INSTANTIATE_LAYER_GPU_FUNCS(LastSliceLayer);

}  // namespace caffe
