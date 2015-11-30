#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/weighted_split_layer.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void WeightedSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(count_, Dtype(0.), bottom[0]->mutable_gpu_diff());
  const Dtype *diff_weights_data = diff_weights_.gpu_data();
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype diff_weight = diff_weights_data[i];
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy(count_, Dtype(diff_weight), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedSplitLayer);

}  // namespace caffe
