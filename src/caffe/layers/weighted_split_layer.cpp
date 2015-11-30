#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/weighted_split_layer.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  vector<int> blob_shape;
  blob_shape.clear();
  blob_shape.push_back(num_top);
  diff_weights_.Reshape(blob_shape);
  Dtype *diff_weights_data = diff_weights_.mutable_cpu_data();
  const WeightedSplitParameter layer_param = this->layer_param_.weighted_split_param();
  if (layer_param.diff_weight_size() > 0) {
    CHECK_EQ(layer_param.diff_weight_size(), num_top);
    for (int weight_id = 0; weight_id < num_top; ++weight_id)
      diff_weights_data[weight_id] = Dtype(layer_param.diff_weight(weight_id));
  } else {
    for (int weight_id = 0; weight_id < num_top; ++weight_id)
      diff_weights_data[weight_id] = Dtype(1.);
  }
}


template <typename Dtype>
void WeightedSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the WeightedSplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

template <typename Dtype>
void WeightedSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void WeightedSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_set(count_, Dtype(0.), bottom[0]->mutable_cpu_diff());
  const Dtype *diff_weights_data = diff_weights_.cpu_data();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype diff_weight = diff_weights_data[i];
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_axpy(count_, Dtype(diff_weight), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(WeightedSplitLayer);
#endif

INSTANTIATE_CLASS(WeightedSplitLayer);
REGISTER_LAYER_CLASS(WeightedSplit);

}  // namespace caffe
