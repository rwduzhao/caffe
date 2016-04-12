#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/group_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int group_id = 0; group_id < group_ids_vec_.size(); ++group_id) {
    const vector<int> group_ids = group_ids_vec_[group_id];
    const int group_size = group_ids.size();

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_.Reshape(1, group_size, 1, 1);
    softmax_bottom_vec.push_back(&softmax_bottom_);
    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_.Reshape(1, group_size, 1, 1);
    softmax_top_vec.push_back(&softmax_top_);
    softmax_layer_->Reshape(softmax_bottom_vec, softmax_top_vec);

    // copy data form bottom to softmax bottom
    const Dtype* bottom_data = bottom[0]->cpu_data();
    for (int index = 0; index < group_ids.size(); ++index) {
      Dtype* softmax_bottom_data = softmax_bottom_.mutable_cpu_data();
      softmax_bottom_data[index] = bottom_data[group_ids[index]];
    }

    // forward the softmax
    softmax_layer_->Forward(softmax_bottom_vec, softmax_top_vec);

    // copy data from softmax top to top
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int index = 0; index < group_ids.size(); ++index) {
      const Dtype* softmax_top_data = softmax_top_.mutable_cpu_data();
      top_data[group_ids[index]] = softmax_top_data[index];
    }
  }
}

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    for (int group_id = 0; group_id < group_ids_vec_.size(); ++group_id) {
      const vector<int> group_ids = group_ids_vec_[group_id];
      const int group_size = group_ids.size();

      vector<Blob<Dtype>*> softmax_bottom_vec;
      softmax_bottom_.Reshape(1, group_size, 1, 1);
      softmax_bottom_vec.push_back(&softmax_bottom_);
      vector<Blob<Dtype>*> softmax_top_vec;
      softmax_top_.Reshape(1, group_size, 1, 1);
      softmax_top_vec.push_back(&softmax_top_);
      softmax_layer_->Reshape(softmax_bottom_vec, softmax_top_vec);

      // copy diff form top to softmax top
      const Dtype* top_diff = top[0]->cpu_diff();
      for (int index = 0; index < group_ids.size(); ++index) {
        Dtype* softmax_top_diff = softmax_top_.mutable_cpu_diff();
        softmax_top_diff[index] = top_diff[group_ids[index]];
      }

      // backward the softmax
      softmax_layer_->Backward(softmax_top_vec, propagate_down, softmax_bottom_vec);

      // copy diff from softmax bottom to bottom
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      for (int index = 0; index < group_ids.size(); ++index) {
        const Dtype* softmax_bottom_diff = softmax_bottom_.mutable_cpu_diff();
        bottom_diff[group_ids[index]] = softmax_bottom_diff[index];
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GroupSoftmaxLayer);

}  // namespace caffe
