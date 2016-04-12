#include <algorithm>
#include <vector>

#include "caffe/layers/group_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  vector<Blob<Dtype>*> softmax_bottom_vec;
  softmax_bottom_vec.push_back(bottom[0]);
  softmax_layer_->SetUp(softmax_bottom_vec, top);
}

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Blob<Dtype>* data_blob = bottom[0];
  Blob<Dtype>* rois_blob = bottom[1];
  CHECK_EQ(data_blob->num(), rois_blob->num());

  const int rois_data_count = rois_blob->count();
  const int num_rois = rois_blob->num();
  const int rois_step = rois_data_count / num_rois;

  const Dtype *rois_data = rois_blob->cpu_data();
  int num_group = 0;
  group_ids_.Reshape(num_rois, 1, 1, 1);
  int *group_ids_data = group_ids_.mutable_cpu_data();
  for (int rois_id = 0; rois_id < num_rois; ++rois_id) {
    const int group_id = static_cast<int>(rois_data[rois_id * rois_step]);
    group_ids_data[rois_id] = group_id;
    num_group = std::max(num_group, group_id + 1);
  }

  group_ids_vec_.clear();
  for (int group_id = 0; group_id < num_group; ++group_id) {
    vector<int> ids;
    ids.clear();
    group_ids_vec_.push_back(ids);
  }

  for (int rois_id = 0; rois_id < num_rois; ++rois_id) {
    const int group_id = group_ids_data[rois_id];
    group_ids_vec_[group_id].push_back(rois_id);
  }

//  softmax_layers_.clear();
//  for (int group_id = 0; group_id < num_group; ++group_id) {
//    const int group_size = group_ids_vec_[group_id].size();
//
//    vector<Blob<Dtype> > softmax_bottom_vec;
//    Blob<Dtype> softmax_bottom(1, group_size, 1, 1);
//    softmax_bottom_vecs_.push_back(softmax_bottom_vec);
//    vector<Blob<Dtype>*> softmax_bottom_ptr_vec;
//    softmax_bottom_ptr_vec.push_back(&softmax_bottom);
//    softmax_bottom_ptr_vecs_.push_back(softmax_bottom_ptr_vec);
//
//    vector<Blob<Dtype> > softmax_top_vec;
//    Blob<Dtype> softmax_top(1, group_size, 1, 1);
//    vector<Blob<Dtype>*> softmax_top_ptr_vec;
//    softmax_top_vecs_.push_back(softmax_top_vec);
//    softmax_top_ptr_vec.push_back(&softmax_top);
//    softmax_bottom_ptr_vecs_.push_back(softmax_top_ptr_vec);
//
//    shared_ptr<Layer<Dtype> > softmax_layer;
//    LayerParameter softmax_param(this->layer_param_);
//    softmax_param.set_type("Softmax");
//    softmax_layer = LayerRegistry<Dtype>::CreateLayer(softmax_param);
//    softmax_layer->SetUp(softmax_bottom_ptr_vec, softmax_top_ptr_vec);
//    softmax_layer->Reshape(softmax_bottom_ptr_vec, softmax_top_ptr_vec);
//    softmax_layers_.push_back(softmax_layer);
//  }

  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GroupSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(GroupSoftmaxLayer);
#endif

INSTANTIATE_CLASS(GroupSoftmaxLayer);
REGISTER_LAYER_CLASS(GroupSoftmax);

}  // namespace caffe
