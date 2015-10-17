/*=============================================================================
#     FileName: self_gated_layer.cu
#   Desciption: self gated layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-16 15:59:23
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/self_gated_layer.hpp"

namespace caffe {

template <typename Dtype>
void SelfGatedLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // input -> pre_gate -> gate
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_layer_; ++gate_net_layer_id) {
    const int gate_layer_input_dim = gate_layer_input_dims_[gate_net_layer_id];
    const int gate_layer_output_dim = gate_layer_output_dims_[gate_net_layer_id];

    // bias -> pre_top
    const int bias_blob_id = gate_net_layer_id * 2;
    const Dtype *bias_data = this->blobs_[bias_blob_id]->gpu_data();
    Dtype *gate_net_pre_top_data = gate_net_pre_tops_[gate_net_layer_id]->mutable_gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, gate_layer_output_dim, 1,
                   Dtype(1.), bias_multiplier_.gpu_data(), bias_data, Dtype(0.), gate_net_pre_top_data);

    // weight -> pre_top
    const Dtype *bottom_data = gate_net_layer_id == 0 ? bottom[0]->gpu_data() : gate_net_tops_[gate_net_layer_id - 1]->gpu_data();
    const int weight_blob_id = gate_net_layer_id * 2 + 1;
    const Dtype *weight_data = this->blobs_[weight_blob_id]->gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_, gate_layer_output_dim, gate_layer_input_dim,
                   Dtype(1.), bottom_data, weight_data, Dtype(1.), gate_net_pre_top_data);

    // pre_top -> top
    Dtype *gate_net_top_data = gate_net_tops_[gate_net_layer_id]->mutable_gpu_data();
    caffe_gpu_sigmoid(gate_net_tops_[gate_net_layer_id]->count(), gate_net_pre_top_data, gate_net_top_data);
  }

  // output
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *gate_data = gate_.gpu_data();
  Dtype *top_data = top_.mutable_gpu_data();
  caffe_gpu_mul(top_.count(), gate_data, bottom_data, top_data);
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  // top_diff -> gate_diff
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *top_diff = top_.gpu_diff();
  Dtype *gate_diff = gate_.mutable_gpu_diff();
  caffe_gpu_mul(gate_.count(), bottom_data, top_diff, gate_diff);

  for (int gate_net_layer_id = num_gate_layer_ - 1; gate_net_layer_id >= 0; --gate_net_layer_id) {
    const int gate_layer_input_dim = gate_layer_input_dims_[gate_net_layer_id];
    const int gate_layer_output_dim = gate_layer_output_dims_[gate_net_layer_id];

    // gate_net_top_diff -> gate_net_pre_top_diff
    const int count = gate_net_pre_tops_[gate_net_layer_id]->count();
    const Dtype *gate_net_top_data = gate_net_tops_[gate_net_layer_id]->gpu_data();
    const Dtype *gate_net_top_diff = gate_net_tops_[gate_net_layer_id]->gpu_diff();
    Dtype *gate_net_pre_top_diff = gate_net_pre_tops_[gate_net_layer_id]->mutable_gpu_diff();
    caffe_gpu_sigmoid_diff(count, gate_net_top_data, gate_net_top_diff, gate_net_pre_top_diff);
    if (clipping_threshold_ > 0.0f)
      caffe_gpu_bound(count, gate_net_pre_top_diff, -clipping_threshold_, clipping_threshold_, gate_net_pre_top_diff);

    // gradient w.r.t. bias
    const int bias_blob_id = gate_net_layer_id * 2;
    Dtype *bias_diff = this->blobs_[bias_blob_id]->mutable_gpu_diff();
    caffe_gpu_gemv(CblasTrans, time_step_ * batch_size_, gate_layer_output_dim,
                   Dtype(1.), gate_net_pre_top_diff, bias_multiplier_.gpu_data(), Dtype(0.), bias_diff);
    // gradient w.r.t. weight
    const int weight_blob_id = gate_net_layer_id * 2 + 1;
    Dtype *weight_diff = this->blobs_[weight_blob_id]->mutable_gpu_diff();
    const Dtype *bottom_data = gate_net_layer_id == 0 ? bottom[0]->gpu_data() : gate_net_tops_[gate_net_layer_id - 1]->gpu_data();
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, gate_layer_output_dim, gate_layer_input_dim, time_step_ * batch_size_,
                   Dtype(1.), gate_net_pre_top_diff, bottom_data, Dtype(0.), weight_diff);
    // gradient w.r.t. bottom data
    const Dtype *weight_data = this->blobs_[weight_blob_id]->gpu_data();
    Dtype *bottom_diff = gate_net_layer_id == 0 ? bottom[0]->mutable_gpu_diff() : gate_net_tops_[gate_net_layer_id - 1]->mutable_gpu_diff();
    if (gate_net_layer_id > 0) {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, gate_layer_input_dim, gate_layer_output_dim,
                     Dtype(1.), gate_net_pre_top_diff, weight_data, Dtype(0.), bottom_diff);
    } else if (gate_net_layer_id == 0 && propagate_down[0]) {
      const Dtype *gate_data = gate_.gpu_data();
      const Dtype *top_diff = top_.gpu_diff();
      caffe_gpu_mul(bottom[0]->count(), gate_data, top_.gpu_diff(), bottom_diff);
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, gate_layer_input_dim, gate_layer_output_dim,
                     Dtype(1.), gate_net_pre_top_diff, weight_data, Dtype(1.), bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SelfGatedLayer);

}  // namespace caffe
