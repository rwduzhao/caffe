/*=============================================================================
#     FileName: one_step_unified_forget_gate_memory_layer.cu
#   Desciption: One step unified forget gate memory layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-12 14:40:46
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/one_step_unified_forget_gate_memory_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_row_sum(
  const int num_row, const int dim,
  const Dtype* data, Dtype* results) {

  CUDA_KERNEL_LOOP(row_id, num_row) {
    Dtype result = 0;
    for (int d = 0; d < dim; ++d)
      result += data[row_id * dim + d];
    results[row_id] = result;
  }
}

template <typename Dtype>
__global__ void kernel_row_mean(
  const int num_row, const int dim,
  const Dtype* data, Dtype* results) {

  CUDA_KERNEL_LOOP(row_id, num_row) {
    Dtype result = 0;
    for (int d = 0; d < dim; ++d)
      result += data[row_id * dim + d];
    results[row_id] = result / Dtype(dim);
  }
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // initialize previous state
  CHECK_EQ(bottom[1]->count(), c_0_.count());
  caffe_copy(c_0_.count(), bottom[1]->gpu_data(), c_0_.mutable_gpu_data());
  caffe_copy(h_0_.count(), c_0_.gpu_data(), h_0_.mutable_gpu_data());
  CHECK_EQ(bottom[2]->count(), e_0_.count());
  caffe_copy(e_0_.count(), bottom[2]->gpu_data(), e_0_.mutable_gpu_data());

  // bias -> unified_pre_gate
  const int bias_blob_id = 0;
  const Dtype *bias = this->blobs_[bias_blob_id]->gpu_data();
  Dtype *unified_pre_gate_data = unified_pre_gate_.mutable_gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, num_gate_ * unified_dim_, 1,
                 (Dtype)1., bias_multiplier_.gpu_data(), bias, (Dtype)0., unified_pre_gate_data);
  // input -> unified_pre_gate
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const int weight_i_blob_id = 1;
  const Dtype *weight_i = this->blobs_[weight_i_blob_id]->gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_, num_gate_ * unified_dim_, input_dim_,
                 (Dtype)1., bottom_data, weight_i, (Dtype)1., unified_pre_gate_data);

  // compute recurrent forward propagation
  for (int t = 0; t < time_step_; ++t) {
    // h_t_1 -> unified_pre_gate
    Dtype *top_data = top_.mutable_gpu_data();
    Dtype *h_t = top_data + top_.offset(t);
    const Dtype *h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();
    const int weight_h_blob_id = 2;
    const Dtype *weight_h = this->blobs_[weight_h_blob_id]->gpu_data();
    Dtype *unified_pre_o_t = unified_pre_gate_data + unified_pre_gate_.offset(t);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1 * batch_size_, num_gate_ * unified_dim_, hidden_dim_,
                   (Dtype)1., h_t_1, weight_h, (Dtype)1., unified_pre_o_t);
    // extra -> unified_pre_gate
    Dtype *e_t = e_0_.mutable_gpu_data() + e_0_.offset(t);  //TOFIX
    const Dtype *e_t_1 = t > 0 ? (e_t - e_0_.offset(1)) : e_0_.gpu_data();
    const int weight_e_blob_id = 3;
    const Dtype *weight_e = this->blobs_[weight_e_blob_id]->gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1 * batch_size_, num_gate_ * unified_dim_, extra_dim_,
                   (Dtype)1., e_t_1, weight_e, (Dtype)1., unified_pre_o_t);

    // unified_pre-gate -> pre_gate
    vector<int> weight_u_shape;
    weight_u_shape.push_back(hidden_dim_);
    weight_u_shape.push_back(1);
    Blob<Dtype> weight_u_;
    weight_u_.Reshape(weight_u_shape);
    caffe_gpu_set(weight_u_.count(), (Dtype)1.0, weight_u_.mutable_gpu_data());
    Dtype *pre_gate_data = pre_gate_.mutable_gpu_data();
    Dtype *pre_o_t = pre_gate_data + pre_gate_.offset(t);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1 * batch_size_, num_gate_ * hidden_dim_, unified_dim_,
                   (Dtype)1., unified_pre_o_t, weight_u_.gpu_data(), (Dtype)0., pre_o_t);

    // pre_gate -> gate
    Dtype *gate_data = gate_.mutable_gpu_data();
    Dtype *o_t = gate_data + gate_.offset(t);
    caffe_gpu_sigmoid(batch_size_ * hidden_dim_, pre_o_t, o_t);
    // c_t = c_t_1 = c_0
    Dtype *cell_data = cell_.mutable_gpu_data();
    Dtype *c_t = cell_data + cell_.offset(t);
    const Dtype *c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();
    caffe_copy(batch_size_ * hidden_dim_, c_t_1, c_t);
    // c_t & gate -> h_t
    caffe_gpu_mul(batch_size_ * hidden_dim_, o_t, c_t, h_t);
  }
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  for (int t = time_step_ - 1; t >= 0; --t) {
    // dh_t -> do_t
    const Dtype *cell_data = cell_.gpu_data();
    const Dtype *c_t = cell_data + cell_.offset(t);
    Dtype *top_diff = top_.mutable_gpu_diff();
    Dtype *dh_t = top_diff + top_.offset(t);
    Dtype *gate_diff = gate_.mutable_gpu_diff();
    Dtype *do_t = gate_diff + gate_.offset(t);
    caffe_gpu_mul(batch_size_ * hidden_dim_, c_t, dh_t, do_t);

    // do_t -> pre_do_t
    const Dtype *gate_data = gate_.gpu_data();
    const Dtype *o_t = gate_data + gate_.offset(t);
    Dtype *pre_gate_diff = pre_gate_.mutable_gpu_diff();
    Dtype *pre_do_t = pre_gate_diff + pre_gate_.offset(t);
    caffe_gpu_sigmoid_diff(batch_size_ * hidden_dim_, o_t, do_t, pre_do_t);
    if (clipping_threshold_ > 0.0f)
      caffe_gpu_bound(batch_size_ * hidden_dim_, pre_do_t, -clipping_threshold_, clipping_threshold_, pre_do_t);

    // pre_do_t -> unified_pre_do_t
    Dtype *unified_pre_gate_diff = unified_pre_gate_.mutable_gpu_diff();
    Dtype *unified_pre_do_t = unified_pre_gate_diff + unified_pre_gate_.offset(t);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_row_sum<Dtype><<<CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS>>>(
      batch_size_, hidden_dim_, pre_do_t, unified_pre_do_t);
  }
  Dtype *cell_diff = cell_.mutable_gpu_diff();
  caffe_gpu_mul(cell_.count(), gate_.gpu_diff(), top_.gpu_diff(), cell_diff);

  // gradients
  const Dtype *unified_pre_gate_diff = unified_pre_gate_.gpu_diff();
  // gradient w.r.t. bias
  const int bias_blob_id = 0;
  if (this->param_propagate_down_[bias_blob_id]) {
    Dtype *bias_diff = this->blobs_[bias_blob_id]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[bias_blob_id]->count(), (Dtype)0.0, bias_diff);
    caffe_gpu_gemv(CblasTrans, time_step_ * batch_size_, num_gate_ * unified_dim_,
                   (Dtype)1., unified_pre_gate_diff, bias_multiplier_.gpu_data(), (Dtype)1., bias_diff);
  }
  // gradient w.r.t. weight_i
  const int weight_i_blob_id = 1;
  if (this->param_propagate_down_[weight_i_blob_id]) {
    Dtype *weight_i_diff = this->blobs_[weight_i_blob_id]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[weight_i_blob_id]->count(), (Dtype)0.0, weight_i_diff);
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_gate_ * unified_dim_, input_dim_, time_step_ * batch_size_,
                   (Dtype)1., unified_pre_gate_diff, bottom[0]->gpu_data(), (Dtype)1., weight_i_diff);
  }
  // gradient w.r.t. weight_h
  const int weight_h_blob_id = 2;
  if (this->param_propagate_down_[weight_h_blob_id]) {
    Dtype *weight_h_diff = this->blobs_[weight_h_blob_id]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[weight_h_blob_id]->count(), (Dtype)0.0, weight_h_diff);
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_gate_ * unified_dim_, hidden_dim_, 1 * batch_size_,
                   (Dtype)1., unified_pre_gate_diff, h_0_.gpu_data(), (Dtype)1., weight_h_diff);
  }
  // gradient w.r.t. weight_e
  const int weight_e_blob_id = 3;
  if (this->param_propagate_down_[weight_e_blob_id]) {
    Dtype *weight_e_diff = this->blobs_[weight_e_blob_id]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[weight_e_blob_id]->count(), (Dtype)0.0, weight_e_diff);
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_gate_ * unified_dim_, extra_dim_, 1 * batch_size_,
                   (Dtype)1., unified_pre_gate_diff, e_0_.gpu_data(), (Dtype)1., weight_e_diff);
  }
  // gradient w.r.t. bottom data
  if (propagate_down[0])
    caffe_copy(bottom[0]->count(), cell_.gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(OneStepUnifiedForgetGateMemoryLayer);

}  // namespace caffe
