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
  CHECK_EQ(c_0_.count(), bottom[1]->count());
  caffe_copy(c_0_.count(), bottom[1]->gpu_data(), c_0_.mutable_gpu_data());
  CHECK_EQ(e_0_.count(), bottom[2]->count());
  caffe_copy(e_0_.count(), bottom[2]->gpu_data(), e_0_.mutable_gpu_data());

  // compute input to hidden forward propagation and add bias
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *weight_i = this->blobs_[0]->gpu_data();
  Dtype *unified_pre_gate_data = unified_pre_gate_.mutable_gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_, 1 * 1, input_dim_, (Dtype)1.,
                 bottom_data, weight_i, (Dtype)0., unified_pre_gate_data);
  const Dtype *bias = this->blobs_[2]->gpu_data();
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, 1 * 1, 1, (Dtype)1.,
                 bias_multiplier_.gpu_data(), bias, (Dtype)1., unified_pre_gate_data);

  // compute recurrent forward propagation
  for (int t = 0; t < time_step_; ++t) {
    Dtype *top_data = top_.mutable_gpu_data();
    Dtype *h_t = top_data + top_.offset(t);
    Dtype *cell_data = cell_.mutable_gpu_data();
    Dtype *c_t = cell_data + cell_.offset(t);
    Dtype *e_t = e_0_.mutable_gpu_data() + e_0_.offset(t);
    Dtype *gate_data = gate_.mutable_gpu_data();
    Dtype *o_t = gate_data + gate_.offset(t);
    Dtype *pre_gate_data = pre_gate_.mutable_gpu_data();
    Dtype *pre_o_t = pre_gate_data + pre_gate_.offset(t);
    Dtype *unified_pre_o_t = unified_pre_gate_data + unified_pre_gate_.offset(t);
    const Dtype *c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();
    const Dtype *e_t_1 = t > 0 ? (e_t - e_0_.offset(1)) : e_0_.gpu_data();

    // extra-to-hidden propagation
    const Dtype *weight_e = this->blobs_[1]->gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, 1 * 1, extra_dim_, (Dtype)1.,
                   e_t_1, weight_e, (Dtype)1., unified_pre_o_t);

    // unified pre-gate to pre-gate
    vector<int> weight_u_shape;
    weight_u_shape.push_back(hidden_dim_);
    weight_u_shape.push_back(1);
    Blob<Dtype> weight_u_;
    weight_u_.Reshape(weight_u_shape);
    caffe_gpu_set(weight_u_.count(), Dtype(1.0), weight_u_.mutable_gpu_data());
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, hidden_dim_, 1, (Dtype)1.,
                   unified_pre_o_t, weight_u_.mutable_gpu_data(), (Dtype)0., pre_o_t);
    pre_o_t = pre_gate_data + pre_gate_.offset(t);
    caffe_gpu_sigmoid(batch_size_ * hidden_dim_, pre_o_t, o_t);

    // compute cell : c(t) = f(t) * c(t - 1) + i(t) * g(t)
    caffe_copy(batch_size_ * hidden_dim_, c_t_1, c_t);
    caffe_gpu_mul(batch_size_ * hidden_dim_, o_t, c_t, h_t);
  }
//  // Preserve cell state and output value for truncated BPTT
//  const Dtype *cell_data = cell_.mutable_gpu_data();
//  caffe_copy(batch_size_ * hidden_dim_, cell_data + cell_.offset(time_step_ - 1), c_T_.mutable_gpu_data());
//  caffe_copy(batch_size_ * hidden_dim_, top_data + top_.offset(time_step_ - 1), h_T_.mutable_gpu_data());
}

template <typename Dtype>
void OneStepUnifiedForgetGateMemoryLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // const Dtype* top_data = top_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // const Dtype *weight_i = this->blobs_[0]->gpu_data();
  // const Dtype *weight_e = this->blobs_[1]->gpu_data();


  for (int t = time_step_ - 1; t >= 0; --t) {
    // dh_t -> do_t
    Dtype *top_diff = top_.mutable_gpu_diff();
    Dtype *dh_t = top_diff + top_.offset(t);
    const Dtype *cell_data = cell_.gpu_data();
    const Dtype *c_t = cell_data + cell_.offset(t);
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

    Dtype *unified_pre_gate_diff = unified_pre_gate_.mutable_gpu_diff();
    Dtype *unified_pre_do_t = unified_pre_gate_diff + unified_pre_gate_.offset(t);

    // pre_do_t -> unified_pre_do_t
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_row_sum<Dtype><<<CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS>>>(
      batch_size_, hidden_dim_, pre_do_t, unified_pre_do_t);
  }

  Dtype *unified_pre_gate_diff = unified_pre_gate_.mutable_gpu_diff();

  if (this->param_propagate_down_[0]) {
    // gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 1 * 1, input_dim_, time_step_ * batch_size_, (Dtype)1.,
                   unified_pre_gate_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // gradient w.r.t. hidden-to-hidden weight
    // caffe_gpu_gemm(CblasTrans, CblasNoTrans, 1 * hidden_dim_, hidden_dim_, (time_step_ - 1) * batch_size_, (Dtype)1.,
    //                pre_gate_diff + pre_gate_.offset(1), top_data, (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
    // add gradient from previous time-step
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 1 * 1, extra_dim_, 1 * batch_size_, (Dtype)1.,
                   unified_pre_gate_diff, e_0_.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[2]) {
    // gradient w.r.t. bias
    caffe_gpu_gemv(CblasTrans, time_step_ * batch_size_, 1 * 1, (Dtype)1., unified_pre_gate_diff,
                   bias_multiplier_.gpu_data(), (Dtype)1., this->blobs_[2]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OneStepUnifiedForgetGateMemoryLayer);

}  // namespace caffe
