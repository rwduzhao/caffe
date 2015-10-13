/*=============================================================================
#     FileName: one_step_forget_gate_memory_layer.cpp
#   Desciption: One step forget gate memory layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-08 21:49:02
#      History:
=============================================================================*/

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/layers/one_step_forget_gate_memory_layer.hpp"

namespace caffe {

template <typename Dtype>
void OneStepForgetGateMemoryLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const OneStepForgetGateMemoryParameter layer_param = this->layer_param_.one_step_forget_gate_memory_param();
  clipping_threshold_ = layer_param.clipping_threshold();
  hidden_dim_ = layer_param.num_output();

  vector<int> clip_mul_shape(1, hidden_dim_);
  clip_multiplier_.Reshape(clip_mul_shape);
  caffe_set(clip_multiplier_.count(), Dtype(1), clip_multiplier_.mutable_cpu_data());

  if (this->blobs_.size() == 3) {
    // blobs_[0]: weight_i (1 * hidden_dim_ by input_dim_)
    // blobs_[1]: weight_e (1 * hidden_dim_ by hidden_dim_)
    // blobs_[2]: bias (1 * hidden_dim_)
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }

}

template <typename Dtype>
void OneStepForgetGateMemoryLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const OneStepForgetGateMemoryParameter layer_param = this->layer_param_.one_step_forget_gate_memory_param();

  // check ups
  // input
  const Blob<Dtype> *x_0 = bottom[0];
  time_step_ = 1;
  batch_size_ = x_0->num() / time_step_;
  input_dim_ = x_0->channels();
  CHECK_EQ(input_dim_, x_0->count() / x_0->num());
  // c_0
  const Blob<Dtype> *c_0 = bottom[1];
  CHECK_EQ(batch_size_, c_0->num());
  CHECK_EQ(hidden_dim_, c_0->channels());
  // e_0
  const Blob<Dtype> *e_0 = bottom[2];
  CHECK_EQ(batch_size_, e_0->num());
  extra_dim_ = e_0->channels();

  if (this->blobs_.size() == 0) {  // init weights and biases
    this->blobs_.resize(3);
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(layer_param.weight_filler()));

    // weight_hi
    vector<int> weight_shape;
    weight_shape.push_back(1 * hidden_dim_);
    weight_shape.push_back(input_dim_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // weight_he
    weight_shape.clear();
    weight_shape.push_back(1 * hidden_dim_);
    weight_shape.push_back(extra_dim_);
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // bias term
    vector<int> bias_shape(1, 1 * hidden_dim_);
    this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(layer_param.bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }

  // c_0
  vector<int> cell_shape;
  cell_shape.push_back(batch_size_);
  cell_shape.push_back(hidden_dim_);
  c_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);

  // e_0
  vector<int> extra_shape;
  extra_shape.clear();
  extra_shape.push_back(batch_size_);
  extra_shape.push_back(extra_dim_);
  e_0_.Reshape(extra_shape);

  // gate and pre_gate
  vector<int> gate_shape;
  gate_shape.push_back(time_step_);
  gate_shape.push_back(batch_size_);
  gate_shape.push_back(1);
  gate_shape.push_back(hidden_dim_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);

  // top and cell
  vector<int> original_top_shape;
  original_top_shape.push_back(time_step_ * batch_size_);
  original_top_shape.push_back(hidden_dim_);
  top[0]->Reshape(original_top_shape);
  vector<int> top_shape;
  top_shape.push_back(time_step_);
  top_shape.push_back(batch_size_);
  top_shape.push_back(hidden_dim_);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);
  cell_.Reshape(top_shape);

  // bias multiplier
  vector<int> multiplier_shape(1, batch_size_ * time_step_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void OneStepForgetGateMemoryLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype *bottom_data = bottom[0]->cpu_data();
  const Dtype *weight_i = this->blobs_[0]->cpu_data();
  const Dtype *weight_e = this->blobs_[1]->cpu_data();
  const Dtype *bias = this->blobs_[2]->cpu_data();

  CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
  Dtype *top_data = top_.mutable_cpu_data();

  Dtype *pre_gate_data = pre_gate_.mutable_cpu_data();
  Dtype *gate_data = gate_.mutable_cpu_data();
  Dtype *cell_data = cell_.mutable_cpu_data();

  // initialize previous state
  CHECK_EQ(c_0_.count(), bottom[1]->count());
  caffe_copy(c_0_.count(), bottom[1]->cpu_data(), c_0_.mutable_cpu_data());
  CHECK_EQ(e_0_.count(), bottom[2]->count());
  caffe_copy(e_0_.count(), bottom[2]->cpu_data(), e_0_.mutable_cpu_data());

  // compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_, 1 * hidden_dim_, input_dim_, (Dtype)1.,
                 bottom_data, weight_i, (Dtype)0., pre_gate_data);

  // Add bias
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, 1 * hidden_dim_, 1, (Dtype)1.,
                 bias_multiplier_.cpu_data(), bias, (Dtype)1., pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < time_step_; ++t) {
    Dtype *h_t = top_data + top_.offset(t);
    Dtype *c_t = cell_data + cell_.offset(t);
    Dtype *e_t = e_0_.mutable_cpu_data() + e_0_.offset(t);
    Dtype *o_t = gate_data + gate_.offset(t);
    Dtype *pre_o_t = pre_gate_data + pre_gate_.offset(t);
    const Dtype *c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();
    const Dtype *e_t_1 = t > 0 ? (e_t - e_0_.offset(1)) : e_0_.cpu_data();

    // extra-to-hidden propagation
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, extra_dim_, hidden_dim_, (Dtype)1.,
                   e_t_1, weight_e, (Dtype)1., pre_gate_data + pre_gate_.offset(t));

    for (int n = 0; n < batch_size_; ++n) {
      caffe_sigmoid(1 * hidden_dim_, pre_o_t, o_t);

      // compute cell : c(t) = f(t) * c(t - 1) + i(t) * g(t)
      caffe_copy(hidden_dim_, c_t_1, c_t);

      // compute output
      caffe_mul(hidden_dim_, o_t, c_t, h_t);

      h_t += hidden_dim_;
      e_t_1 += extra_dim_;
      c_t += hidden_dim_;
      c_t_1 += hidden_dim_;
      o_t += 1 * hidden_dim_;
      pre_o_t += 1 * hidden_dim_;
    }
  }
  // Preserve cell state and output value for truncated BPTT
  caffe_copy(batch_size_ * hidden_dim_, cell_data + cell_.offset(time_step_ - 1), c_T_.mutable_cpu_data());
  caffe_copy(batch_size_ * hidden_dim_, top_data + top_.offset(time_step_ - 1), h_T_.mutable_cpu_data());
}

template <typename Dtype>
void OneStepForgetGateMemoryLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  // const Dtype* top_data = top_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // const Dtype *weight_i = this->blobs_[0]->cpu_data();
  // const Dtype *weight_e = this->blobs_[1]->cpu_data();
  const Dtype *gate_data = gate_.cpu_data();
  const Dtype *cell_data = cell_.cpu_data();

  Dtype *top_diff = top_.mutable_cpu_diff();
  Dtype *pre_gate_diff = pre_gate_.mutable_cpu_diff();
  Dtype *gate_diff = gate_.mutable_cpu_diff();
  Dtype *cell_diff = cell_.mutable_cpu_diff();

  for (int t = time_step_ - 1; t >= 0; --t) {
    Dtype *dh_t = top_diff + top_.offset(t);
    const Dtype *c_t = cell_data + cell_.offset(t);
    Dtype *dc_t = cell_diff + cell_.offset(t);
    const Dtype *o_t = gate_data + gate_.offset(t);
    Dtype *do_t = gate_diff + gate_.offset(t);
    Dtype *pre_do_t = pre_gate_diff + pre_gate_.offset(t);

    for (int n = 0; n < batch_size_; ++n) {
      // Output gate : tanh(c(t)) * h_diff(t)
      // caffe_mul(hidden_dim_, tanh_c_t, dh_t, do_t);
      caffe_mul(hidden_dim_, c_t, dh_t, do_t);

      // Compute derivate before nonlinearity
      caffe_sigmoid_diff(1 * hidden_dim_, o_t, do_t, pre_do_t);

      // Clip deriviates before nonlinearity
      if (clipping_threshold_ > 0.0f) {
        caffe_bound(1 * hidden_dim_, pre_do_t, -clipping_threshold_, clipping_threshold_, pre_do_t);
      }

      dh_t += hidden_dim_;
      c_t += hidden_dim_;
      dc_t += hidden_dim_;
      do_t += hidden_dim_;
      o_t += hidden_dim_;
      pre_do_t += hidden_dim_;
    }

  //if (t > 0) {
  //  Dtype *dh_t_1 = top_diff + top_.offset(t - 1);
  //  // Backprop output errors to the previous time step
  //  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, batch_size_, hidden_dim_, 1 * hidden_dim_,
  //      (Dtype)1., pre_gate_diff + pre_gate_.offset(t),
  //      weight_h, (Dtype)0., clipped_.mutable_cpu_data());
  //  if (clip)
  //    caffe_mul(batch_size_ * hidden_dim_, clipped_.cpu_data(), mask + clip_mask_.offset(t), clipped_.mutable_cpu_data());
  //  caffe_add(batch_size_ * hidden_dim_, dh_t_1, clipped_.cpu_data(), dh_t_1);
  //}
  }

  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 1 * hidden_dim_, input_dim_, time_step_ * batch_size_, (Dtype)1.,
                   pre_gate_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    // caffe_cpu_gemm(CblasTrans, CblasNoTrans, 1 * hidden_dim_, hidden_dim_, (time_step_ - 1) * batch_size_, (Dtype)1.,  //TODO
    //                pre_gate_diff + pre_gate_.offset(1), top_data,
    //                (Dtype)1., this->blobs_[1]->mutable_cpu_diff());

    // Add Gradient from previous time-step
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, 1 * extra_dim_, hidden_dim_, 1, (Dtype)1.,
                   pre_gate_diff, e_0_.cpu_data(),
                   (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }
  if (this->param_propagate_down_[2]) {
    // Gradient w.r.t. bias
    caffe_cpu_gemv(CblasTrans, time_step_ * batch_size_, 1 * hidden_dim_, (Dtype)1., pre_gate_diff,
                   bias_multiplier_.cpu_data(), (Dtype)1.,
                   this->blobs_[2]->mutable_cpu_diff());
  }
//  if (propagate_down[0]) {
//    // Gradient w.r.t. bottom data
//    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, input_dim_, 1 * hidden_dim_, (Dtype)1.,
//                   pre_gate_diff, weight_i, (Dtype)0., bottom[0]->mutable_cpu_diff());
//  }
}

#ifdef CPU_ONLY
STUB_GPU(OneStepForgetGateMemoryLayer);
#endif

INSTANTIATE_CLASS(OneStepForgetGateMemoryLayer);
REGISTER_LAYER_CLASS(OneStepForgetGateMemory);

}  // namespace caffe
