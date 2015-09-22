/*
 * =====================================================================================
 *       Filename:  balanced_sigmoid_cross_entropy_loss_layer.cu
 *    Description:  Multilabel sigmoid cross entropy loss layer.
 *        Version:  1.0
 *        Created:  09/21/2015 06:12:51 PM
 *       Revision:  none
 *       Compiler:  gcc
 *         Author:  rw.du.zhao@gmail.com
 * =====================================================================================
 */

#include <vector>
#include "caffe/layers/balanced_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // caffe_gpu_scal(count, loss_weight / num, bottom_diff);
    caffe_gpu_scal(count, loss_weight, bottom_diff);

    Dtype pos_count;
    caffe_gpu_dot(count, target, target, &pos_count);
    Dtype neg_count = Dtype(count) - pos_count;
    Dtype *cpu_bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* cpu_target = bottom[1]->cpu_data();
    const int num_label = count / num;
    const Dtype pos_scale = (pos_count / Dtype(num_label));
    const Dtype neg_scale = (neg_count / Dtype(num_label));
    for (int index = 0; index < count; ++index) {
      if (cpu_target[index] == 1) {
        cpu_bottom_diff[index] /= pos_scale;
      } else {
        cpu_bottom_diff[index] /= neg_scale;
      }
    }
    bottom_diff = bottom[0]->mutable_gpu_diff();
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(BalancedSigmoidCrossEntropyLossLayer);

}  // namespace caffe
