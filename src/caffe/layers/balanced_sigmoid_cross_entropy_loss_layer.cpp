#include <algorithm>
#include <vector>

#include "caffe/layers/balanced_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BalancedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }

  prop_skip_period_ = this->layer_param_.prop_skip_period();
  prop_skip_index_ = this->layer_param_.prop_skip_index();
  prop_skip_count_ = 0;
}

template <typename Dtype>
void BalancedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
    "BALANCED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
template <typename Dtype>
Dtype BalancedSigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void BalancedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  pos_count_ = 0;
  neg_count_ = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    ++valid_count;
    if (target[i] == 1)
      pos_count_ += Dtype(1.);
    else
      neg_count_ += Dtype(1.);
  }
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void BalancedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  const bool period_prop_down = (prop_skip_period_ == 0) ||
    (prop_skip_count_ != prop_skip_index_);
  if (propagate_down[0] && period_prop_down) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      for (int i = 0; i < count; ++i) {
        const int target_value = static_cast<int>(target[i]);
        if (target_value == ignore_label_) {
          bottom_diff[i] = 0;
        }
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    const Dtype pos_scale = pos_count_ == 0 ? 1.0 : loss_weight * (neg_count_ / (pos_count_ + neg_count_));
    const Dtype neg_scale = neg_count_ == 0 ? 1.0 : loss_weight * (pos_count_ / (pos_count_ + neg_count_));
    for (int index = 0; index < count; ++index) {
      if (target[index] == 1)
        bottom_diff[index] *= pos_scale;
      else
        bottom_diff[index] *= neg_scale;
    }
  }
  ++prop_skip_count_;
  if (prop_skip_count_ >= prop_skip_period_)
    prop_skip_count_ = 0;
}

INSTANTIATE_CLASS(BalancedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(BalancedSigmoidCrossEntropyLoss);

}  // namespace caffe
