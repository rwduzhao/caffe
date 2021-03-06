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
#include "caffe/util/io_extra.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/layers/self_gated_layer.hpp"

namespace caffe {

template <typename Dtype>
void SummarizeArrayRows(const Dtype* data, const int dim,
                        const int row_start, const int row_span,
                        const int col_start, const int col_span) {
  for (int row = row_start; row < row_start + row_span; ++row) {
    Dtype row_mean = 0.0;
    Dtype row_min = Dtype(+1E10);
    Dtype row_max = Dtype(-1E10);
    for (int col = col_start; col < col_start + col_span; ++col) {
      const Dtype row_value = data[row * dim + col];
      row_mean += row_value;
      if (row_value < row_min)
        row_min = row_value;
      if (row_value > row_max)
        row_max = row_value;
    }
    row_mean /= Dtype(col_span);
    printf("(%g,%g,%g): ", row_min, row_mean, row_max);

    for (int col = col_start; col < col_start + col_span; ++col) {
      printf("%g", data[row * dim + col]);
      if (col < col_start + col_span - 1)
        printf(" ");
      else
        printf("\n");
    }
  }
}

template <typename Dtype>
void SummarizeArrayRows(const Dtype* data, const Dtype* gates, const int dim,
                        const int row_start, const int row_span,
                        const int col_start, const int col_span) {
  for (int row = row_start; row < row_start + row_span; ++row) {
    printf("%g", caffe_cpu_asum(dim, data + row * dim) / Dtype(dim));
    printf("(%g): ", caffe_cpu_asum(dim, gates + row * dim) / Dtype(dim));
    for (int col = col_start; col < col_start + col_span; ++col) {
      printf("%g(%g)", data[row * dim + col], gates[row * dim + col]);
      if (col < col_start + col_span - 1)
        printf(" ");
      else
        printf("\n");
    }
  }
}

template <typename Dtype>
void SummarizeArrayCols(const Dtype* data, const int dim,
                        const int row_start, const int row_span,
                        const int col_start, const int col_span) {
  for (int col = col_start; col < col_start + col_span; ++col) {
    Dtype col_mean = 0.0;
    Dtype col_min = Dtype(+1E10);
    Dtype col_max = Dtype(-1E10);
    for (int row = row_start; row < row_start + row_span; ++row) {
      const Dtype col_value = data[row * dim + col];
      col_mean += col_value;
      if (col_value < col_min)
        col_min = col_value;
      if (col_value > col_max)
        col_max = col_value;
    }
    col_mean /= Dtype(row_span);
    printf("(%g,%g,%g) ", col_min, col_mean, col_max);
  }
  printf("\n");
}

template <typename Dtype>
void PrintDataToFile(const char *filename, const Dtype* data, const int dim,
                     const int row_start, const int row_span,
                     const int col_start, const int col_span) {
  FILE *fid = fopen(filename, "w");
  for (int row = row_start; row < row_start + row_span; ++row) {
    for (int col = col_start; col < col_start + col_span; ++col) {
      fprintf(fid, "%g", data[row * dim + col]);
      if (col < col_start + col_span - 1)
        fprintf(fid, " ");
      else
        fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

template <typename Dtype>
__global__ void self_gated_layer_kernel_row_sum(
  const int num_row, const int dim, const Dtype* data, Dtype* results) {

  CUDA_KERNEL_LOOP(row_id, num_row) {
    Dtype result = 0;
    for (int d = 0; d < dim; ++d)
      result += data[row_id * dim + d];
    results[row_id] = result;
  }
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  // gate net input -> gate net output
  for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
    const int gate_net_layer_input_dim = gate_net_layer_input_dims_[gate_net_layer_id];
    const int gate_net_layer_output_dim = gate_net_layer_output_dims_[gate_net_layer_id];

    // gate net input weight -> gate_net_pre_top
    const Dtype *gate_net_bottom_data = gate_net_layer_id == 0 ? bottom[1]->gpu_data() : gate_net_tops_[gate_net_layer_id - 1]->gpu_data();
    const int weight_blob_id = gate_net_layer_id * 2 + 0;
    const Dtype *weight_data = this->blobs_[weight_blob_id]->gpu_data();
    Dtype *gate_net_pre_top_data = gate_net_pre_tops_[gate_net_layer_id]->mutable_gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_, gate_net_layer_output_dim, gate_net_layer_input_dim,
                   Dtype(1.), gate_net_bottom_data, weight_data, Dtype(0.), gate_net_pre_top_data);
    // gate net input bias -> gate_net_pre_top
    const Dtype *bias_multiplier_data = bias_multiplier_.gpu_data();
    const int bias_blob_id = gate_net_layer_id * 2 + 1;
    const Dtype *bias_data = this->blobs_[bias_blob_id]->gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, gate_net_layer_output_dim, 1,
                   Dtype(1.), bias_multiplier_data, bias_data, Dtype(1.), gate_net_pre_top_data);

    // gate_net_pre_top -> gate_net_top
    Dtype *gate_net_top_data = gate_net_tops_[gate_net_layer_id]->mutable_gpu_data();
    const int gate_net_top_count = gate_net_tops_[gate_net_layer_id]->count();
    caffe_gpu_sigmoid(gate_net_top_count, gate_net_pre_top_data, gate_net_top_data);
  }

  // gate_net_top -> gate
  if (is_last_unit_gated_) {
    const Dtype *last_gate_net_top_data = gate_net_tops_[gate_net_tops_.size() - 1]->gpu_data();
    const Dtype *last_gate_net_top_multiplier_data = last_gate_net_top_multiplier_.gpu_data();
    Dtype *gate_data = gate_.mutable_gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, time_step_ * batch_size_ * num_gate_, output_dim_, 1,
                   Dtype(1.), last_gate_net_top_data, last_gate_net_top_multiplier_data, Dtype(0.), gate_data);
  }

  // input -> output
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *gate_data = gate_.gpu_data();
  Dtype *top_data = top_.mutable_gpu_data();
  const int top_count = top_.count();
  caffe_gpu_mul(top_count, gate_data, bottom_data, top_data);

  if (false) {
    LOG(INFO) << "data input data:";
    PrintDataToFile("input_data.txt", bottom[0]->cpu_data(), bottom[0]->channels(), 0, 1, 0, bottom[0]->channels());
    LOG(INFO) << "gate input data:";
    PrintDataToFile("gate_input_data.txt", bottom[1]->cpu_data(), bottom[1]->channels(), 0, 1, 0, bottom[1]->channels());
    LOG(INFO) << "top data:";
    PrintDataToFile("top_data.txt", top[0]->cpu_data(), top[0]->channels(), 0, 1, 0, top[0]->channels());
    LOG(INFO) << "gate data:";
    PrintDataToFile("gate_data.txt", gate_.cpu_data(), gate_.width(), 0, gate_.count() / gate_.width(), 0, gate_.width());

    for (int gate_net_layer_id = 0; gate_net_layer_id < num_gate_net_layer_; ++gate_net_layer_id) {
      char weight_filename[100];
      sprintf(weight_filename, "weight_%d.txt", gate_net_layer_id);
      LOG(INFO) << weight_filename;
      const int weight_blob_id = 2 * gate_net_layer_id + 0;
      PrintDataToFile(weight_filename, this->blobs_[weight_blob_id]->cpu_data(),
                      this->blobs_[weight_blob_id]->channels(),
                      0, this->blobs_[weight_blob_id]->num(),
                      0, this->blobs_[weight_blob_id]->channels());
      char bias_filename[100];
      sprintf(bias_filename, "bias_%d.txt", gate_net_layer_id);
      LOG(INFO) << bias_filename;
      const int bias_blob_id = 2 * gate_net_layer_id + 1;
      PrintDataToFile(bias_filename, this->blobs_[bias_blob_id]->cpu_data(),
                      this->blobs_[bias_blob_id]->num(),
                      0, 1, 0, this->blobs_[bias_blob_id]->num());
    }

    LOG(INFO) << "gate col info:";
    SummarizeArrayCols(gate_.cpu_data(), gate_.width(), 0, gate_.count() / gate_.width(), 0, gate_.width());
    exit(0);
  }
}

template <typename Dtype>
void SelfGatedLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom) {

  // top_diff -> gate_diff
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *top_diff = top_.gpu_diff();
  Dtype *gate_diff = gate_.mutable_gpu_diff();
  const int gate_count = gate_.count();
  caffe_gpu_mul(gate_count, bottom_data, top_diff, gate_diff);

  // gate_diff -> last gate_net_top_diff
  if (is_last_unit_gated_) {
    const Dtype *gate_diff = gate_.gpu_diff();
    Dtype *last_gate_net_top_diff = gate_net_tops_[gate_net_tops_.size() - 1]->mutable_gpu_diff();
    self_gated_layer_kernel_row_sum<Dtype><<<CAFFE_GET_BLOCKS(time_step_ * batch_size_ * num_gate_), CAFFE_CUDA_NUM_THREADS>>>(
      time_step_ * batch_size_ * num_gate_, output_dim_, gate_diff, last_gate_net_top_diff);
  }

  for (int gate_net_layer_id = num_gate_net_layer_ - 1; gate_net_layer_id >= 0; --gate_net_layer_id) {
    const int gate_net_layer_input_dim = gate_net_layer_input_dims_[gate_net_layer_id];
    const int gate_net_layer_output_dim = gate_net_layer_output_dims_[gate_net_layer_id];

    // gate_net_top_diff -> gate_net_pre_top_diff
    const Dtype *gate_net_top_data = gate_net_tops_[gate_net_layer_id]->gpu_data();
    const Dtype *gate_net_top_diff = gate_net_tops_[gate_net_layer_id]->gpu_diff();
    Dtype *gate_net_pre_top_diff = gate_net_pre_tops_[gate_net_layer_id]->mutable_gpu_diff();
    const int gate_net_pre_top_count = gate_net_pre_tops_[gate_net_layer_id]->count();
    caffe_gpu_sigmoid_diff(gate_net_pre_top_count, gate_net_top_data, gate_net_top_diff, gate_net_pre_top_diff);
    if (clipping_threshold_ > 0.0f)
      caffe_gpu_bound(gate_net_pre_top_count, gate_net_pre_top_diff, -clipping_threshold_, clipping_threshold_, gate_net_pre_top_diff);

    // gradient w.r.t. gate net weight
    const int weight_blob_id = gate_net_layer_id * 2 + 0;
    Dtype *weight_diff = this->blobs_[weight_blob_id]->mutable_gpu_diff();
    const int bottom_id = 1;
    const Dtype *gate_net_bottom_data = gate_net_layer_id > 0 ? gate_net_tops_[gate_net_layer_id - 1]->gpu_data() : bottom[bottom_id]->gpu_data();
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, gate_net_layer_output_dim, gate_net_layer_input_dim, time_step_ * batch_size_,
                   Dtype(1.), gate_net_pre_top_diff, gate_net_bottom_data, Dtype(0.), weight_diff);
    // gradient w.r.t. gate net bias
    const Dtype *bias_multiplier_data = bias_multiplier_.gpu_data();
    const int bias_blob_id = gate_net_layer_id * 2 + 1;
    Dtype *bias_diff = this->blobs_[bias_blob_id]->mutable_gpu_diff();
    caffe_gpu_gemv(CblasTrans, time_step_ * batch_size_, gate_net_layer_output_dim,
                   Dtype(1.), gate_net_pre_top_diff, bias_multiplier_data, Dtype(0.), bias_diff);

    // gradient w.r.t. gate net bottom data
    if (gate_net_layer_id > 0 || propagate_down[bottom_id]) {
      const Dtype *weight_data = this->blobs_[weight_blob_id]->gpu_data();
      Dtype *gate_net_bottom_diff = gate_net_layer_id > 0 ? gate_net_tops_[gate_net_layer_id - 1]->mutable_gpu_diff() : bottom[bottom_id]->mutable_gpu_diff();
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, time_step_ * batch_size_, gate_net_layer_input_dim, gate_net_layer_output_dim,
                     Dtype(1.), gate_net_pre_top_diff, weight_data, Dtype(0.), gate_net_bottom_diff);
    }
  }
  // gradient w.r.t. bottom data
  const int bottom_id = 0;
  if (propagate_down[bottom_id]) {
    const Dtype *gate_data = gate_.gpu_data();
    const Dtype *top_diff = top_.gpu_diff();
    Dtype *bottom_diff = bottom[bottom_id]->mutable_gpu_diff();
    const int bottom_count = bottom[bottom_id]->count();
    caffe_gpu_mul(bottom_count, gate_data, top_diff, bottom_diff);
  }

  // clear extra bottom diff
  for (int bottom_id = 2; bottom_id < bottom.size(); ++bottom_id) {
    Dtype *bottom_diff = bottom[bottom_id]->mutable_gpu_diff();
    const int bottom_count = bottom[bottom_id]->count();
    caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
  }

  if (false) {
    LOG(INFO) << "display gradients";
    LOG(INFO) << "top diff amount: " << caffe_cpu_asum(top_.count(), top_.cpu_diff());
    LOG(INFO) << "gate diff amount: " << caffe_cpu_asum(gate_.count(), gate_.cpu_diff());
    if (is_last_unit_gated_)
      LOG(INFO) << "last gate net top diff amount: " << caffe_cpu_asum(gate_net_tops_[gate_net_tops_.size() - 1]->count(), gate_net_tops_[gate_net_tops_.size() - 1]->cpu_diff());

    for (int gate_net_layer_id = num_gate_net_layer_ - 1; gate_net_layer_id >= 0; --gate_net_layer_id) {
      const int gate_net_layer_input_dim = gate_net_layer_input_dims_[gate_net_layer_id];
      const int gate_net_layer_output_dim = gate_net_layer_output_dims_[gate_net_layer_id];

      const int weight_blob_id = gate_net_layer_id * 2 + 0;
      LOG(INFO) << "gate net weight[" << gate_net_layer_id << "] diff amount: " << caffe_cpu_asum(this->blobs_[weight_blob_id]->count(), this->blobs_[weight_blob_id]->cpu_diff());
      const int bias_blob_id = gate_net_layer_id * 2 + 1;
      LOG(INFO) << "gate net   bias[" << gate_net_layer_id << "] diff amount: " << caffe_cpu_asum(this->blobs_[bias_blob_id]->count(), this->blobs_[bias_blob_id]->cpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SelfGatedLayer);

}  // namespace caffe
