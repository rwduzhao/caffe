/*=============================================================================
#     FileName: gate_layer.cu
#   Desciption: gate layer
#       Author: rwduzhao
#        Email: rw.du.zhao@gmail.com
#     HomePage: rw.du.zhao@gmail.com
#      Version: 0.0.1
#   LastChange: 2015-10-26 21:13:54
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
#include "caffe/layers/gate_layer.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

  if (is_unit_gated_) {  // gate input data -> gate
    const Dtype *gate_input_data = bottom[1]->gpu_data();
    const Dtype *gate_dim_multiplier_data = gate_dim_multiplier_.gpu_data();
    Dtype *gate_data = gate_.mutable_gpu_data();
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_, 1,
                   Dtype(1.), gate_input_data, gate_dim_multiplier_data, Dtype(0.), gate_data);
    if (false) {
      const Dtype *gate_input_cpu_data = bottom[1]->cpu_data();
      printf("gv ->");
      const int max_num_print = 5;
      for (int id = 0; id < std::min(bottom[1]->count(), max_num_print); ++id)
        fprintf(stdout, " %0.3f", gate_input_cpu_data[id]);
      fprintf(stdout, " ...");
      for (int id = std::max(0, bottom[1]->count() - max_num_print); id < bottom[1]->count(); ++id)
        fprintf(stdout, " %0.3f", gate_input_cpu_data[id]);
      fprintf(stdout, "\n");
    }
  }

  // input -> output
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *gate_data = gate_.gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();
  caffe_gpu_mul(top_count, gate_data, bottom_data, top_data);
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype> *> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype> *> &bottom) {

  // top_diff -> gate_diff
  const Dtype *bottom_data = bottom[0]->gpu_data();
  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype *gate_diff = gate_.mutable_gpu_diff();
  const int gate_count = gate_.count();
  caffe_gpu_mul(gate_count, bottom_data, top_diff, gate_diff);
  if (is_unit_gated_) {  // gate_diff -> gate input diff
    const Dtype *gate_diff = gate_.gpu_diff();
    Dtype *gate_input_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_row_sum(batch_size_, output_dim_, gate_diff, gate_input_diff);
    // caffe_gpu_scale(bottom[1]->count(), Dtype(1.) / Dtype(output_dim_), bottom[1]->gpu_diff(), bottom[1]->mutable_gpu_diff());
  }

  // top_diff -> bottom_diff
  const int bottom_id = 0;
  if (propagate_down[bottom_id]) {
    const Dtype *gate_data = gate_.gpu_data();
    const Dtype *top_diff = top[0]->gpu_diff();
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
}

INSTANTIATE_LAYER_GPU_FUNCS(GateLayer);

}  // namespace caffe
