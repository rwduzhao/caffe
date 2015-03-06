#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

bool debug = true;

struct data_struct {
  int index;
  float value;
};

__device__
void swap(struct data_struct * array, int i, int j) {
  int tmp_index = array[i].index;
  float tmp_value = array[i].value;
  array[i].index = array[j].index;
  array[i].value = array[j].value;
  array[j].index = tmp_index;
  array[j].value = tmp_value;
}

__device__
void heap_adjust_descend_by_value(struct data_struct array[], int i, int array_length) {
  while (2 * i + 1 < array_length) {
    int child_index = 2 * i + 1;
    if (child_index < array_length - 1 && array[child_index + 1].value < array[child_index].value)
      ++child_index;
    if (array[i].value > array[child_index].value) {
      swap(array, i, child_index);
    } else
      break;
    i = child_index;
  }
}

__device__
void heap_sort_descend_by_value(struct data_struct array[], int length) {
  if (length == 1)
    return;
  for (int i = (length - 2) / 2; i >= 0; --i)
    heap_adjust_descend_by_value(array, i, length);
  for (int i = length - 1; i > 0; --i) {
    swap(array, i, 0);
    heap_adjust_descend_by_value(array, 0, i);
  }
}

__device__
void heap_adjust_ascend_by_value(struct data_struct array[], int i, int array_length) {
  while (2 * i + 1 < array_length) {
    int child_index = 2 * i + 1;
    if (child_index < array_length - 1 && array[child_index + 1].value > array[child_index].value)
      ++child_index;
    if (array[i].value < array[child_index].value) {
      swap(array, i, child_index);
    } else
      break;
    i = child_index;
  }
}

__device__
void heap_sort_ascend_by_value(struct data_struct array[], int length) {
  if (length == 1)
    return;
  for (int i = (length - 2) / 2; i >= 0; --i)
    heap_adjust_ascend_by_value(array, i, length);
  for (int i = length - 1; i > 0; --i) {
    swap(array, i, 0);
    heap_adjust_ascend_by_value(array, 0, i);
  }
}

__device__
void heap_adjust_ascend_by_index(struct data_struct array[], int i, int array_length) {
  int child_index = -1;
  for (; 2 * i + 1 < array_length; i = child_index) {
    child_index = 2*i + 1;
    if (child_index < array_length - 1 && array[child_index + 1].index > array[child_index].index)
      ++child_index;
    if (array[i].index < array[child_index].index) {
      swap(array, i, child_index);
    } else
      break;
  }
}

__device__
void heap_sort_ascend_by_index(struct data_struct array[], int length) {
  if (length == 1)
    return;
  for (int i = (length - 2) / 2; i >= 0; --i)
    heap_adjust_ascend_by_index(array, i, length);
  for (int i = length - 1; i > 0; --i) {
    swap(array, 0, i);
    heap_adjust_ascend_by_index(array, 0, i);
  }
}

__device__
void print_data_array(const struct data_struct * array, const int length) {
  for (int i = 0; i < length; ++i)
    printf("%d:%f ", array[i].index, array[i].value);
  printf("\n");
}

namespace caffe {

template <typename Dtype>
void print_blob_data(const Blob<Dtype> * blob, const Dtype * blob_data) {
  int data_index = 0;
  for (int n = 0; n < blob->num(); ++n) {
    for (int c = 0; c < blob->channels(); ++c) {
      std::cout << "num:" << n << ", channel:" << c << std::endl;
      for (int h = 0; h < blob->height(); ++h) {
        for (int w = 0; w < blob->width(); ++w) {
          std::cout << blob_data[data_index++];
          if (blob->width() != w + 1)
            std::cout << " ";
          else
            std::cout << std::endl;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
    int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void KMaxPoolForward(const int top_count, const Dtype* bottom_data,
                                const int bottom_num, const int bottom_channels,
                                const int bottom_height, const int bottom_width,
                                const int top_height, const int top_width,
                                const int kernel_h, const int kernel_w,
                                const int stride_h, const int stride_w,
                                const int pad_h, const int pad_w,
                                const int direction, const int top_k,
                                Dtype* top_data, int* mask, Dtype* top_mask) {
  int pooled_height = 0;
  int pooled_width = 0;
  if (direction == PoolingParameter_PoolDirection_HORIZONTAL) {
    pooled_height = top_height;
    pooled_width = top_width / top_k;
  } else if (direction == PoolingParameter_PoolDirection_VERTICAL) {
    pooled_height = top_height / top_k;
    pooled_width = top_width;
  }
  const int n_pooling = top_count/top_k;

  CUDA_KERNEL_LOOP(index, n_pooling) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % bottom_channels;
    int n = index / pooled_width / pooled_height / bottom_channels;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, bottom_height);
    int wend = min(wstart + kernel_w, bottom_width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    int data_array_length = (hend - hstart + 1) * (wend - wstart + 1);
    struct data_struct * data_array = (struct data_struct *)malloc(data_array_length * sizeof(struct data_struct));
    int data_array_index = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int position_offset = h * bottom_width + w;
        data_array[data_array_index].index = position_offset;
        data_array[data_array_index].value = bottom_data[position_offset];
        ++data_array_index;
      }
    }

    heap_sort_ascend_by_value(data_array, data_array_length);
    for (int data_array_index = 0; data_array_index < top_k; ++data_array_index) {
      if (data_array_index < data_array_length / 2)
        swap(data_array, data_array_index, data_array_length - data_array_index - 1);
      else
        break;
    }
    heap_sort_ascend_by_index(data_array, top_k);

    for (int data_array_index = 0; data_array_index < top_k; ++data_array_index) {
      int top_data_index = 0;
      if (direction == PoolingParameter_PoolDirection_HORIZONTAL)
        // top_data_index = (n * bottom_channels + c) * top_height * top_width + ph * top_width + pw * pooled_width + data_array_index;
        top_data_index = (n * bottom_channels + c) * top_height * top_width + ph * top_width + pw * top_k + data_array_index;
      else if (direction == PoolingParameter_PoolDirection_VERTICAL)
        // top_data_index = (n * bottom_channels + c) * top_height * top_width + (ph + data_array_index) * top_width + pw;
        top_data_index = (n * bottom_channels + c) * top_height * top_width + (ph * top_k + data_array_index) * top_width + pw;

      top_data[top_data_index] = data_array[data_array_index].value;
      if (mask)
        mask[top_data_index] = data_array[data_array_index].index;
      else
        top_mask[top_data_index] = data_array[data_array_index].index;
    }

    free(data_array);
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top->size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (Caffe::phase() == Caffe::TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  case PoolingParameter_PoolMethod_KMAX:
    if (use_top_mask) {
      top_mask = (*top)[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    KMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data,
      bottom[0]->num(), channels_,
      height_, width_,
      pooled_height_, pooled_width_,
      kernel_h_, kernel_w_,
      stride_h_, stride_w_,
      pad_h_, pad_w_,
      this->layer_param_.pooling_param().direction(), this->layer_param_.pooling_param().top_k(),
      top_data, mask, top_mask);
    if (debug) {
      std::cout << "forward phase" << std::endl;
      std::cout << "bottom data" << std::endl;
      print_blob_data(bottom[0], bottom[0]->cpu_data());
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void KMaxPoolBackward(const int bottom_count, const Dtype* top_diff,
                                 const int* mask, const Dtype* top_mask,
                                 const int num, const int channels,
                                 const int bottom_height, const int bottom_width,
                                 const int top_height, const int top_width,
                                 const int kernel_h, const int kernel_w,
                                 const int stride_h, const int stride_w,
                                 const int pad_h, const int pad_w,
                                 const int direction, const int top_k,
                                 Dtype* bottom_diff) {
  const int nthreads = bottom_count;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pooled_height = 0;
    int pooled_width = 0;
    if (direction == PoolingParameter_PoolDirection_HORIZONTAL) {
      pooled_height = top_height;
      pooled_width = top_width / top_k;
    } else if (direction == PoolingParameter_PoolDirection_VERTICAL) {
      pooled_height = top_height / top_k;
      pooled_width = top_width;
    }

    int w = index % bottom_width;
    int h = (index / bottom_width) % bottom_height;
    int c = (index / bottom_width / bottom_height) % channels;
    int n = index / bottom_width / bottom_height / channels;
    int phstart = (h + pad_h < kernel_h) ?
      0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart = (w + pad_w < kernel_w) ?
      0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    if (direction == PoolingParameter_PoolDirection_HORIZONTAL) {
      pwstart *= top_k;
      pwend *= top_k;
    } else if (direction == PoolingParameter_PoolDirection_VERTICAL) {
      phstart *= top_k;
      phend *= top_k;
    }

    Dtype gradient = 0;
    int offset = (n * channels + c) * top_height * top_width;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * top_width + pw] == h * bottom_width + w) {
            gradient += top_diff[ph * top_width + pw];
          }
        }
      }
    } else {
      top_mask += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * top_width + pw] == h * bottom_width + w) {
            gradient += top_diff[ph * top_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_KMAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    if (debug) {
      std::cout << "backward phase" << std::endl;
      std::cout << "top data" << std::endl;
      print_blob_data(top[0], top[0]->cpu_data());
      std::cout << "top mask" << std::endl;
      print_blob_data(&max_idx_, max_idx_.cpu_data());
      std::cout << "top diff" << std::endl;
      print_blob_data(top[0], top[0]->cpu_diff());
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    KMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff,
      mask, top_mask,
      top[0]->num(), channels_,
      height_, width_,
      pooled_height_, pooled_width_,
      kernel_h_, kernel_w_,
      stride_h_, stride_w_,
      pad_h_, pad_w_,
      this->layer_param_.pooling_param().direction(), this->layer_param_.pooling_param().top_k(),
      bottom_diff);
    if (debug) {
      std::cout << "bottom diff" << std::endl;
      print_blob_data((*bottom)[0], (*bottom)[0]->cpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(PoolingLayer);


}  // namespace caffe
