#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_pool_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void PrintBlobData(Blob<Dtype>* blob) {
  const Dtype* data = blob->cpu_data();
  for (int n = 0; n < blob->num(); ++n) {
    for (int c = 0; c < blob->channels(); ++c) {
      for (int h = 0; h < blob->height(); ++h) {
        for (int w = 0; w < blob->width(); ++w) {
          Dtype value = data[blob->offset(n, c, h, w)];
          fprintf(stdout, "%.1f ", value);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
  }
}

template <typename Dtype>
void PrintBlobDiff(Blob<Dtype>* blob) {
  const Dtype* data = blob->cpu_diff();
  for (int n = 0; n < blob->num(); ++n) {
    for (int c = 0; c < blob->channels(); ++c) {
      for (int h = 0; h < blob->height(); ++h) {
        for (int w = 0; w < blob->width(); ++w) {
          Dtype value = data[blob->offset(n, c, h, w)];
          fprintf(stdout, "%.1f ", value);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
  }
}

template <typename TypeParam>
class SpatialPoolLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  SpatialPoolLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_rois_(new Blob<Dtype>()),
        blob_bottom_imsz_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    Caffe::set_random_seed(1701);

    const int num_img = 2;
    const int num_roi = 20;
    const int dim = 5;

    this->blob_bottom_->Reshape(num_roi, dim, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    this->blob_bottom_rois_->Reshape(num_roi, 5, 1, 1);
    this->blob_bottom_imsz_->Reshape(num_img, 2, 1, 1);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_bottom_vec_.push_back(blob_bottom_imsz_);


    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SpatialPoolLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_rois_;
    delete blob_bottom_imsz_;
    delete blob_top_;
    delete blob_top_mask_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_bottom_imsz_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void InitSimpleData() {
    // blob_bottom_:
    //   [1  2  3  4
    //    5  6  7  8
    //    9 10 11 12]
    // blob_bottom_rois_:
    //   [0  0  0  9  9
    //    0 10 10 19 19
    //    1  5  5  9  9]
    // blob_bottom_imsz_:
    //   [20 20
    //    20 20]

    const int num_img = 2;
    const int num_roi = 3;
    const int dim = 4;

    this->blob_bottom_->Reshape(num_roi, dim, 1, 1);
    Dtype* bottom_data = blob_bottom_->mutable_cpu_data();
    Dtype value = Dtype(1.);
    for (int roi_id = 0; roi_id < num_roi; ++roi_id) {
      for (int dim_id = 0; dim_id < dim; ++dim_id) {
        bottom_data[blob_bottom_->offset(roi_id, dim_id, 0, 0)] = value;
        value += 1.;
      }
    }

    this->blob_bottom_rois_->Reshape(num_roi, 5, 1, 1);
    Dtype* roi_data = blob_bottom_rois_->mutable_cpu_data();

    roi_data[blob_bottom_rois_->offset(0, 0)] = 0.;
    roi_data[blob_bottom_rois_->offset(0, 1)] = 0.;
    roi_data[blob_bottom_rois_->offset(0, 2)] = 0.;
    roi_data[blob_bottom_rois_->offset(0, 3)] = 9.;
    roi_data[blob_bottom_rois_->offset(0, 4)] = 9.;

    roi_data[blob_bottom_rois_->offset(1, 0)] = 0.;
    roi_data[blob_bottom_rois_->offset(1, 1)] = 10.;
    roi_data[blob_bottom_rois_->offset(1, 2)] = 10.;
    roi_data[blob_bottom_rois_->offset(1, 3)] = 19.;
    roi_data[blob_bottom_rois_->offset(1, 4)] = 19.;

    roi_data[blob_bottom_rois_->offset(2, 0)] = 1.;
    roi_data[blob_bottom_rois_->offset(2, 1)] = 5.;
    roi_data[blob_bottom_rois_->offset(2, 2)] = 5.;
    roi_data[blob_bottom_rois_->offset(2, 3)] = 9.;
    roi_data[blob_bottom_rois_->offset(2, 4)] = 9.;

    this->blob_bottom_imsz_->Reshape(num_img, 2, 1, 1);
    Dtype* imsz_data = blob_bottom_imsz_->mutable_cpu_data();

    imsz_data[blob_bottom_imsz_->offset(0, 0)] = 20.;
    imsz_data[blob_bottom_imsz_->offset(0, 1)] = 20.;
    imsz_data[blob_bottom_imsz_->offset(1, 0)] = 20.;
    imsz_data[blob_bottom_imsz_->offset(1, 1)] = 20.;
  }

  void TestForward() {
    LayerParameter layer_param;
    ROIPoolingParameter* param = layer_param.mutable_roi_pooling_param();
    param->set_pooled_h(3);
    param->set_pooled_w(3);
    param->set_pool(ROIPoolingParameter_PoolMethod_MAX);
    InitSimpleData();
    SpatialPoolLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num(), blob_bottom_imsz_->num());
    EXPECT_EQ(this->blob_top_->channels(), blob_bottom_->count() / blob_bottom_->num());
    EXPECT_EQ(this->blob_top_->height(), 3);
    EXPECT_EQ(this->blob_top_->width(), 3);
    PrintBlobData(this->blob_top_);

    for (int id = 0; id < this->blob_top_->count(); ++id)
      this->blob_top_->mutable_cpu_diff()[id] = 1.;
    vector<bool> propagate_down;
    propagate_down.push_back(true);
    propagate_down.push_back(false);
    propagate_down.push_back(false);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);
    PrintBlobDiff(this->blob_bottom_);
  }

};

TYPED_TEST_CASE(SpatialPoolLayerTest, TestDtypesAndDevices);

TYPED_TEST(SpatialPoolLayerTest, TestForwardMax) {
  this->TestForward();
}

}  // namespace caffe
