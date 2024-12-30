// Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace sparse_end2end {
namespace preprocessor {

TEST(Eigen3rdPartyUnitTest, EigenIncludeTest) {
  Eigen::Matrix<float, 2, 2> matrix;
  matrix << 1.0F, 2.0F, 3.0F, 4.0F;
  const float* matrix2 = matrix.data();
  EXPECT_FLOAT_EQ(matrix2[0], matrix(0, 0));
  EXPECT_FLOAT_EQ(matrix2[1], matrix(1, 0));
  EXPECT_FLOAT_EQ(matrix2[2], matrix(0, 1));
  EXPECT_FLOAT_EQ(matrix2[3], matrix(1, 1));
}

}  // namespace preprocessor
}  // namespace sparse_end2end