//
//  test_catconv.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <gtest/gtest.h>
// #include "../src/eigen/Eigen/Core"
#include "../src/eigen/unsupported/Eigen/CXX11/Tensor"

#include "../src/catconv.hpp"

template<class DataT> using Ar2D = Eigen::Tensor<DataT, 2, Eigen::RowMajor>;
template<class DataT> using Ar3D = Eigen::Tensor<DataT, 3, Eigen::RowMajor>;

TEST(CatConv, SmokeTest) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}
