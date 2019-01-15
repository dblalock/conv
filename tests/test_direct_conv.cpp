
#include <gtest/gtest.h>
#include "../src/eigen/Eigen/Core"
#include "../src/eigen/unsupported/Eigen/CXX11/Tensor"

#include "../src/direct_conv.hpp"

template<class DataT>
using Ar2D = Eigen::Tensor<DataT, 2, Eigen::RowMajor>;

TEST(DirectConv, DidItWork) {
    auto x = 5;
    EXPECT_EQ(x, 5);
    EXPECT_GT(x, 4);
}

TEST(Sanity, EigenTensor) {
    Eigen::Tensor<float, 2> X(2, 3);
    EXPECT_EQ(X.size(), 6);
    EXPECT_EQ(X.dimensions()[0], 2);
    EXPECT_EQ(X.dimensions()[1], 3);
}

// class NaiveConvTest : public ::testing::Test {

//     NaiveConvTest() {

//     }

//     auto nrows = 2;
//     auto ncols = 3;
//     Ar2D<float> X(nrows, ncols);
//     Ar2D<float> filt(1, 1);
//     Ar2D<float> out(nrows, ncols);
//     Ar2D<float> ans(nrows, ncols);
// };



TEST(DirectConv, X_2x3_rowmajor) {
    auto nrows = 2;
    auto ncols = 3;
    Ar2D<float> X(nrows, ncols);

    { // new scope so I can pretend this is Catch instead of Gtest...
        Ar2D<float> filt(1, 1);
        Ar2D<float> out(nrows, ncols);
        Ar2D<float> ans(nrows, ncols);
        for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
        filt(0, 0) = 2;
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                ans(i, j) = X(i, j) * 2;
            }
        }

        conv2dx2d_valid_rowmajor(X.data(), nrows, ncols,
            filt.data(), 1, 1, out.data());

        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                EXPECT_EQ(out(i, j), ans(i, j));
            }
        }
    }
    {
        Ar2D<float> filt(2, 2);
        auto nrow_positions = nrows - filt.dimensions()[0] + 1;
        auto ncol_positions = ncols - filt.dimensions()[1] + 1;
        Ar2D<float> out(nrow_positions, ncol_positions);
        Ar2D<float> ans(nrow_positions, ncol_positions);
        for (int i = 0; i < X.size(); i++) { X.data()[i] = i; }
        filt(0, 0) = 2;
        filt(0, 1) = 1;
        filt(1, 0) = 0;
        filt(1, 1) = 3;
        ans(0, 0) = X(0, 0)*filt(0, 0) + X(0, 1)*filt(0, 1) +
            X(1, 0)*filt(1, 0) + X(1, 1)*filt(1, 1);
        ans(0, 1) = X(0, 1)*filt(0, 0) + X(0, 2)*filt(0, 1) +
            X(1, 1)*filt(1, 0) + X(1, 2)*filt(1, 1);

        conv2dx2d_valid_rowmajor(X.data(), nrows, ncols,
            filt.data(), 2, 2, out.data());

        for (int i = 0; i < nrow_positions; i++) {
            for (int j = 0; j < ncol_positions; j++) {
                EXPECT_EQ(out(i, j), ans(i, j)) << "i, j = " << i << ", " << j;
            }
        }
    }
}
