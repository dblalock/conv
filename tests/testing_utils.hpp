//
//  testing_utils.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef testing_utils_hpp
#define testing_utils_hpp

#include <stdio.h>

#include "../src/eigen/Eigen/Core"
#include "../src/eigen/unsupported/Eigen/CXX11/Tensor"

template<class DataT> using Ar2D = Eigen::Tensor<DataT, 2, Eigen::RowMajor>;
template<class DataT> using Ar3D = Eigen::Tensor<DataT, 3, Eigen::RowMajor>;
template<class DataT> using Ar4D = Eigen::Tensor<DataT, 4, Eigen::RowMajor>;
template<class DataT> using Ar5D = Eigen::Tensor<DataT, 5, Eigen::RowMajor>;

// what I should do is figure out how to slice this such that it returns
// a tensor of lower rank and implement one recursive function; but the
// eigen tensor API sucks, so longer, hackier code it is
template<class EigenTensorT>
static const void print_tensor5(
    const EigenTensorT& tensor, const char* name="")
{
    if (strlen(name)) { printf("%s:\n", name); }

    auto shape = tensor.dimensions();

    // for (int n = 1; n < 2; n++) {
    for (int n = 0; n < shape[0]; n++) {
        printf("[");
        for (int c = 0; c < shape[1]; c++) {
            if (c > 0) { printf(" "); }
            printf("[");
            for (int i = 0; i < shape[2]; i++) {
                if (i > 0) { printf("\t"); }
                printf("[");
                for (int j = 0; j < shape[3]; j++) {
                    printf("[ ");
                    for (int k = 0; k < shape[4]; k++) {
                        std::cout << tensor(n, c, i, j, k);
                        if (k < shape[4] - 1) { printf(", "); }
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("]");
            if (c < shape[1] - 1) { printf("\n"); }
        }
        printf("]\n");
        if (n < shape[0] - 1) { printf("\n"); }
    }
}


template<class EigenTensorT>
static const void print_tensor4(
    const EigenTensorT& tensor, const char* name="")
{
    if (strlen(name)) { printf("%s:\n", name); }

    auto shape = tensor.dimensions();

    // for (int n = 1; n < 2; n++) {
    for (int n = 0; n < shape[0]; n++) {
        printf("[");
        for (int c = 0; c < shape[1]; c++) {
            if (c > 0) { printf(" "); }
            printf("[");
            for (int i = 0; i < shape[2]; i++) {
                printf("[ ");
                for (int j = 0; j < shape[3]; j++) {
                    std::cout << tensor(n, c, i, j);
                    if (j < shape[3] - 1) { printf(", "); }
                }
                printf("]");
            }
            printf("]");
            if (c < shape[1] - 1) { printf("\n"); }
        }
        printf("]\n");
        if (n < shape[0] - 1) { printf("\n"); }
    }
}

#endif // testing_utils_hpp
