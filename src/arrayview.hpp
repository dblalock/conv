//
//  arrayview2.hpp
//  Arr
//
//  Created by DB on 11/21/18.
//  Copyright © 2018 D Blalock. All rights reserved.
//

#ifndef arrayview2_h
#define arrayview2_h

#include <array>
#include <assert.h>

//#include "bint.hpp"
#include "macros.hpp" // for SFINAE macros

namespace ar {

// struct formats {
//     enum {
//         Scalar,
//         RowMajor1D,
//         RowMajor2D,
//         RowMajor3D,
//         RowMajor4D,
//         RowMajor5D,
//     }
// };

namespace format {
    struct Scalar {};
    template<int rank> struct RowMajor {};
    using RowMajor1D = RowMajor<1>;
    using RowMajor2D = RowMajor<2>;
    using RowMajor3D = RowMajor<3>;
    using RowMajor4D = RowMajor<4>;
    using RowMajor5D = RowMajor<5>;
    using RowMajor6D = RowMajor<6>;

    namespace internal {
        struct BaseDense {};
        struct BaseRowMajor {};
    }

    // TODO add colmajor
}

namespace storageOrder {
    enum { RowMajor = 0, ColMajor = 1};
};

using DefaultIndexType = int32_t;


//
// TODO make axis_traits a thing so we can handle (partially) static shapes
//
// template<class AxisT, int... Args> struct axis_traits {
//     static const int static_size = -1;
//     static const int min_size = -1;
//     static const int max_size = -1;
// };

// be default, look for a member named traits in FmtT; our strategy for getting
// composability is to have a static setters for each valid attribute in format
// traits, and to let types define their own traits by starting with a few
// default sets of traits and customizing them from there
// template<template<int...> class FmtT, int ...Args> struct format_traits {
template<class FmtT> struct format_traits {
private:
    // using fullType = FmtT<Args...>;
    // using traits = typename fullType::traits;
    using traits = typename FmtT::traits;
public:
    using idx_t = typename traits::idx_t;
    static const int rank = traits::rank;
    static const int is_dense = traits::is_dense;
    static const int is_contig = traits::is_contig;
    static const int order = traits::order;
};

// template<> struct format_traits<format::internal::BaseRowMajor> {
template<> struct format_traits<format::internal::BaseDense> {
    using idx_t = DefaultIndexType;
    static const int is_dense = 1;
    static const int is_contig = 1; // when allocated, but may not stay true...
};
using BaseDenseTraits = format_traits<format::internal::BaseDense>;

template<> struct format_traits<format::internal::BaseRowMajor> {
    static const int order = storageOrder::RowMajor;
};
using BaseRowMajorTraits = format_traits<format::internal::BaseRowMajor>;

// template<int Rank> format_traits<format::RowMajor<rank> > : BaseDenseTraits, BaseRowMajorTraits {
//     static const int rank = Rank;
//     static const int order = storageOrder::RowMajor;
// }

template<> struct format_traits<format::Scalar> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 0;
};
template<> struct format_traits<format::RowMajor1D> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 1;
};
template<> struct format_traits<format::RowMajor2D> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 2;
};
template<> struct format_traits<format::RowMajor3D> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 3;
};
template<> struct format_traits<format::RowMajor4D> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 4;
};
template<> struct format_traits<format::RowMajor5D> : BaseDenseTraits, BaseRowMajorTraits{
    static const int rank = 5;
};
template<> struct format_traits<format::RowMajor6D> : BaseDenseTraits, BaseRowMajorTraits {
    static const int rank = 6;
};


// use these to derive new formats as needed
// TODO maybe have setters return new formats, instead of traits?
template<class traits, class IdxT> struct setIdxT {
    struct type : traits { using idx_t = IdxT; };
};
template<class traits, int IsDense> struct setDense {
    struct type : traits { static const int is_dense = IsDense; };
};
template<class traits, int IsContig> struct setContig {
    struct type : traits { static const int is_contig = IsContig; };
};
template<class traits, int Order> struct setOrder {
    struct type : traits { static const int order = Order; };
};
template<class Traits> struct as_format { using traits = Traits; };


// ------------------------------------------------ strides for various axes

template<int rank, int order, class IdxT=DefaultIndexType>
std::array<IdxT, rank> default_strides_for_shape(
    std::array<IdxT, rank> shape)
{
    // static const int rank = Format::rank;
    // static const int order = Format::order;
    static_assert(rank >= 0, "Rank must be >= 0!");
    // static_assert(rank <= 2 || AxesT::is_contig, "Only dense axes can use default strides!");
    static_assert((order == storageOrder::RowMajor ||
            order == storageOrder::ColMajor),
        "Only StorageOrders RowMajor and ColMajor supported!");

    std::array<IdxT, rank> strides{0};
    if (rank == 1) {
        strides[0] = 1;
        return strides;
    }
    if (rank == 2) {
        if (order == storageOrder::ColMajor) { // colmajor
            strides[0] = 1;
            strides[1] = shape[0];
        } else { // rowmajor
            strides[0] = shape[1];
            strides[1] = 1;
        }
        return strides;
    }
    switch(order) { // rank 3+ if we got to here
        case storageOrder::RowMajor:
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--) {
                strides[i] = shape[i + 1] * strides[i + 1];
            }
            break;
        case storageOrder::ColMajor:
            strides[0] = 1;
            for (int i = 1; i < rank; i++) {
                strides[i] = shape[i - 1] * strides[i - 1];
            }
            break;
        default:
            assert("Somehow got unrecognized storage order!");
            break; // can't happen
    }
    return strides;
}

// ------------------------------------------------ ArrayView

template<class DataT, class Format=format::RowMajor1D>
struct ArrayView {
    using fmt_traits = format_traits<Format>;
    static const int rank = fmt_traits::rank;
    static const int order = fmt_traits::order;
    using IdxT = typename fmt_traits::idx_t;
    using strides_t = std::array<IdxT, rank>;
    using shape_t = std::array<IdxT, rank>;
    using idxs_t = strides_t;

    ArrayView(DataT *const data, const shape_t& shape):
        _data(data),
        _shape(shape),
        _strides(default_strides_for_shape<rank, order>(_shape))
    {
        // printf("I'm an ArrayView and I have rank %d!\n", rank);
    };

    IdxT flatten_idxs(const idxs_t& idxs) {
        IdxT idx = 0;
        for (int i = 0; i < rank; i++) {
            idx += idxs[i] * _strides[i];
        }
        return idx;
    }

    DataT& operator[](const idxs_t& idxs) {
        return _data[flatten_idxs(idxs)];
    }

    const shape_t& shape() const { return _shape; }
    const shape_t& strides() const { return _strides; }
    const IdxT size() const {
        IdxT sz = 1;
        for (int i = 0; i < rank; i++) {
            sz *= _shape[i];
        }
        return sz;
    }

    void setValue(DataT val) {
        static_assert(fmt_traits::is_contig,
            "setValue() only implemented for dense arrayviews!");
        if (fmt_traits::is_contig) {
            // use memset if it will preserve correctness
            if (sizeof(val) == 1) {
                memset(_data, val, size());
                // DataT converted = ((DataT)((unsigned char)val));
                // if (converted == val) {
                //     memset(_data, val, sizeof(DataT)*size());
                // }
            } else {
                for (IdxT i = 0; i < size(); i++) {
                    _data[i] = val;
                }
            }
        }
    }

    void setZero() {
        // setValue(0);
        static_assert(fmt_traits::is_contig,
            "setZero() only implemented for contiguous arrayviews!");
        if (fmt_traits::is_contig) {
            memset(_data, 0, sizeof(DataT)*size());
        }
    }

private:
    DataT *const _data;
    const shape_t _shape;
    const strides_t _strides;
};

// template<class DataT, int Rank>
// struct RowMajorView : public ar::ArrayView<DataT, ar::format::RowMajor<Rank> > {};
template<class DataT, int Rank>
using RowMajorView = ar::ArrayView<DataT, ar::format::RowMajor<Rank> >;

template<class DataT>
auto make_view(DataT* data, int32_t idx0) {
    return RowMajorView<DataT, 1>(data, {idx0});
}
template<class DataT>
auto make_view(DataT* data, int32_t idx0, int32_t idx1) {
    return RowMajorView<DataT, 2>(data, {idx0, idx1});
}
template<class DataT>
auto make_view(DataT* data, int32_t idx0, int32_t idx1, int32_t idx2) {
    return RowMajorView<DataT, 3>(data, {idx0, idx1, idx2});
}
template<class DataT>
auto make_view(DataT* data, int32_t idx0, int32_t idx1, int32_t idx2,
    int32_t idx3)
{
    return RowMajorView<DataT, 4>(data, {idx0, idx1, idx2, idx3});
}
template<class DataT>
auto make_view(DataT* data, int32_t idx0, int32_t idx1, int32_t idx2,
    int32_t idx3, int32_t idx4)
{
    return RowMajorView<DataT, 5>(data, {idx0, idx1, idx2, idx3, idx4});
}
template<class DataT>
auto make_view(DataT* data, int32_t idx0, int32_t idx1, int32_t idx2,
    int32_t idx3, int32_t idx4, int32_t idx5)
{
    return RowMajorView<DataT, 6>(data, {idx0, idx1, idx2, idx3, idx4, idx5});
}



} // ar

#endif // arrayview2_h
