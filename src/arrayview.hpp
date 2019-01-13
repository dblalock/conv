//
//  array.hpp
//  Arr
//
//  Created by DB on 11/21/18.
//  Copyright © 2018 D Blalock. All rights reserved.
//

#ifndef arrayview_h
#define arrayview_h

#include <array>
#include <assert.h>

//#include "bint.hpp"
#include "macros.hpp" // for SFINAE macros

namespace ar {

//#include "array_utils.hpp" // for debug

// /** elements of tensor are contiguous along this ax with stride of 1;
//  * e.g., axis 1 in a row-major matrix
//  */
// class ContiguousAx {
//     static const bool is_contig = true;
//     static const bool is_dense = true;
//     static const bool is_strided = true;
// };
/** elements of tensor are not contiguous along this ax, but
 * sub-arrays are; e.g., axis 0 in a row-major matrix
 *
 * Dense along ax0:
 * x x x x
 * x x x x
 * x x x x
 *
 * Not dense along ax0, but still strided:
 *
 * x x x x
 *
 * x x x x
 *
 * x x x x
 *
 * Not strided (ie, with uniform stride) along ax0:
 *
 * x x x x
 * x x x x
 *
 * x x x x
 *
 * Dense along ax0, but only strided along ax1:
 *
 * x   x   x   x
 * x   x   x   x
 * x   x   x   x
 */
// template<int Used=true>
// class DenseAx {
//     static const int is_used = Used;
//     static const bool is_contig = false;
//     static const bool is_dense = true;
//     static const bool is_strided = true;
// };
// template<int Used=false> class StridedAx {
//     static const int is_used = Used;
//     static const bool is_contig = false;
//     static const bool is_dense = false;
//     static const bool is_strided = true;
// };
// // for stuff like where(); not sure how to support this
// template<int Used=false> class ChaosAx {
//     static const int is_used = Used;
//     static const bool is_contig = false;
//     static const bool is_dense = false;
//     static const bool is_strided = false;
// };

using DefaultIndexType = int32_t;

// =============================================================== Static Sizes

struct anySz {
    static const int is_valid = false;
    static const int min = 0;
    static const int max = 0;
};
using NoBounds = anySz; // AnySz is shorter in errors, but less clear in code

template<int Value=-1> struct ConstSize {
    static const int is_valid = Value > 0;
    static const int min = Value;
    static const int max = Value;
};

template<int StaticSize, typename = void>
struct getSizeBound { using type = NoBounds; };

template<int StaticSize>
struct getSizeBound<StaticSize, ENABLE_IF(StaticSize > 0)>
{
    using type = ConstSize<StaticSize>;
};

// ================================================================ Axis

struct AxisAttr {
    enum {
        Unused      = 0,
        Used        = 1 << 0,
        Strided     = 1 << 1,
        Dense       = 1 << 2,
        Contiguous  = 1 << 3,

        // these are for convenience when defining axis attributes, since
        // stronger contiguity traits imply less strong ones
        StridedMask = Strided,
        DenseMask = Strided | Dense,
        ContiguousMask = Strided | Dense | Contiguous
    };
};

template<int Attrs=AxisAttr::Unused, class SizeBounds=NoBounds>
struct Axis {
    static const int attrs = Attrs;
    static const int is_used = Attrs & AxisAttr::Used;
    static const bool is_strided = Attrs & AxisAttr::Strided;
    static const bool is_dense = Attrs & AxisAttr::Dense;
    static const bool is_contig = Attrs & AxisAttr::Contiguous;
    // static const int stride = StaticStride;
    using size_bounds = SizeBounds;
};

template<bool Contiguous=false, bool Dense=true, bool Strided=true,
    int Used=true, class SizeBounds=NoBounds>
    // int Used=true, int StaticStride=0, class SizeBounds=NoBounds>
struct make_axis {
    static const int attrs =
        (Contiguous ? AxisAttr::Contiguous : 0) |
        (Dense ? AxisAttr::Dense : 0) |
        (Strided ? AxisAttr::Strided : 0) |
        (Used ? AxisAttr::Used : 0);
    using type = Axis<attrs, SizeBounds>;
};

// ------------------------------------------------ axis aliases

using AxisContig = Axis<AxisAttr::ContiguousMask | AxisAttr::Used>;
using AxisDense = Axis<AxisAttr::DenseMask | AxisAttr::Used>;
using AxisStrided = Axis<AxisAttr::StridedMask | AxisAttr::Used>;
using AxisUnused = Axis<AxisAttr::Unused>;

// ------------------------------------------------ axis manipulation


template<class AxisT, bool Contig> struct setAxisContiguous {
    using type = typename make_axis<Contig, AxisT::is_dense, AxisT::is_strided,
        AxisT::is_used, typename AxisT::SizeBounds>::type;
};
template<class AxisT, bool Dense> struct setAxisDense {
    using type = typename make_axis<AxisT::is_contig, Dense, AxisT::is_strided,
        AxisT::is_used, typename AxisT::SizeBounds>::type;
};
template<class AxisT, bool Strided> struct setAxisStrided {
    using type = typename make_axis<AxisT::is_contig, AxisT::is_dense, Strided,
        AxisT::is_used, typename AxisT::SizeBounds>::type;
};
template<class AxisT, int Used> struct setAxisUsed {
    using type = typename make_axis<AxisT::is_contig, AxisT::is_dense,
        AxisT::is_strided, Used, typename AxisT::SizeBounds>::type;
};
// template<class AxisT, int Stride> struct setAxisStaticStride {
//     using type = Axis<AxisT::is_contig && (Stride == 1), AxisT::is_dense,
//     AxisT::is_strided, AxisT::is_used, Stride, typename AxisT::SizeBounds>;
// };
template<class AxisT, class SizeBounds> struct setAxisSizeBounds {
    static const int attrs = AxisT::attrs;
    using type = Axis<attrs, SizeBounds>;
};


template<class AxisT, int StaticSize> struct setAxisStaticSize {
    // using SizeBoundsT = ConstSize<StaticSize>;
    using SizeBoundsT = typename getSizeBound<StaticSize>::type;
    using type = typename setAxisSizeBounds<AxisT, SizeBoundsT>::type;
};


#define SET_AXIS_PROP(STRUCT_NAME, AXES_T, AX, VAL) \
    typename STRUCT_NAME<GET_AXIS_T(AXES_T, AX), VAL>::type

#define SET_AXIS_CONTIGUOUS(AXES_T, AX, BOOL) \
    SET_AXIS_PROP(setAxisContiguous, AXES_T, AX, BOOL)

#define SET_AXIS_DENSE(AXES_T, AX, BOOL) \
    SET_AXIS_PROP(setAxisDense, AXES_T, AX, BOOL)

#define SET_AXIS_STRIDED(AXES_T, AX, BOOL) \
    SET_AXIS_PROP(setAxisStrided, AXES_T, AX, BOOL)

#define SET_AXIS_USED(AXES_T, AX, BOOL) \
    SET_AXIS_PROP(setAxisUsed, AXES_T, AX, BOOL)
    // typename setAxisContiguous(GET_AXIS_T(AXES_T, AX), BOOL)::type

#define SET_AXIS_SIZE(AXES_T, AX, BOOL) \
    SET_AXIS_PROP(setAxisStaticSize, AXES_T, AX, BOOL)


// ============================================================= Storage Order

struct StorageOrders {
    // enum { Unspecified = -1, RowMajor = 0, ColMajor = 1, NCHW = 2};
    enum { Unspecified = 0, RowMajor = 1, ColMajor = 2, NCHW = 3};
};

// template<int Order> struct StorageOrder { };
// template<> struct StorageOrder<StorageOrders::RowMajor> {
//     static constexpr std::array<int, 4> order {3, 2, 1, 0};
// };

/** This is to get ND indices from cuda thread/block indices */
template<int Rank, int Order> struct idxs_from_flat_idx {};

template<int Order> struct idxs_from_flat_idx<1, Order> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        return idx;
    }
};

template<> struct idxs_from_flat_idx<2, StorageOrders::RowMajor> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        return {idx / shape[1], idx % shape[1]};
    }
};
template<> struct idxs_from_flat_idx<2, StorageOrders::ColMajor> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        return {idx % shape[0], idx / shape[0]};
    }
};

template<> struct idxs_from_flat_idx<3, StorageOrders::RowMajor> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        auto rowidx = idx / (shape[1] * shape[2]);
        auto flat_idx_into_row = idx % (shape[1] * shape[2]);
        auto colidx = (flat_idx_into_row / shape[2]);
        return {rowidx, colidx, idx % shape[2]};
    }
};
template<> struct idxs_from_flat_idx<3, StorageOrders::ColMajor> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        auto chanidx = idx / (shape[0] * shape[1]);
        auto flat_idx_into_channel = idx % (shape[0] * shape[1]);
        auto colidx = flat_idx_into_channel / shape[0];
        return {idx % shape[0], colidx, chanidx};
    }
};
template<> struct idxs_from_flat_idx<4, StorageOrders::RowMajor> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        auto one_sample_sz = shape[1] * shape[2] * shape[3];
        auto sampleidx = idx / one_sample_sz;
        auto idx_into_sample = idx % one_sample_sz;
        auto sample_idxs = idxs_from_flat_idx<3, StorageOrders::RowMajor>{}(
            ShapeT{shape[1], shape[1], shape[3]}, idx_into_sample);
        return {sampleidx, sample_idxs[0], sample_idxs[1], sample_idxs[2]};
    }
};
template<> struct idxs_from_flat_idx<4, StorageOrders::ColMajor> {
    // TODO impl this if needed
};
template<> struct idxs_from_flat_idx<4, StorageOrders::NCHW> {
    template<class ShapeT, class IdxT>
    ShapeT operator()(const ShapeT& shape, IdxT idx) {
        auto one_sample_sz = shape[1] * shape[2] * shape[3];
        auto sample_idx = idx / one_sample_sz;

        auto idx_into_sample = idx % one_sample_sz;
        auto one_channel_sz = shape[1] * shape[2];
        auto chan_idx = idx_into_sample / one_channel_sz;

        auto idx_into_channel = idx_into_sample % one_channel_sz;
        auto one_row_sz = shape[2];
        auto row_idx = idx_into_channel / one_row_sz;
        auto col_idx = idx_into_channel % one_row_sz;

        return {sample_idx, row_idx, col_idx, chan_idx};
    }
};


// compute Axes storage order based on axis characteristics

// ================================================================ Axes

// ------------------------------------------------ axis type

template<class Ax0=AxisContig, class Ax1=AxisUnused,
         class Ax2=AxisUnused, class Ax3=AxisUnused,
         int Order=StorageOrders::Unspecified>
struct Axes {
    using AxisT0 = Ax0;
    using AxisT1 = Ax1;
    using AxisT2 = Ax2;
    using AxisT3 = Ax3;

    // static const int debug = false; // TODO rm

    static const int order = Order;
    static const int rank =
        Ax0::is_used + Ax1::is_used + Ax2::is_used + Ax3::is_used;
    static const bool is_any_ax_contig =
        Ax0::is_contig || Ax1::is_contig || Ax2::is_contig || Ax3::is_contig;
    static const bool is_dense = (rank >= 1) &&
        (Ax0::is_dense || !Ax0::is_used) &&
        (Ax1::is_dense || !Ax1::is_used) &&
        (Ax2::is_dense || !Ax2::is_used) &&
        (Ax3::is_dense || !Ax3::is_used);

    // static const int is_rowmajor = Ax0::is_contig


    // is_dense -> one ax must be contiguous, or this makes no sense
    static_assert(!is_dense || is_any_ax_contig,
        "Somehow dense, but nothing contiguous!?");
};


// template<int Ax, class Axis> struct getAxis {};
// template<class Axis>

// ------------------------------------------------ aliases for common axes

using AxesDense1D =
    Axes<AxisContig, AxisUnused, AxisUnused, AxisUnused, StorageOrders::RowMajor>;

using AxesRowMajor2D =
    Axes<AxisDense, AxisContig, AxisUnused, AxisUnused, StorageOrders::RowMajor>;
using AxesRowMajor3D =
    Axes<AxisDense, AxisDense, AxisContig, AxisUnused, StorageOrders::RowMajor>;
using AxesRowMajor4D =
    Axes<AxisDense, AxisDense, AxisDense, AxisContig, StorageOrders::RowMajor>;

using AxesColMajor2D =
    Axes<AxisContig, AxisDense, AxisUnused, AxisUnused, StorageOrders::ColMajor>;
using AxesColMajor3D =
    Axes<AxisContig, AxisDense, AxisDense, AxisUnused, StorageOrders::ColMajor>;
using AxesColMajor4D =
    Axes<AxisContig, AxisDense, AxisDense, AxisDense, StorageOrders::ColMajor>;

using AxesNCHW =
    Axes<AxisDense, AxisDense, AxisContig, AxisDense, StorageOrders::NCHW>;

// ------------------------------------------------ axes manipulation

template<int ax, class Axes> struct getAxis {};
template<class Axes> struct getAxis<0, Axes> { using type = typename Axes::AxisT0; };
template<class Axes> struct getAxis<1, Axes> { using type = typename Axes::AxisT1; };
template<class Axes> struct getAxis<2, Axes> { using type = typename Axes::AxisT2; };
template<class Axes> struct getAxis<3, Axes> { using type = typename Axes::AxisT3; };

#define GET_AXIS_T(AXES, INT) typename getAxis<INT, AXES>::type

template<class AxesT, int StaticDim0=0, int StaticDim1=0, int StaticDim2=0, int StaticDim3=0>
struct setStaticSizes {
    using AxisT0 = SET_AXIS_SIZE(AxesT, 0, StaticDim0);
    using AxisT1 = SET_AXIS_SIZE(AxesT, 1, StaticDim1);
    using AxisT2 = SET_AXIS_SIZE(AxesT, 2, StaticDim2);
    using AxisT3 = SET_AXIS_SIZE(AxesT, 3, StaticDim3);
    static const int order = AxesT::order;
    using type = Axes<AxisT0, AxisT1, AxisT2, AxisT3, order>;
};

template<int Rank, int Order> struct GetDefaultAxesType {};
template<> struct GetDefaultAxesType<1, StorageOrders::RowMajor> { using type = AxesDense1D; };
template<> struct GetDefaultAxesType<1, StorageOrders::ColMajor> { using type = AxesDense1D; };
template<> struct GetDefaultAxesType<2, StorageOrders::RowMajor> { using type = AxesRowMajor2D; };
template<> struct GetDefaultAxesType<2, StorageOrders::ColMajor> { using type = AxesColMajor2D; };
template<> struct GetDefaultAxesType<3, StorageOrders::RowMajor> { using type = AxesRowMajor3D; };
template<> struct GetDefaultAxesType<3, StorageOrders::ColMajor> { using type = AxesColMajor3D; };
template<> struct GetDefaultAxesType<4, StorageOrders::RowMajor> { using type = AxesRowMajor4D; };
template<> struct GetDefaultAxesType<4, StorageOrders::ColMajor> { using type = AxesColMajor4D; };
template<> struct GetDefaultAxesType<4, StorageOrders::NCHW>     { using type = AxesNCHW; };

template<int Rank, int Order, int StaticDim0, int StaticDim1,
    int StaticDim2, int StaticDim3>
struct GetAxesType {
    using baseAxesType = typename GetDefaultAxesType<Rank, Order>::type;
//    static_assert(baseAxesType::attrItDoesntHave, "type of GetAxesType: ");
    using type = typename setStaticSizes<
        baseAxesType, StaticDim0, StaticDim1, StaticDim2, StaticDim3>::type;
};


// ------------------------------------------------ strides for various axes

template<class AxesT, class IdxT=DefaultIndexType>
std::array<IdxT, AxesT::rank> default_strides_for_shape(
    std::array<IdxT, AxesT::rank> shape)
{
    static const int rank = AxesT::rank;
    static const int order = AxesT::order;
    static_assert(rank >= 0, "Rank must be >= 0!");
    static_assert(rank <= 4, "Rank must be <= 4!");
    // TODO rm rank <= 2 after debug
    static_assert(rank <= 2 || AxesT::is_dense, "Only dense axes can use default strides!");
    static_assert(rank <= 2 || order != StorageOrders::Unspecified,
        "Must specify storage order for rank 3 tensors and above!");
    static_assert((order == StorageOrders::RowMajor ||
            order == StorageOrders::ColMajor ||
            order == StorageOrders::NCHW ||
            order == StorageOrders::Unspecified),
        "Only StorageOrders RowMajor, ColMajor, NCHW, "
            "and Unspecified supported!");
    static_assert(order != StorageOrders::NCHW || rank == 4,
        "NCHW order only supported for rank 4 tensors!");

    std::array<IdxT, AxesT::rank> strides{0};
    if (rank == 1) {
        // static_assert(AxisT0::is_contig,
        //     "1D array must be contiguous to use default strides!");
        strides[0] = 1;
        return strides;
    }
    if (rank == 2) {
        if (AxesT::AxisT0::is_contig) { // colmajor
            strides[0] = 1;
            strides[1] = shape[0];
        } else { // rowmajor
            strides[0] = shape[1];
            strides[1] = 1;
        }
        return strides;
    }
    switch(order) { // rank 3 or 4 if we got to here
        case StorageOrders::RowMajor:
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; i--) {
                strides[i] = shape[i + 1] * strides[i + 1];
            }
            break;
        case StorageOrders::ColMajor:
            strides[0] = 1;
            for (int i = 1; i < rank; i++) {
                strides[i] = shape[i - 1] * strides[i - 1];
            }
            break;
        case StorageOrders::NCHW:
            // conceptually, axes mean NHWC: #imgs, nrows, ncols, nchannels
            strides[0] = shape[1] * shape[2] * shape[3]; // sz of whole img
            strides[1] = shape[2];  // row stride = number of cols
            strides[2] = 1;         // col stride = 1, like rowmajor
            strides[3] = shape[1] * shape[2];  // channel stride = nrows * ncols
            break;
        default:
            assert("Somehow got unrecognized storage order!");
            break; // can't happen
    }
    return strides;
}

template<class AxesT, class ShapeT>
ShapeT clip_shape_to_static_bounds(const ShapeT& shape) {
    static const int rank = AxesT::rank;
    ShapeT ret(shape);
    using bounds0 = GET_AXIS_T(AxesT, 0)::size_bounds;
    using bounds1 = GET_AXIS_T(AxesT, 1)::size_bounds;
    using bounds2 = GET_AXIS_T(AxesT, 2)::size_bounds;
    using bounds3 = GET_AXIS_T(AxesT, 3)::size_bounds;
    if (rank >= 1 && bounds0::is_valid) {
        ret[0] = MIN(MAX(bounds0::min, shape[0]), bounds0::max);
    }
    if (rank >= 2 && bounds1::is_valid) {
        ret[1] = MIN(MAX(bounds1::min, shape[1]), bounds1::max);
    }
    if (rank >= 3 && bounds2::is_valid) {
        ret[2] = MIN(MAX(bounds2::min, shape[2]), bounds2::max);
    }
    if (rank >= 4 && bounds3::is_valid) {
        ret[3] = MIN(MAX(bounds3::min, shape[3]), bounds3::max);
    }
    return ret;
}

template<class DataT, class AxesT=AxesDense1D, class IdxT=DefaultIndexType>
struct ArrayView {
    static const int rank = AxesT::rank;
    static const int order = AxesT::order;
    using axes_t = AxesT;
    using strides_t = std::array<IdxT, rank>;
    using shape_t = std::array<IdxT, rank>;
    using idxs_t = strides_t;

    ArrayView(DataT *const data, const shape_t& shape):
        _data(data),
        _shape(clip_shape_to_static_bounds<AxesT>(shape)),
        _strides(default_strides_for_shape<AxesT>(_shape))
    {
        // printf("I'm an ArrayView and I have rank %d!\n", rank);
    };

    // these two funcs are to make using CUDA thread/block indices easier
    IdxT flatten_idxs(const idxs_t& idxs) {
        IdxT idx = 0;
        for (int i = 0; i < rank; i++) {
            idx += idxs[i] * _strides[i];
        }
        return idx;
    }
    // needs dense array; TODO allow arbitrary strides
    idxs_t unflatten_dense_idx(IdxT idx) {
        auto helper = idxs_from_flat_idx<rank, order>();
        return helper(_shape, idx);
    }

    // template<class IntT=IdxT, REQ_RANGE_ENCOMPASSES(IdxT, IntT)>
    // DataT& operator[](const std::array<IntT, NumIdxs>& idxs) {
    template<class IntT=IdxT, REQ_RANGE_ENCOMPASSES(IdxT, IntT)>
    DataT& operator[](const idxs_t& idxs) {
        return _data[flatten_idxs(idxs)];
    }

    const shape_t& shape() const { return _shape; }
    const shape_t& strides() const { return _strides; }

private:
    DataT *const _data;
    const shape_t _shape;
    const strides_t _strides;
};


template<int Rank, int Order=StorageOrders::RowMajor,
    int StaticDim0=0, int StaticDim1=0, int StaticDim2=0, int StaticDim3=0,
    class IdxT=DefaultIndexType, class DataT=void>
struct GetArrayViewType {
    using AxesT = typename GetAxesType<Rank, Order,
        StaticDim0, StaticDim1, StaticDim2, StaticDim3>::type;
//    static_assert(AxesT::attrItDoesntHave, "type of GetAxesType: ");
    using type = ArrayView<DataT, AxesT, IdxT>;
};

// wrappers to make views of different ranks
template<int Order=StorageOrders::RowMajor,
    int StaticDim0=0, int StaticDim1=0, int StaticDim2=0, int StaticDim3=0,
    class IdxT=DefaultIndexType, class DataT=void>
auto make_view(DataT* data, IdxT dim0, IdxT dim1, IdxT dim2, IdxT dim3)
    -> typename GetArrayViewType<4, Order,
        StaticDim0, StaticDim1, StaticDim2, StaticDim3, IdxT, DataT>::type
{
    using ArrayViewT = typename GetArrayViewType<4, Order,
        StaticDim0, StaticDim1, StaticDim2, StaticDim3, IdxT, DataT>::type;
    return ArrayViewT(data, {dim0, dim1, dim2, dim3});
}
template<int Order=StorageOrders::RowMajor,
    int StaticDim0=0, int StaticDim1=0, int StaticDim2=0,
    class IdxT=DefaultIndexType, class DataT=void>
auto make_view(DataT* data, IdxT dim0, IdxT dim1, IdxT dim2)
    -> typename GetArrayViewType<3, Order,
        StaticDim0, StaticDim1, StaticDim2, 0, IdxT, DataT>::type
{
    using ArrayViewT = typename GetArrayViewType<3, Order,
        StaticDim0, StaticDim1, StaticDim2, 0, IdxT, DataT>::type;
    return ArrayViewT(data, {dim0, dim1, dim2});
}
template<int Order=StorageOrders::RowMajor, int StaticDim0=0, int StaticDim1=0,
    class IdxT=DefaultIndexType, class DataT=void>
auto make_view(DataT* data, IdxT dim0, IdxT dim1)
    -> typename GetArrayViewType<2, Order,
        StaticDim0, StaticDim1, 0, 0, IdxT, DataT>::type
{
    using ArrayViewT = typename GetArrayViewType<2, Order,
        StaticDim0, StaticDim1, 0, 0, IdxT, DataT>::type;
    return ArrayViewT(data, {dim0, dim1});
}
template<int Order=StorageOrders::RowMajor, int StaticDim0=0,
    class IdxT=DefaultIndexType, class DataT=void>
auto make_view(DataT* data, IdxT dim0)
    -> typename GetArrayViewType<1, Order,
        StaticDim0, 0, 0, 0, IdxT, DataT>::type
{
    using ArrayViewT = typename GetArrayViewType<1, Order,
        StaticDim0, 0, 0, 0, IdxT, DataT>::type;
    return ArrayViewT(data, {dim0});
}

} // namespace ar

#endif /* array_h */
