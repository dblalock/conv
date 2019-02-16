//
//  ksparse_util.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_util_hpp
#define _ksparse_util_hpp

#include "dtype_traits.hpp"
#include "macros.hpp"  // for type traits macros

/** packs idx in lower NBits bits and sets rest of bits to those in val */
// template<int NBits, bool Safe=false, class IdxT=void, class DataT=void,
template<bool Safe=false, class IdxT=void, class DataT=void, REQUIRE_INT(IdxT)>
static inline DataT pack_idx_val(uint8_t nbits, IdxT idx, DataT val) {
    static_assert(sizeof(DataT) >= sizeof(IdxT),
        "Index must use no more bytes than data type");
    using uint_t = typename scalar_traits<DataT>::reinterpret_as_uint_type;
    auto val_bytes = (uint_t)val;
    auto idx_bytes = (uint_t)idx;
    if (Safe) { // zero bits above lowest Nbits of them in idx
        IdxT mask = (((uint_t)1) << nbits) - 1;
        idx_bytes = idx_bytes & mask;
    }
    // printf("val_bytes: %u\n", val_bytes);
    return ((val_bytes >> nbits) << nbits) | idx_bytes;
}

/** unpacks idx from lower NBits bits and takes rest of bits as upper bits
 * of val */
// template<int NBits, class IdxT, class DataT, REQUIRE_INT(IdxT)>
// static inline void unpack_idx_val(DataT packed, IdxT* idx, DataT* val) {
template<class IdxT, class DataT, REQUIRE_INT(IdxT)>
static inline void unpack_idx_val(DataT packed, uint8_t nbits,
    IdxT* idx, DataT* val)
{
    static_assert(sizeof(DataT) >= sizeof(IdxT),
        "Index must use no more bytes than data type");
    using uint_t = typename scalar_traits<DataT>::reinterpret_as_uint_type;
    auto packed_bytes = (uint_t)packed;
    // printf("unpack saw packed bytes: %u\n", packed_bytes);
    *val = (DataT)((packed_bytes >> nbits) << nbits);
    uint_t mask = (((uint_t)1) << nbits) - 1;
    *idx = packed_bytes & mask;
    // uint8_t idx_shift = 8*sizeof(uint_t) - nbits;
    // printf("unpack nbits, idx_shift: %u, %u\n", nbits, idx_shift);
    // *idx = (IdxT)((packed_bytes << idx_shift) >> idx_shift);
    // printf("unpacked to idx, val: %u, %g\n", *idx, (float)*val);
}


#endif // _ksparse_util_hpp
