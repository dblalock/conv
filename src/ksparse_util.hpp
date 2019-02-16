//
//  ksparse_util.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_util_hpp
#define _ksparse_util_hpp

#include "macros.hpp"  // for type traits macros


/** packs idx in lower NBits bits and sets rest of bits to those in val */
template<int NBits, bool Safe=false, class IdxT=void, class DataT=void,
    REQUIRE_INT(IdxT)>
static inline DataT pack_idx_val(IdxT idx, DataT val) {
    static_assert(sizeof(DataT) >= sizeof(IdxT),
        "Index must use no more bytes than data type");
    using uint_t = typename scalar_traits<DataT>::reinterpret_as_uint_type;
    auto val_bytes = (uint_t)val;
    auto idx_bytes = (uint_t)idx;
    if (Safe) { // zero bits above lowest Nbits of them in idx
        IdxT mask = (((uint_t)1) << NBits) - 1;
        idx_bytes = idx_bytes & mask;
    }
    return ((val_bytes >> NBits) << NBits) | idx_bytes;
}

/** unpacks idx from lower NBits bits and takes rest of bits as upper bits
 * of val */
template<int NBits, class IdxT, class DataT, REQUIRE_INT(IdxT)>
static inline void unpack_idx_val(DataT packed, IdxT* idx, DataT* val) {
    static_assert(sizeof(DataT) >= sizeof(IdxT),
        "Index must use no more bytes than data type");
    using uint_t = typename scalar_traits<DataT>::reinterpret_as_uint_type;
    auto packed_bytes = (uint_t)packed;
    *val = (DataT)((packed_bytes >> NBits) << NBits);
    auto idx_shift = 8*sizeof(uint_t) - NBits;
    *idx = (DataT)((packed_bytes << idx_shift) >> idx_shift);
}


#endif // _ksparse_util_hpp
