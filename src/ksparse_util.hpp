//
//  ksparse_util.hpp
//  Copyright © 2019 D Blalock. All rights reserved.
//

#ifndef _ksparse_util_hpp
#define _ksparse_util_hpp

#include <stdio.h> // TODO rm after debug

#include "dtype_traits.hpp"
#include "macros.hpp"  // for type traits macros

// NOTE to self: do not reinterpret_cast, even with pointers

/** packs idx in lower NBits bits and sets rest of bits to those in val */
// template<int NBits, bool Safe=false, class IdxT=void, class DataT=void,
template<bool Safe=false, class IdxT=void, class DataT=void, REQUIRE_INT(IdxT)>
static inline DataT pack_idx_val(uint8_t nbits, IdxT idx, DataT val) {
    static_assert(sizeof(DataT) >= sizeof(IdxT),
        "Index must use no more bytes than data type");
    using uint_t = typename scalar_traits<DataT>::reinterpret_as_uint_type;
    // auto val_bytes = (uint_t)val;
    auto idx_bytes = (uint_t)idx;
    // auto val_bytes = reinterpret_cast<uint_t>(val);
    // auto idx_bytes = reinterpret_cast<uint_t>(idx);
    // uint_t val_bytes = *reinterpret_cast<uint_t*>(&val);
    uint_t val_bytes = *((uint_t*)&val);
    // uint_t idx_bytes = *reinterpret_cast<uint_t*>(&idx);
    if (Safe) { // zero bits above lowest Nbits of them in idx
        IdxT mask = (((uint_t)1) << nbits) - 1;
        idx_bytes = idx_bytes & mask;
    }
    // printf("val_bytes: %u\n", val_bytes);
    // printf("val_bytes with low bits gone: %u\n", (val_bytes >> nbits) << nbits);
    // printf("idx_bytes: %u\n", idx_bytes);
    uint_t packed_bytes = ((val_bytes >> nbits) << nbits) | idx_bytes;
    // printf("packed_bytes: %u\n", packed_bytes);
    // printf("val_bytes packed: %u\n", (val_bytes >> nbits) << nbits);
    // return *reinterpret_cast<DataT*>(&packed_bytes);
    return *((DataT*)&packed_bytes);
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
    // auto packed_bytes = reinterpret_cast<uint_t>(packed);
    // uint_t packed_bytes = *reinterpret_cast<uint_t*>(&packed);
    uint_t packed_bytes = *((uint_t*)&packed);
    // printf("\nunpack saw packed bytes: %u\n", packed_bytes);
    // printf("packed bytes after shifting: %u\n", (packed_bytes >> nbits) << nbits);
    uint_t packed_without_idx = (packed_bytes >> nbits) << nbits;
    *val = *((DataT*)&packed_without_idx);
    uint_t mask = (((uint_t)1) << nbits) - 1;
    *idx = (IdxT)(packed_bytes & mask);
    // uint8_t idx_shift = 8*sizeof(uint_t) - nbits;
    // printf("unpack nbits, idx_shift: %u, %u\n", nbits, idx_shift);
    // *idx = (IdxT)((packed_bytes << idx_shift) >> idx_shift);
    // printf("unpacked to idx, val: %u, %g\n", *idx, (float)*val);
}


#endif // _ksparse_util_hpp
