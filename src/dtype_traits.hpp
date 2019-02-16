

#ifndef _dtype_traits_hpp
#define _dtype_traits_hpp

#include <utility>

template<int SizeNBytes> struct uint_type {};
template<> struct uint_type<1> { using type = uint8_t; };
template<> struct uint_type<2> { using type = uint16_t; };
template<> struct uint_type<4> { using type = uint32_t; };
template<> struct uint_type<8> { using type = uint64_t; };

template<int SizeNBytes> struct int_type {};
template<> struct int_type<1> { using type = int8_t; };
template<> struct int_type<2> { using type = int16_t; };
template<> struct int_type<4> { using type = int32_t; };
template<> struct int_type<8> { using type = int64_t; };

// template<class T1, class T2>
// static constexpr auto div_round_up(T1 val, T2 div_by) {
//     return (NBits / 8)  + ((Nbits % 8) > 0);
// };

// template<int NBits> struct needed_nbytes_for_nbits {
//     static const int value = (NBits / 8)  + ((Nbits % 8) > 0);
// };

// template<typename T, typename T2>
// static CONSTEXPR inline T round_up_to_multiple(T x, T2 multipleof) {
//     T remainder = x % multipleof;
//     return remainder ? (x + multipleof - remainder) : x;
// }

// template<typename T, typename T2>
// static CONSTEXPR inline auto div_round_up(T x, T2 y) -> decltype(x + y) {
//     return (x / y) + ((x % y) > 0);
//     // T remainder = x % multipleof;
//     // return remainder ? (x + multipleof - remainder) : x;
// }


template<class T1, class T2=T1>
struct scalar_traits {
    using sum_type = decltype(std::declval<T1>() + std::declval<T2>());
    using prod_type = decltype(std::declval<T1>() * std::declval<T2>());
    using reinterpret_as_uint_type = typename uint_type<sizeof(T1)>::type;
    using reinterpret_as_int_type = typename int_type<sizeof(T1)>::type;
};



#endif // _dtype_traits_hpp
