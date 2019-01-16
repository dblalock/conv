

#ifndef _dtype_traits_hpp
#define _dtype_traits_hpp

#include <utility>

template<class T1, class T2>
struct scalar_traits {
    using sum_type = decltype(std::declval<T1>() + std::declval<T2>());
    using prod_type = decltype(std::declval<T1>() * std::declval<T2>());
};



#endif // _dtype_traits_hpp
