#ifndef __SSSFILE_COMMON_
#define __SSSFILE_COMMON_

#ifdef assert
#undef assert
#endif

#ifdef NDEBUG
#define assert(...)
#define ensure(x, message) (x)
#else
#import <stdexcept>
#define STR(x) #x
#define TO_STR(x) STR(x)
#define assert(x, ...)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(x))                                                                                                      \
            throw std::runtime_error("Assertion Failed!\n" __FILE__ ":" TO_STR(__LINE__) "\n\t" #x "\n" __VA_ARGS__);  \
    } while (false)
#define ensure(x, message) assert(x, message)
#endif

#include <cmath>
#include <string_view>

namespace SSSFile
{
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    bool to_numeric(const std::string_view &string, T &out)
    {
        const size_t size = string.length();
        if (size == 0)
        {
            return false;
        }

        T ret = 0;
        size_t start = 0;
        const char *const f_str = string.data();

        while (f_str[start] == ' ')
        {
            start++;
        }

        // Find sign
        int sign = 1;
        if (std::is_signed<T>::value)
        {
            switch (f_str[start])
            {
            case '-':
                sign = -1;
            case '+':
                start++;
            default:
                break;
            }
        }

        // Find whole number component
        auto pos = start;
        for (; pos < size &&
               (!std::is_floating_point<T>::value || (f_str[pos] != 'E' && f_str[pos] != 'e' && f_str[pos] != '.'));
             pos++)
        {
            const char c = f_str[pos];
            if (c > '9' || c < '0')
                return false;
            ret = 10 * ret + (c - '0');
        }

        if (std::is_floating_point<T>::value && pos < size)
        {
            // Find fractional component
            if (f_str[pos] == '.')
            {
                pos += 1;
                T divisor = 1.0;
                for (; pos < size && f_str[pos] != 'E' && f_str[pos] != 'e'; pos++)
                {
                    const char c = f_str[pos];
                    if (c > '9' || c < '0')
                        return false;
                    divisor /= 10;
                    ret = ret + (c - '0') * divisor;
                }
            }

            // Find mantisa
            if (++pos < size)
            {
                auto mantisa_str = std::string_view(&(f_str[pos]), size - pos);
                int mantisa = 0;
                if (!to_numeric(mantisa_str, mantisa))
                {
                    return false;
                }
                ret *= std::pow(10, mantisa);
            }
        }
        out = ret * sign;
        return true;
    };
}
#endif
