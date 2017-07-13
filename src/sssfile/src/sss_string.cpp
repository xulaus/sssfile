#include <cmath>
#include <gsl/gsl>

#include "sss_string.h"

#ifdef assert
  #import <stdexcept>
  #undef assert
  #define STR(x) #x
  #define TO_STR(x) STR(x)
  #define assert(x, ...) do{ if(!(x)) throw std::runtime_error("Assertion Failed!\n" __FILE__ ":" TO_STR(__LINE__) "\n\t" #x "\n" __VA_ARGS__); } while(false)
#endif

bool SSSFile::operator==(const SSSFile::string &a, const SSSFile::string &b)
{
    if (a.length() != b.length())
    {
        return false;
    }

    const char *this_str = a.data();
    const char *that_str = b.data();
    for (size_t i = 0; i < a.length(); i++)
    {
        if (this_str[i] != that_str[i])
        {
            return false;
        }
    }
    return true;
}

template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
T to_numeric(const gsl::span<const char> string)
{
    const size_t size = string.length();
    if (size == 0)
    {
        return NAN;
    }

    T ret = 0;
    size_t start = 0;
    const char *const f_str = string.data();

    while(f_str[start] == ' ') { start++; }

    // Find sign
    int sign = 1;
    if(std::is_signed<T>::value)
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
    for (; pos < size && (!std::is_floating_point<T>::value || ( f_str[pos] != 'E' && f_str[pos] != 'e' && f_str[pos] != '.'));
         pos++)
    {
        const char c = f_str[pos];
        assert(c <= '9' && c >= '0');
        ret = 10 * ret + (c - '0');
    }

    if(std::is_floating_point<T>::value)
    {
        // Find fractional component
        if (f_str[pos] == '.')
        {
            pos += 1;
            T divisor = 1.0;
            for (; pos < size && f_str[pos] != 'E' && f_str[pos] != 'e'; pos++)
            {
                const char c = f_str[pos];
                assert(c <= '9' && c >= '0');
                divisor /= 10;
                ret = ret + (c - '0') * divisor;
            }
        }

        // Find mantisa
        if (++pos < size)
        {
            auto mantisa = gsl::span<const char>(&(f_str[pos]), size - pos);
            ret *= std::pow(10, to_numeric<int>(mantisa));
        }
    }
    return ret * sign;
};

int SSSFile::to_i(const gsl::span<const char> string)
{
    return to_numeric<int>(string);
}

double SSSFile::to_f(const gsl::span<const char> string)
{
    return to_numeric<double>(string);
}
