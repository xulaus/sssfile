#include <cstring>
#include <string_view>

namespace SSSFile
{
    class string
    {
      public:
        enum
        {
            HEAP_ALLOC_THRESHOLD = sizeof(char *) / sizeof(char)
        };

        union {
            char *heap;
            char inplace[HEAP_ALLOC_THRESHOLD];
        } str;

        int len;

        string(const char *in_string, size_t n)
            : len(n)
        {
            if (heap_allocated())
            {
                str.heap = new char[n];
            }
            memcpy(data(), in_string, length());
        }

        template <size_t N>
        string(const char (&in_string)[N])
            : string(in_string, N)
        {
        }

        string()
            : string(nullptr, 0){};

        string(const string &b)
            : string(b.data(), b.length())
        {
        }

        string(string &&b)
            : len(b.length())
        {
            if (heap_allocated())
            {
                str.heap = b.str.heap;
                b.str.heap = nullptr;
                b.len = 0;
            }
            else
            {
                memcpy(data(), b.data(), length());
            }
        }

        string &operator=(string &&b)
        {
            std::swap(*this, b);
            return *this;
        }

        string &operator=(const string &b)
        {
            string temp = string(b);
            return (*this = (std::move(temp)));
        }

        ~string()
        {
            if (heap_allocated())
            {
                delete[] str.heap;
            }
        };

        bool heap_allocated() const { return length() > HEAP_ALLOC_THRESHOLD; }

        size_t length() const { return len; }

        const char *data() const
        {
            return (heap_allocated()) ? str.heap : str.inplace;
        }

        char *data()
        {
            auto const_this = const_cast<const string *>(this);
            return const_cast<char *>(const_this->data());
        }

        const auto get() const { return std::string_view (data(), length()); }
    };

    bool operator==(const SSSFile::string &a, const SSSFile::string &b);

    int to_i(const std::string_view  string);
    double to_f(const std::string_view  string);
} // namespace SSSFile

namespace std
{
    template <>
    struct hash<SSSFile::string>
    {
        size_t operator()(const SSSFile::string &str) const
        {
            size_t h = 14695981039346656037ull;
            for (auto &&c : str.get())
            {
                h = (h ^ c) * 1099511628211ull;
            }
            return h;
        }
    };
} // namespace std
