#ifdef assert
  #undef assert
#endif

#ifdef DEBUG
  #import <stdexcept>
  #define STR(x) #x
  #define TO_STR(x) STR(x)
  #define assert(x, ...) do{ if(!(x)) throw std::runtime_error("Assertion Failed!\n" __FILE__ ":" TO_STR(__LINE__) "\n\t" #x "\n" __VA_ARGS__); } while(false)
#else
    #define assert(...)
#endif
