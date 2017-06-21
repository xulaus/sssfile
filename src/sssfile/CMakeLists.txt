project(libsssfile LANGUAGES CXX)
add_library (sssfile src/sss_string.cpp)

set_property(TARGET sssfile PROPERTY CXX_STANDARD 14)
target_compile_definitions(sssfile PRIVATE EXPORTING)

target_include_directories(sssfile PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/sssfile>
    PRIVATE src)

include_directories(${CPP_GSL_INCLUDE_DIRS})

install(TARGETS sssfile EXPORT SSSFileConfig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT SSSFileConfig DESTINATION share/SSSFile/cmake)
export(TARGETS sssfile FILE SSSFileConfig.cmake)

add_executable(test_sssfile test/main.cpp test/sss_string_test.cpp src/sss_string.cpp)
set_property(TARGET test_sssfile PROPERTY CXX_STANDARD 14)
target_include_directories(test_sssfile PRIVATE include)
include_directories(${CATCH_INCLUDE_DIRS})

include(ParseAndAddCatchTests)
ParseAndAddCatchTests(test_sssfile)
