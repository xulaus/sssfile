if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


file(GLOB sssfile_src "src/*.cpp" "src/*.h")

project(libsssfile LANGUAGES CXX)
find_package(RapidXML REQUIRED)
include_directories(${RAPIDXML_INCLUDE_DIRS})
add_library (sssfile  ${sssfile_src})

set_property(TARGET sssfile PROPERTY CXX_STANDARD 17)
target_compile_definitions(sssfile PRIVATE EXPORTING)

target_include_directories(sssfile PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/sssfile>
    PRIVATE src)

install(TARGETS sssfile EXPORT SSSFileConfig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT SSSFileConfig DESTINATION share/SSSFile/cmake)
export(TARGETS sssfile FILE SSSFileConfig.cmake)

file(GLOB test_sssfile_src "test/*.cpp")
add_executable(test_sssfile ${test_sssfile_src} ${sssfile_src})
set_property(TARGET test_sssfile PROPERTY CXX_STANDARD 17)
target_include_directories(test_sssfile PRIVATE include)

include_directories(${CATCH_INCLUDE_DIRS})
include_directories(src)

include(ParseAndAddCatchTests)
ParseAndAddCatchTests(test_sssfile)
