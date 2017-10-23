cmake_minimum_required(VERSION 3.9)

if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
   set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Output directory for libraries")
endif()

if(NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
   set(CMAKE_INSTALL_INCLUDEDIR include CACHE PATH "Output directory for includes")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
project (sssfile)

find_package(Catch REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CATCH_CMAKE_DIR})
enable_testing()
add_subdirectory (sssfile)
