find_package(Git REQUIRED)
set(CATCH_URL https://github.com/catchorg/Catch2)
set(CATCH_VERSION v2.0.1)
message(STATUS "Using and shipping ${CATCH_URL} version ${CATCH_VERSION}")
set(CATCH_DIR "${CMAKE_SOURCE_DIR}/ext/Catch")
if(NOT EXISTS "${CATCH_DIR}")
  execute_process(COMMAND ${GIT_EXECUTABLE} clone ${CATCH_URL} "${CATCH_DIR}")
endif()

execute_process(COMMAND ${GIT_EXECUTABLE} fetch -p
                WORKING_DIRECTORY "${CATCH_DIR}")

execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${CATCH_VERSION}
                WORKING_DIRECTORY "${CATCH_DIR}")

set(CATCH_INCLUDE_DIR "${CATCH_DIR}/single_include")
set(CATCH_CMAKE_DIR "${CATCH_DIR}/contrib")

set(CATCH_INCLUDE_DIRS ${CATCH_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(catch DEFAULT_MSG CATCH_INCLUDE_DIR)
mark_as_advanced (CATCH_INCLUDE_DIR)
