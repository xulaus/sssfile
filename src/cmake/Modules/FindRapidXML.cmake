find_package(Git REQUIRED)
set(RAPIDXML_URL https://github.com/smanders/rapidxml)
set(RAPIDXML_VERSION v1.13)
message(STATUS "Using and shipping ${RAPIDXML_URL} version ${RAPIDXML_VERSION}")
set(RAPIDXML_DIR "${CMAKE_SOURCE_DIR}/ext/RapidXML")
if(NOT EXISTS "${RAPIDXML_DIR}")
  execute_process(COMMAND ${GIT_EXECUTABLE} clone ${RAPIDXML_URL} "${RAPIDXML_DIR}")
endif()

execute_process(COMMAND ${GIT_EXECUTABLE} fetch -p
                WORKING_DIRECTORY "${CATCH_DIR}")

execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${RAPIDXML_VERSION}
                WORKING_DIRECTORY "${RAPIDXML_DIR}")

set(RAPIDXML_INCLUDE_DIR "${RAPIDXML_DIR}")

set(RAPIDXML_INCLUDE_DIRS ${RAPIDXML_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RAPIDXML DEFAULT_MSG RAPIDXML_INCLUDE_DIR)
mark_as_advanced (RAPIDXML_INCLUDE_DIR)
