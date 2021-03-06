find_program(PYTHON "python3")

if (PYTHON)
    set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

    file(GLOB MODULE_SOURCE "bindings/*.cpp" "bindings/*.h")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PYTHON_FLAGS "-O0 -g")
    else()
        set(PYTHON_FLAGS "-O3")
    endif()

    add_custom_command(OUTPUT build/sssfile.so
        COMMAND
         ${CMAKE_COMMAND} -E env LDFLAGS="-L${CMAKE_CURRENT_BINARY_DIR}/../sssfile" CXXFLAGS=${PYTHON_FLAGS} CFLAGS=${PYTHON_FLAGS} ${PYTHON} ${SETUP_PY} bdist_wheel -k -d${CMAKE_CURRENT_BINARY_DIR} -b${CMAKE_CURRENT_BINARY_DIR}/build
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        MAIN_DEPENDENCY sssfile
        DEPENDENCY ${MODULE_SOURCE}
        VERBATIM
    )

    add_custom_target(python_module ALL DEPENDS build/sssfile.so)
    add_custom_target(python DEPENDS python_module)
    add_test(NAME pytest COMMAND ${PYTHON} -m pytest "${CMAKE_CURRENT_SOURCE_DIR}" WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build")
    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install ${SETUP_PY_OPTIONS})")
else()
    message(ERROR "Cannot find python")
endif()
