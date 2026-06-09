cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(knowhere CXX C)

# Use short SHA1 as version
# v2.6 tag is used for cardinal v2
set(CARDINAL_VERSION v3.0.1)
set(CARDINAL_REPO_URL "https://github.com/zilliztech/cardinal.git")
set(CARDINAL_EXTRA_FILTER_PATCH "${CMAKE_CURRENT_LIST_DIR}/extra_filter.patch")

set(CARDINAL_ROOT  "${KNOWHERE_THRID_ROOT}/cardinalv2")
message(STATUS "Build Cardinal-${CARDINAL_VERSION}")

function(apply_cardinal_extra_filter_patch)
    execute_process(COMMAND git -C ${CARDINAL_ROOT} apply --reverse --check ${CARDINAL_EXTRA_FILTER_PATCH}
            RESULT_VARIABLE CARDINAL_PATCH_REVERSE_CHECK_RESULT
            OUTPUT_VARIABLE CARDINAL_PATCH_REVERSE_CHECK_OUTPUT
            ERROR_VARIABLE CARDINAL_PATCH_REVERSE_CHECK_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (CARDINAL_PATCH_REVERSE_CHECK_RESULT EQUAL "0")
        message(STATUS "Cardinal extra filter patch already applied")
        return()
    endif()

    execute_process(COMMAND git -C ${CARDINAL_ROOT} apply --check ${CARDINAL_EXTRA_FILTER_PATCH}
            RESULT_VARIABLE CARDINAL_PATCH_CHECK_RESULT
            OUTPUT_VARIABLE CARDINAL_PATCH_CHECK_OUTPUT
            ERROR_VARIABLE CARDINAL_PATCH_CHECK_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (NOT CARDINAL_PATCH_CHECK_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to check cardinal extra filter patch: ${CARDINAL_PATCH_CHECK_ERROR}")
    endif()

    execute_process(COMMAND git -C ${CARDINAL_ROOT} apply ${CARDINAL_EXTRA_FILTER_PATCH}
            RESULT_VARIABLE CARDINAL_PATCH_RESULT
            OUTPUT_VARIABLE CARDINAL_PATCH_OUTPUT
            ERROR_VARIABLE CARDINAL_PATCH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (NOT CARDINAL_PATCH_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to apply cardinal extra filter patch: ${CARDINAL_PATCH_ERROR}")
    else()
        message(STATUS "Successfully applied cardinal extra filter patch")
    endif()
endfunction()

# Clone and checkout cardinal with given repo url and version
if (NOT EXISTS "${CARDINAL_ROOT}/.git")
    execute_process(COMMAND git clone ${CARDINAL_REPO_URL} ${CARDINAL_ROOT}
            RESULT_VARIABLE CARDINAL_CLONE_RESULT
            OUTPUT_VARIABLE CARDINAL_CLONE_OUTPUT
            ERROR_VARIABLE CARDINAL_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
    if (NOT CARDINAL_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone cardinal: ${CARDINAL_CLONE_ERROR}")
    else()
      message(STATUS "Successfully Clone Cardinal Repo")
      execute_process(COMMAND git -C ${CARDINAL_ROOT} checkout ${CARDINAL_VERSION}
            RESULT_VARIABLE CARDINAL_CHECKOUT_RESULT
            OUTPUT_VARIABLE CARDINAL_CHECKOUT_OUTPUT
            ERROR_VARIABLE CARDINAL_CHECKOUT_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE)
        if (NOT CARDINAL_CHECKOUT_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to checkout cardinal: ${CARDINAL_CHECKOUT_ERROR}")
        else()
            message(STATUS "Successfully checkout Cardinal Version : ${CARDINAL_VERSION}")
        endif()
    endif()
else()
    execute_process(
          COMMAND git -C ${CARDINAL_ROOT} rev-parse HEAD
          OUTPUT_VARIABLE GIT_COMMIT_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Cardinal repo already exist! git commit :  ${GIT_COMMIT_HASH}")
endif()

apply_cardinal_extra_filter_patch()

# Force checkout the version specified as `CARDINAL_VERSION` if `CARDINAL_VERSION_FORCE_CHECKOUT` is set
# Default do not checkout for better development convenience
if(CARDINAL_VERSION_FORCE_CHECKOUT)
    message(STATUS "Checking out cardinal version ${CARDINAL_VERSION}")

    execute_process(
            COMMAND git -C ${CARDINAL_ROOT} fetch
            RESULT_VARIABLE CARDINAL_FETCH_RESULT
            OUTPUT_VARIABLE CARDINAL_FETCH_OUTPUT
            ERROR_VARIABLE CARDINAL_FETCH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

    if(NOT CARDINAL_FETCH_RESULT EQUAL "0")
        message(
                FATAL_ERROR "Failed to fetch cardinal: ${CARDINAL_FETCH_ERROR}")
    endif()

    message(STATUS "Fetched cardinal ${CARDINAL_FETCH_OUTPUT}")

    execute_process(
            COMMAND git -C ${CARDINAL_ROOT} checkout ${CARDINAL_VERSION}
            RESULT_VARIABLE CARDINAL_CHECKOUT_RESULT
            OUTPUT_VARIABLE CARDINAL_CHECKOUT_OUTPUT
            ERROR_VARIABLE CARDINAL_CHECKOUT_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

    if(NOT CARDINAL_CHECKOUT_RESULT EQUAL "0")
        message(
                FATAL_ERROR "Failed to checkout cardinal: ${CARDINAL_CHECKOUT_ERROR}")
    endif()

    apply_cardinal_extra_filter_patch()
endif()

include(${CARDINAL_ROOT}/know/libcardinal.cmake)
