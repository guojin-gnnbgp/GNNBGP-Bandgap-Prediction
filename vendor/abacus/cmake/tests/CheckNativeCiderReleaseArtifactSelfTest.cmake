if(NOT DEFINED ABACUS_SOURCE_DIR)
  message(FATAL_ERROR "ABACUS_SOURCE_DIR is required")
endif()
if(NOT DEFINED CIDER_TEST_BINARY OR NOT EXISTS "${CIDER_TEST_BINARY}")
  message(FATAL_ERROR "CIDER_TEST_BINARY must name a working executable")
endif()
if(NOT DEFINED CIDER_SELFTEST_DIR OR CIDER_SELFTEST_DIR STREQUAL "")
  message(FATAL_ERROR "CIDER_SELFTEST_DIR must name an explicit scratch directory")
endif()

set(_root "${CIDER_SELFTEST_DIR}")
file(MAKE_DIRECTORY "${_root}")
set(_guard "${ABACUS_SOURCE_DIR}/cmake/tests/CheckNativeCiderReleaseArtifact.cmake")

function(_expect_rejected _name _cache_extra _link_extra)
  set(_cache "${_root}/${_name}.cache")
  set(_link "${_root}/${_name}.link")
  file(WRITE "${_cache}"
       "ENABLE_CIDER:BOOL=ON\nENABLE_LIBXC:BOOL=ON\n${_cache_extra}\n")
  file(WRITE "${_link}" "c++ main.o ${_link_extra} -o abacus\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DCIDER_CACHE=${_cache}"
            "-DCIDER_LINK=${_link}"
            "-DCIDER_BINARY=${CIDER_TEST_BINARY}"
            -P "${_guard}"
    RESULT_VARIABLE _status
    OUTPUT_QUIET ERROR_QUIET)
  if(_status EQUAL 0)
    message(FATAL_ERROR "release guard accepted prohibited ${_name} dependency")
  endif()
endfunction()

_expect_rejected(python-executable "Python3_EXECUTABLE:FILEPATH=/python" "")
_expect_rejected(python-library "Python3_LIBRARIES:FILEPATH=/libpython.so" "")
_expect_rejected(numpy-cache "Python3_NumPy_INCLUDE_DIRS:PATH=/numpy" "")
_expect_rejected(numpy-generic-cache "NumPy_INCLUDE_DIR:PATH=/numpy" "")
_expect_rejected(ciderpress-cache "CIDERPRESS_DIR:PATH=/ciderpress" "")
_expect_rejected(external-enabled "ENABLE_CIDER_EXTERNAL:BOOL=ON" "")
_expect_rejected(python-link "" "-lpython3.10")
_expect_rejected(numpy-link "" "-lnumpy")
_expect_rejected(ciderpress-link "" "CiderPress/runtime.o")
_expect_rejected(bridge-link "" "-lciderbridge")

message(STATUS "native CIDER release artifact guard self-test passed")
