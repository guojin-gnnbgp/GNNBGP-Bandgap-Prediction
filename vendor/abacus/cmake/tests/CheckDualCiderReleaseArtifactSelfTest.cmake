if(NOT DEFINED ABACUS_SOURCE_DIR)
  message(FATAL_ERROR "ABACUS_SOURCE_DIR is required")
endif()
foreach(_name CIDER_DUAL_CACHE CIDER_DUAL_LINK CIDER_DUAL_BINARY CIDER_DUAL_SELFTEST_DIR)
  if(NOT DEFINED ${_name} OR "${${_name}}" STREQUAL "")
    message(FATAL_ERROR "${_name} is required")
  endif()
endforeach()

set(_guard "${ABACUS_SOURCE_DIR}/cmake/tests/CheckDualCiderReleaseArtifact.cmake")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DCIDER_CACHE=${CIDER_DUAL_CACHE}
          -DCIDER_LINK=${CIDER_DUAL_LINK}
          -DCIDER_BINARY=${CIDER_DUAL_BINARY}
          -P "${_guard}"
  RESULT_VARIABLE _positive_result
  OUTPUT_VARIABLE _positive_stdout
  ERROR_VARIABLE _positive_stderr)
if(NOT _positive_result EQUAL 0)
  message(FATAL_ERROR
          "dual release guard rejected a valid artifact:\n${_positive_stdout}${_positive_stderr}")
endif()

file(MAKE_DIRECTORY "${CIDER_DUAL_SELFTEST_DIR}")
set(_native_cache "${CIDER_DUAL_SELFTEST_DIR}/native-only.cache")
file(WRITE "${_native_cache}"
     "ENABLE_CIDER:BOOL=ON\nENABLE_CIDER_EXTERNAL:BOOL=OFF\nENABLE_LIBXC:BOOL=ON\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DCIDER_CACHE=${_native_cache}
          -DCIDER_LINK=${CIDER_DUAL_LINK}
          -DCIDER_BINARY=${CIDER_DUAL_BINARY}
          -P "${_guard}"
  RESULT_VARIABLE _negative_result
  OUTPUT_QUIET ERROR_QUIET)
if(_negative_result EQUAL 0)
  message(FATAL_ERROR "dual release guard accepted an external-disabled cache")
endif()

message(STATUS "dual CIDER release artifact guard self-test passed")
