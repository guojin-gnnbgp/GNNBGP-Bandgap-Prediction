if(NOT DEFINED CIDER_BOUNDARY_SELF_TEST_DIR)
  set(CIDER_BOUNDARY_SELF_TEST_DIR "/tmp/abacus-native-cider-boundary-fixtures")
endif()
set(_fixture_dir "${CIDER_BOUNDARY_SELF_TEST_DIR}")
file(MAKE_DIRECTORY "${_fixture_dir}")

set(_unrelated_python "${_fixture_dir}/unrelated-python.cmake")
file(WRITE "${_unrelated_python}"
     "find_package(Python3 COMPONENTS Interpreter)\n"
     "target_link_libraries(unrelated_tool PRIVATE Python3::Python)\n"
     "if(ENABLE_CIDER)\n  message(STATUS \"native only\")\nendif()\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DABACUS_SOURCE_DIR=${ABACUS_SOURCE_DIR}
          -DCIDER_BUILD_FILES=${_unrelated_python}
          -DCIDER_DEDICATED_BUILD_FILES=
          -P ${ABACUS_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundary.cmake
  RESULT_VARIABLE _unrelated_result
  OUTPUT_VARIABLE _unrelated_stdout
  ERROR_VARIABLE _unrelated_stderr)
if(NOT _unrelated_result EQUAL 0)
  message(FATAL_ERROR
          "boundary guard rejected unrelated Python use:\n${_unrelated_stdout}${_unrelated_stderr}")
endif()

set(_legacy_cider "${_fixture_dir}/legacy-cider.cmake")
file(WRITE "${_legacy_cider}"
     [=[if(ENABLE_CIDER)
  target_link_libraries(abacus PRIVATE ${Python3_LIBRARIES})
endif()
]=])
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DABACUS_SOURCE_DIR=${ABACUS_SOURCE_DIR}
          -DCIDER_BUILD_FILES=${_legacy_cider}
          -DCIDER_DEDICATED_BUILD_FILES=
          -P ${ABACUS_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundary.cmake
  RESULT_VARIABLE _legacy_result
  OUTPUT_VARIABLE _legacy_stdout
  ERROR_VARIABLE _legacy_stderr)
if(_legacy_result EQUAL 0)
  message(FATAL_ERROR "boundary guard accepted a legacy Python CIDER linkage")
endif()

set(_legacy_bridge "${_fixture_dir}/legacy-bridge.cmake")
file(WRITE "${_legacy_bridge}"
     "find_library(CIDERBRIDGE_LIBRARY ciderbridge)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DABACUS_SOURCE_DIR=${ABACUS_SOURCE_DIR}
          -DCIDER_BUILD_FILES=${_legacy_bridge}
          -DCIDER_DEDICATED_BUILD_FILES=
          -P ${ABACUS_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundary.cmake
  RESULT_VARIABLE _bridge_result
  OUTPUT_VARIABLE _bridge_stdout
  ERROR_VARIABLE _bridge_stderr)
if(_bridge_result EQUAL 0)
  message(FATAL_ERROR "boundary guard accepted a legacy CIDER bridge lookup")
endif()

set(_optional_bridge "${_fixture_dir}/optional-bridge.cmake")
file(WRITE "${_optional_bridge}"
     [=[if(ENABLE_CIDER_EXTERNAL)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
  find_library(CIDERBRIDGE_LIBRARY ciderbridge)
  target_link_libraries(abacus PRIVATE Python3::Python ${CIDERBRIDGE_LIBRARY})
endif()
]=])
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -DABACUS_SOURCE_DIR=${ABACUS_SOURCE_DIR}
          -DCIDER_BUILD_FILES=${_optional_bridge}
          -DCIDER_DEDICATED_BUILD_FILES=
          -P ${ABACUS_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundary.cmake
  RESULT_VARIABLE _optional_result
  OUTPUT_VARIABLE _optional_stdout
  ERROR_VARIABLE _optional_stderr)
if(NOT _optional_result EQUAL 0)
  message(FATAL_ERROR
          "boundary guard rejected optional external backend:\n${_optional_stdout}${_optional_stderr}")
endif()

message(STATUS "native CIDER build boundary self-test passed")
