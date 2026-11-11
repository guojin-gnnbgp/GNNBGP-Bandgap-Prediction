if(NOT TARGET cider_native)
  message(FATAL_ERROR "The focused native CIDER test requires the cider_native target")
endif()

add_test(
  NAME MODULE_HAMILT_CIDER_build_boundary_test
  COMMAND ${CMAKE_COMMAND}
          -DABACUS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundary.cmake
)

add_test(
  NAME MODULE_HAMILT_CIDER_build_boundary_self_test
  COMMAND ${CMAKE_COMMAND}
          -DABACUS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DCIDER_BOUNDARY_SELF_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/native-cider-boundary-fixtures
          -P ${CMAKE_SOURCE_DIR}/cmake/tests/CheckNativeCiderBuildBoundarySelfTest.cmake
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_boundary_test
  LIBS cider_native
  SOURCES module_boundary_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_model_loader_test
  LIBS cider_native
  SOURCES model_loader_test.cpp
)

AddTest(
  TARGET MODULE_IO_CIDER_model_input_test
  LIBS cider_native
  SOURCES ../../../../source_io/module_parameter/cider_model_input.cpp
          ../../../../source_io/module_parameter/test/cider_model_input_test.cpp
)
AddTest(
  TARGET MODULE_HAMILT_CIDER_feature_normalizer_test
  LIBS cider_native
  SOURCES feature_normalizer_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_feature_transform_test
  LIBS cider_native
  SOURCES feature_transform_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_spline_evaluator_test
  LIBS cider_native
  SOURCES spline_evaluator_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_mapped_evaluator_test
  LIBS cider_native
  SOURCES mapped_evaluator_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_semilocal_plan_test
  LIBS cider_native
  SOURCES semilocal_plan_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_nldf_plan_test
  LIBS cider_native
  SOURCES nldf_plan_test.cpp
)

AddTest(
  TARGET MODULE_HAMILT_CIDER_evaluator_test
  LIBS cider_native Libxc::xc
  SOURCES evaluator_test.cpp
)

AddTest(
  TARGET MODULE_ESTATE_CIDER_pot_native_test
  LIBS cider_pot_native cider_native Libxc::xc
  SOURCES ../../../../source_estate/module_pot/test/pot_cider_xc_native_test.cpp
)
AddTest(
  TARGET MODULE_ESTATE_CIDER_pot_runtime_factory_test
  LIBS cider_pot_native cider_native Libxc::xc
  SOURCES ../../../../source_estate/module_pot/test/pot_cider_xc_runtime_factory_test.cpp
)
if(ENABLE_CIDER_EXTERNAL)
  AddTest(
    TARGET MODULE_ESTATE_CIDER_pot_external_test
    LIBS cider_pot_native cider_native Libxc::xc
    SOURCES ../../../../source_estate/module_pot/test/pot_cider_xc_external_test.cpp
  )
  AddTest(
    TARGET MODULE_ESTATE_CIDER_dual_backend_test
    LIBS cider_pot_native cider_native Libxc::xc
    SOURCES ../../../../source_estate/module_pot/test/pot_cider_xc_dual_backend_test.cpp
  )
endif()
if(CIDER_NATIVE_CORPUS_ROOT)
  set(_cider_corpus_tests
      MODULE_HAMILT_CIDER_model_loader_test
      MODULE_IO_CIDER_model_input_test
      MODULE_HAMILT_CIDER_spline_evaluator_test
      MODULE_HAMILT_CIDER_mapped_evaluator_test
      MODULE_HAMILT_CIDER_evaluator_test
      MODULE_ESTATE_CIDER_pot_native_test
      MODULE_ESTATE_CIDER_pot_runtime_factory_test)
  if(ENABLE_CIDER_EXTERNAL)
    list(APPEND _cider_corpus_tests
      MODULE_ESTATE_CIDER_pot_external_test
      MODULE_ESTATE_CIDER_dual_backend_test)
  endif()
  set_tests_properties(
    ${_cider_corpus_tests}
    PROPERTIES ENVIRONMENT "CIDER_NATIVE_CORPUS_ROOT=${CIDER_NATIVE_CORPUS_ROOT}"
  )
endif()
