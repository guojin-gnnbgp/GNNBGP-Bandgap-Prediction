add_library(
  cider_native
  STATIC
  evaluator.cpp
  feature_normalizer.cpp
  feature_transform.cpp
  mapped_evaluator.cpp
  model.cpp
  model_loader.cpp
  nldf_backend.cpp
  nldf_convolution.c
  nldf_interpolation.c
  nldf_plan.cpp
  semilocal_plan.cpp
  spline_evaluator.cpp
)

# ABACUS configures this tree as CXX-only.  Keep the provenance-preserving C
# migration as C syntax while compiling it in the private native target.
set_source_files_properties(nldf_convolution.c nldf_interpolation.c PROPERTIES LANGUAGE CXX)

target_include_directories(
  cider_native
  PUBLIC ${CMAKE_SOURCE_DIR}/source
)

if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(cider_native PRIVATE yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  target_link_libraries(cider_native PRIVATE yaml-cpp)
else()
  message(FATAL_ERROR "cider_native requires a yaml-cpp CMake target")
endif()

target_compile_features(cider_native PUBLIC cxx_std_11)
target_link_libraries(cider_native PUBLIC Libxc::xc)
if(TARGET MKL::MKL)
  target_link_libraries(cider_native PRIVATE MKL::MKL)
elseif(TARGET FFTW3::FFTW3)
  target_link_libraries(cider_native PRIVATE FFTW3::FFTW3)
else()
  message(FATAL_ERROR "cider_native NLDF requires the ABACUS-selected MKL or FFTW3 target")
endif()
set_target_properties(cider_native PROPERTIES CXX_VISIBILITY_PRESET hidden)

if(ENABLE_COVERAGE)
  add_coverage(cider_native)
endif()
