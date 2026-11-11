if(NOT DEFINED CIDER_BUILD_FILES)
  set(CIDER_BUILD_FILES
      "${ABACUS_SOURCE_DIR}/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_estate/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_hamilt/module_xc/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_hamilt/module_xc/cider/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_hamilt/module_xc/cider/test/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_io/CMakeLists.txt")
endif()

if(NOT DEFINED CIDER_DEDICATED_BUILD_FILES)
  set(CIDER_DEDICATED_BUILD_FILES
      "${ABACUS_SOURCE_DIR}/source/source_hamilt/module_xc/cider/CMakeLists.txt"
      "${ABACUS_SOURCE_DIR}/source/source_hamilt/module_xc/cider/test/CMakeLists.txt")
endif()

# These names belong exclusively to the optional external bridge integration.
# They are invalid outside an ENABLE_CIDER_EXTERNAL block.
set(_legacy_bridge_patterns
    "CIDERPRESS_DIR"
    "CIDERPRESS_LIB_DIR"
    "CIDERBRIDGE"
    "CIDERBRIDGE_LIBRARY"
    "CIDERBRIDGE_INCLUDE_DIR"
    "cider_bridge"
    "cider_bridge.h"
    "ciderbridge")

# Python itself is not forbidden in ABACUS build logic.  Only dependency hooks
# inside ENABLE_CIDER blocks (or dedicated native-CIDER build files) are banned.
set(_legacy_python_dependency_patterns
    "COMPONENTS Interpreter Development NumPy"
    "COMPONENTS Development NumPy"
    "Python3::Python"
    "Python3::NumPy"
    "Python3_INCLUDE_DIRS"
    "Python3_LIBRARIES"
    "Python3_NumPy"
    "NumPy_INCLUDE"
    "numpy/arrayobject.h"
    "Python.h")

function(_cider_assert_absent _file _contents _patterns)
  foreach(_pattern IN LISTS ${_patterns})
    string(FIND "${_contents}" "${_pattern}" _offset)
    if(NOT _offset EQUAL -1)
      message(FATAL_ERROR
              "native CIDER build boundary regression: ${_file} contains ${_pattern}")
    endif()
  endforeach()
endfunction()

# Remove balanced if(...) regions whose opening condition selects the optional
# external backend. The remaining text is the native-only build surface.
function(_cider_without_external_blocks _contents _output)
  string(REPLACE "\n" ";" _lines "${_contents}")
  set(_active FALSE)
  set(_depth 0)
  set(_native "")
  foreach(_line IN LISTS _lines)
    if(NOT _active AND _line MATCHES "^[ \t]*if[ \t]*\\([^\\)]*ENABLE_CIDER_EXTERNAL")
      set(_active TRUE)
      set(_depth 1)
    elseif(_active)
      if(_line MATCHES "^[ \t]*if[ \t]*\\(")
        math(EXPR _depth "${_depth} + 1")
      elseif(_line MATCHES "^[ \t]*endif([ \t]*\\(|[ \t]*$)")
        math(EXPR _depth "${_depth} - 1")
        if(_depth EQUAL 0)
          set(_active FALSE)
        endif()
      endif()
    else()
      string(APPEND _native "${_line}\n")
    endif()
  endforeach()
  if(_active)
    message(FATAL_ERROR "unbalanced ENABLE_CIDER_EXTERNAL block")
  endif()
  set(${_output} "${_native}" PARENT_SCOPE)
endfunction()

# Collect balanced if(...) regions whose opening condition mentions
# ENABLE_CIDER.  This prevents unrelated Python use elsewhere in a generic
# ABACUS CMakeLists.txt from becoming a false positive.
function(_cider_extract_enabled_blocks _contents _output)
  string(REPLACE "\n" ";" _lines "${_contents}")
  set(_active FALSE)
  set(_depth 0)
  set(_blocks "")
  foreach(_line IN LISTS _lines)
    if(NOT _active AND _line MATCHES "^[ \t]*if[ \t]*\\([^\\)]*ENABLE_CIDER")
      set(_active TRUE)
      set(_depth 1)
      string(APPEND _blocks "${_line}\n")
    elseif(_active)
      string(APPEND _blocks "${_line}\n")
      if(_line MATCHES "^[ \t]*if[ \t]*\\(")
        math(EXPR _depth "${_depth} + 1")
      elseif(_line MATCHES "^[ \t]*endif([ \t]*\\(|[ \t]*$)")
        math(EXPR _depth "${_depth} - 1")
        if(_depth EQUAL 0)
          set(_active FALSE)
        endif()
      endif()
    endif()
  endforeach()
  set(${_output} "${_blocks}" PARENT_SCOPE)
endfunction()

foreach(_file IN LISTS CIDER_BUILD_FILES)
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "native CIDER build boundary file is missing: ${_file}")
  endif()
  file(READ "${_file}" _contents)
  _cider_without_external_blocks("${_contents}" _native_contents)
  _cider_assert_absent("${_file}" "${_native_contents}" _legacy_bridge_patterns)
  _cider_extract_enabled_blocks("${_native_contents}" _cider_blocks)
  _cider_assert_absent("${_file} ENABLE_CIDER block"
                       "${_cider_blocks}"
                       _legacy_python_dependency_patterns)
endforeach()

foreach(_file IN LISTS CIDER_DEDICATED_BUILD_FILES)
  file(READ "${_file}" _contents)
  _cider_assert_absent("${_file}"
                       "${_contents}"
                       _legacy_python_dependency_patterns)
endforeach()

message(STATUS "native CIDER build boundary contains no legacy bridge hooks")
