# Portable post-build release guard for an external-enabled dual CIDER binary.
foreach(_name CIDER_CACHE CIDER_LINK CIDER_BINARY)
  if(NOT DEFINED ${_name} OR NOT EXISTS "${${_name}}")
    message(FATAL_ERROR "${_name} must name an existing file")
  endif()
endforeach()

file(READ "${CIDER_CACHE}" _cache)
foreach(_required
        "ENABLE_CIDER:BOOL=ON"
        "ENABLE_CIDER_EXTERNAL:BOOL=ON"
        "ENABLE_LIBXC:BOOL=ON"
        "CIDERBRIDGE_LIBRARY:FILEPATH=")
  string(FIND "${_cache}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR "dual CIDER release cache is missing ${_required}")
  endif()
endforeach()

file(READ "${CIDER_LINK}" _link)
string(TOLOWER "${_link}" _link_lower)
foreach(_required_link "libciderbridge" "libpython")
  string(FIND "${_link_lower}" "${_required_link}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR "dual CIDER executable link is missing ${_required_link}")
  endif()
endforeach()

file(GET_RUNTIME_DEPENDENCIES
     EXECUTABLES "${CIDER_BINARY}"
     RESOLVED_DEPENDENCIES_VAR _resolved_dependencies
     UNRESOLVED_DEPENDENCIES_VAR _unresolved_dependencies
     CONFLICTING_DEPENDENCIES_PREFIX _conflicting_dependencies)
if(_unresolved_dependencies)
  message(FATAL_ERROR
          "dual CIDER binary has unresolved dependencies: ${_unresolved_dependencies}")
endif()
if(_conflicting_dependencies_FILENAMES)
  message(FATAL_ERROR
          "dual CIDER binary has conflicting dependencies: ${_conflicting_dependencies_FILENAMES}")
endif()
set(_has_bridge FALSE)
set(_has_python FALSE)
foreach(_dependency IN LISTS _resolved_dependencies)
  string(TOLOWER "${_dependency}" _dependency_lower)
  if(_dependency_lower MATCHES "libciderbridge")
    set(_has_bridge TRUE)
  endif()
  if(_dependency_lower MATCHES "libpython")
    set(_has_python TRUE)
  endif()
endforeach()
if(NOT _has_bridge OR NOT _has_python)
  message(FATAL_ERROR
          "dual CIDER runtime closure must contain libciderbridge and libpython")
endif()

execute_process(
  COMMAND "${CIDER_BINARY}" -i
  RESULT_VARIABLE _info_result
  OUTPUT_VARIABLE _info
  ERROR_VARIABLE _info_error)
if(NOT _info_result EQUAL 0)
  message(FATAL_ERROR "dual CIDER build-info command failed: ${_info_error}")
endif()
foreach(_capability "Native CIDER Support:          yes" "External CIDER Bridge:         yes")
  string(FIND "${_info}" "${_capability}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR "dual CIDER build info is missing '${_capability}'")
  endif()
endforeach()

message(STATUS "dual CIDER release artifact contains native and external backends")
