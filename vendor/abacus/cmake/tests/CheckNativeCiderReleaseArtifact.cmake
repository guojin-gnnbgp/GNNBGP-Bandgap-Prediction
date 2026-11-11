# Portable post-build release guard for an ENABLE_CIDER executable.
#
# Required command-line definitions:
#   CIDER_CACHE  - configured CMakeCache.txt
#   CIDER_LINK   - final executable link command (link.txt)
#   CIDER_BINARY - built or installed executable

foreach(_name CIDER_CACHE CIDER_LINK CIDER_BINARY)
  if(NOT DEFINED ${_name} OR NOT EXISTS "${${_name}}")
    message(FATAL_ERROR "${_name} must name an existing file")
  endif()
endforeach()

file(STRINGS "${CIDER_CACHE}" _cache_lines)
string(JOIN "\n" _cache ${_cache_lines})
foreach(_required
        "ENABLE_CIDER:BOOL=ON"
        "ENABLE_LIBXC:BOOL=ON")
  string(FIND "${_cache}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR "native CIDER release cache is missing ${_required}")
  endif()
endforeach()

string(FIND "${_cache}" "ENABLE_CIDER_EXTERNAL:BOOL=ON" _external_offset)
if(NOT _external_offset EQUAL -1)
  message(FATAL_ERROR
          "native-only CIDER release guard rejects ENABLE_CIDER_EXTERNAL=ON")
endif()

foreach(_line IN LISTS _cache_lines)
  if(_line MATCHES "^([^#/:=]+)(:[^=]*)?=")
    set(_cache_key "${CMAKE_MATCH_1}")
    string(TOLOWER "${_cache_key}" _cache_name)
    if(_cache_name MATCHES "python3|numpy|ciderpress|ciderbridge")
      message(FATAL_ERROR
              "native CIDER release cache contains prohibited key ${_cache_key}")
    endif()
  endif()
endforeach()

file(READ "${CIDER_LINK}" _link)
string(TOLOWER "${_link}" _link_lower)
foreach(_legacy_link
        "libpython"
        "-lpython"
        "numpy"
        "ciderpress"
        "libciderbridge"
        "-lciderbridge")
  string(FIND "${_link_lower}" "${_legacy_link}" _offset)
  if(NOT _offset EQUAL -1)
    message(FATAL_ERROR "native CIDER executable link contains ${_legacy_link}")
  endif()
endforeach()

file(GET_RUNTIME_DEPENDENCIES
     EXECUTABLES "${CIDER_BINARY}"
     RESOLVED_DEPENDENCIES_VAR _resolved_dependencies
     UNRESOLVED_DEPENDENCIES_VAR _unresolved_dependencies
     CONFLICTING_DEPENDENCIES_PREFIX _conflicting_dependencies)
if(_unresolved_dependencies)
  message(FATAL_ERROR
          "native CIDER binary has unresolved dependencies: ${_unresolved_dependencies}")
endif()
if(_conflicting_dependencies_FILENAMES)
  message(FATAL_ERROR
          "native CIDER binary has conflicting dependencies: ${_conflicting_dependencies_FILENAMES}")
endif()
if(NOT _resolved_dependencies)
  message(FATAL_ERROR "native CIDER binary dependency inspection returned no libraries")
endif()
foreach(_dependency IN LISTS _resolved_dependencies)
  string(TOLOWER "${_dependency}" _dependency_lower)
  foreach(_legacy_dynamic "libpython" "numpy" "libciderbridge" "ciderpress")
    string(FIND "${_dependency_lower}" "${_legacy_dynamic}" _offset)
    if(NOT _offset EQUAL -1)
      message(FATAL_ERROR
              "native CIDER runtime dependency ${_dependency} contains ${_legacy_dynamic}")
    endif()
  endforeach()
endforeach()

message(STATUS "native CIDER release artifact has no Python or bridge dependency")
