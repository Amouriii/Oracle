# Locate Apple Accelerate (vecLib).
# Usage: find_package(Accelerate REQUIRED)

if(APPLE)
  find_library(ACCELERATE_LIBRARY Accelerate)
  find_path(ACCELERATE_INCLUDE_DIR Accelerate/Accelerate.h
            PATHS /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Headers
                  /System/Library/Frameworks/Accelerate.framework/Headers
            NO_DEFAULT_PATH)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Accelerate DEFAULT_MSG ACCELERATE_LIBRARY)
  if(Accelerate_FOUND AND NOT TARGET Accelerate::Accelerate)
    add_library(Accelerate::Accelerate INTERFACE IMPORTED)
    set_target_properties(Accelerate::Accelerate PROPERTIES
      INTERFACE_LINK_LIBRARIES "${ACCELERATE_LIBRARY}")
  endif()
else()
  set(Accelerate_FOUND FALSE)
endif()
