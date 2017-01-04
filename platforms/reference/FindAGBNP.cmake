# - Find gaussvol
# Find the gaussvol library and includes
#
#  GVOL_INCLUDES        - where to find gaussvol.h
#  GVOL_LIBRARY         - the main gaussvol library.
#  GVOL_FOUND           - True if gaussvol found.

if (GVOL_INCLUDES)
  # Already in cache, be silent
  set (GVOL_FIND_QUIETLY TRUE)
endif (GVOL_INCLUDES)

find_path (GVOL_INCLUDES gaussvol.h HINTS ${GVOL_DIR})

find_library (GVOL_LIBRARY NAMES gaussvol HINTS ${GVOL_DIR} )

# handle the QUIETLY and REQUIRED arguments and set GVOL_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (GVOL DEFAULT_MSG GVOL_LIBRARY GVOL_INCLUDES)

mark_as_advanced (GVOL_LIBRARY  GVOL_INCLUDES)
