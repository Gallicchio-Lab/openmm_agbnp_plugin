#---------------------------------------------------
# OpenMM AGBNP Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET (CMAKE_CXX_STANDARD 11)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# set flags for linking on mac
IF(APPLE)
    SET (CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

#build shared GaussVol library
SET(GAUSSVOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gaussvol")
SET(GAUSSVOLLIB_NAME gaussvol)
SET(GAUSSVOLLIB_INCLUDE_DIR ${GAUSSVOLLIB_DIR})
ADD_LIBRARY(${GAUSSVOLLIB_NAME} SHARED "${GAUSSVOL_DIR}/gaussvol.cpp" "${GAUSSVOL_DIR}/gaussvol.h")
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${GAUSSVOLLIB_NAME})

#
#build/Install the main API library
#
SET(AGBNP_PLUGIN_SOURCE_SUBDIRS openmmapi)
SET(AGBNP_LIBRARY_NAME AGBNPPlugin)
SET(SHARED_AGBNP_TARGET ${AGBNP_LIBRARY_NAME})
#These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "gaussvol" "openmmapi/include" "openmmapi/include/internal")
# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)
# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${AGBNP_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)
INCLUDE_DIRECTORIES(BEFORE ${GAUSSVOL_DIR})

# Create the library.
ADD_LIBRARY(${SHARED_AGBNP_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_AGBNP_TARGET}
    PROPERTIES COMPILE_FLAGS "-DAGBNP_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_AGBNP_TARGET} OpenMM)
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_AGBNP_TARGET})
# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing

#ENABLE_TESTING()
#ADD_SUBDIRECTORY(serialization/tests)

#
# Build/Install for the Reference platform
#
ADD_SUBDIRECTORY(platforms/reference)

#to find FindOpenCL.cmake etc.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#
# Build/Install for the OpenCL platform (incomplete)
#
FIND_PACKAGE(OpenCL QUIET)
IF(OPENCL_FOUND)
     SET(AGBNP_BUILD_OPENCL_LIB ON CACHE BOOL "Build implementation for OpenCL")
ELSE(OPENCL_FOUND)
     SET(AGBNP_BUILD_OPENCL_LIB OFF CACHE BOOL "Build implementation for OpenCL")
ENDIF(OPENCL_FOUND)
IF(AGBNP_BUILD_OPENCL_LIB)
     ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(AGBNP_BUILD_OPENCL_LIB)

# # Build the Python API
FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(AGBNP_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(AGBNP_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(AGBNP_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(AGBNP_BUILD_PYTHON_WRAPPERS)
