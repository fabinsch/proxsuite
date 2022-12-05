# Use floats instead of integers
option(BUILD_QDLDL_WITH_FLOAT "Use float numbers instead of doubles" OFF)
message(STATUS "QDLDL: Use double. Floats are ${BUILD_QDLDL_WITH_FLOAT}")

# Use long integers for indexing
option(BUILD_QDLDL_WITH_LONG "Use long integers (64bit) for indexing" ON)
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
  message(STATUS "Disabling long integers (64bit) on 32bit machine")
  set(BUILD_QDLDL_WITH_LONG OFF)
endif()
message(STATUS "QDLDL: Long integers (64bit) are ${BUILD_QDLDL_WITH_LONG}")

# numeric types
if(BUILD_QDLDL_WITH_FLOAT)
  set(QDLDL_FLOAT_TYPE "float")
else()
  set(QDLDL_FLOAT_TYPE "double")
endif()

if(BUILD_QDLDL_WITH_LONG)
  set(QDLDL_INT_TYPE "long long")
  set(QDLDL_INT_TYPE_MAX "LLONG_MAX")

else()
  set(QDLDL_INT_TYPE "int")
  set(QDLDL_INT_TYPE_MAX "INT_MAX")
endif()

# boolean type is always unsigned char for now, since _Bool does not exist in
# C89 and we want to avoid interoperability problems when calling QDLDL from C++
set(QDLDL_BOOL_TYPE "unsigned char")

configure_file(
  ${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h.in
  ${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h
  NEWLINE_STYLE LF)
set(amd_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_valid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/SuiteSparse_config.c)

set(qdldl_interface_src
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cs.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl.c)

# Create object library for linear system solver interface
add_library(linsys_qdldl SHARED ${qdldl_interface_src} ${amd_sources})
if(BUILD_QDLDL_WITH_FLOAT)
  target_compile_definitions(linsys_qdldl PUBLIC QDLDL_WITH_FLOAT)
endif()
if(BUILD_QDLDL_WITH_LONG)
  target_compile_definitions(linsys_qdldl PUBLIC QDLDL_WITH_LONG)
endif()
