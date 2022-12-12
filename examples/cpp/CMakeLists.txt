#
# Copyright (c) 2022 INRIA
#

add_custom_target(example-cpp)

function(ADD_PROXSUITE_CPP_EXAMPLE EXAMPLE)
  get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
  set(EXAMPLE_TARGET "example-cpp-${EXAMPLE_NAME}")
  add_unit_test(${EXAMPLE_TARGET} "${EXAMPLE}")
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE proxsuite-test-util)

  add_dependencies(example-cpp ${EXAMPLE_TARGET})
endfunction()

file(GLOB_RECURSE ${PROJECT_NAME}_CPP_EXAMPLES *.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND MSVC)
  foreach(example ${${PROJECT_NAME}_CPP_EXAMPLES})
    if(${example} MATCHES "update_dense_qp.cpp" OR ${example} MATCHES "solve_without_api.cpp")
      list(REMOVE_ITEM ${PROJECT_NAME}_CPP_EXAMPLES ${example})
    endif()
  endforeach()
endif()

foreach(EXAMPLE ${${PROJECT_NAME}_CPP_EXAMPLES})
  add_proxsuite_cpp_example(${EXAMPLE})
endforeach(EXAMPLE ${${PROJECT_NAME}_CPP_EXAMPLES})
