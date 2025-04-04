cmake_minimum_required(VERSION 2.8.3)
project(shape-fitting)

find_package(PCL REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)

###########
## Build ##
###########

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -Wno-comment -Wno-deprecated-declarations")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_STANDARD 14)

# FITTING LIB
add_library(fitting_library
  src/cylinder_fitting.cpp
  src/cylinder_fitting_hough.cpp
  src/sphere_fitting_hough.cpp
  src/plane_fitting.cpp
  src/plane_fitting_ransac.cpp
  src/gaussian_sphere.cpp
  src/fitting_data.cpp
)
target_include_directories(fitting_library
  PUBLIC include
  PRIVATE ${PCL_INCLUDE_DIRS}
)
file(GLOB FITTING_LIBRARY_HEADERS "include/*.h")
set_target_properties(fitting_library PROPERTIES PUBLIC_HEADER "${FITTING_LIBRARY_HEADERS}")
target_link_libraries(fitting_library ${PCL_LIBRARIES})

install(
  TARGETS
    fitting_library
  INCLUDES DESTINATION include
  LIBRARY DESTINATION lib
)

# TESTS
option(BUILD_TESTS "Build tests" OFF)
if (BUILD_TESTS)
    find_package(yaml-cpp REQUIRED)
    include_directories(test ${YAML_CPP_INCLUDE_DIR})

    add_executable(cylinder_generation test/cylinder_generation.cpp)
    target_link_libraries(cylinder_generation ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

    add_executable(sphere_generation test/sphere_generation.cpp)
    target_link_libraries(sphere_generation ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

    add_executable(cylinder_fitting_test test/cylinder_fitting_test.cpp)
    target_link_libraries(cylinder_fitting_test ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} fitting_library)

    add_executable(sphere_fitting_test test/sphere_fitting_test.cpp)
    target_link_libraries(sphere_fitting_test ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} fitting_library)

    add_executable(cylinder_real_data_test test/cylinder_real_data_test.cpp)
    target_link_libraries(cylinder_real_data_test ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} fitting_library)

    install(
      TARGETS
        cylinder_generation
        sphere_generation
        cylinder_fitting_test
        sphere_fitting_test
        cylinder_real_data_test
      RUNTIME DESTINATION bin
    )
endif (BUILD_TESTS)
