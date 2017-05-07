cmake_minimum_required(VERSION 2.8)
project(Awesome_Project)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
IF(NOT MARCH)
    SET(MARCH native)
ENDIF()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "march type: " ${MARCH})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O0 -g  -march=${MARCH} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g -march=${MARCH} ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# find required packages
find_package(OpenCV 3.0 REQUIRED)
MESSAGE(STATUS "OpenCV include: " ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV libray: " ${OpenCV_LIBRARIES})
find_package(Eigen3 3.1.0 REQUIRED)
MESSAGE(STATUS "Eigen include: " ${EIGEN3_INCLUDE_DIR})
find_package(Pangolin REQUIRED)
find_package(Boost  1.54.0  	     # Minimum or EXACT version e.g. 1.36.0
  REQUIRED             					# Fail with error if Boost is not found
  COMPONENTS serialization          # Boost libraries by their canonical name
)
#find_package(CVD REQUIRED)
find_package(G2O REQUIRED)
#find_package(CSparse REQUIRED)
find_package(TooN REQUIRED)
#find_package(GooglePerftools)
#if(WITH_KINECT)
#    find_package(OpenNI REQUIRED)
#ENDIF()

# include dir
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    #${CVD_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    #${CSPARSE_INCLUDE_DIR}
    ${TooN_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
    src/G2OExtend.cc
    src/CameraIntrinsic.cc
    src/CameraDevice.cc
    src/ImageFrame.cc
    src/VisionTracker.cc
    src/Mapping.cc
    src/Viewer.cc
    src/Timer.cc
    src/Converter.cc
    src/MedianFilter.cc
    src/Initializer.cc
    src/ThirdPartyInit.cc
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${Boost_LIBRARIES}
    #${CVD_LIBRARY}
    #${CSPARSE_LIBRARY}
    ${TooN_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

#add_executable(testCameraDevice
#test/test.cc)
#target_link_libraries(testCameraDevice ${OpenCV_LIBRARIES} ${PROJECT_NAME} )
#
#add_executable(testUndistort
#    test/udistort.cc)
#target_link_libraries(testUndistort ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testFAST
    test/FAST.cc)
target_link_libraries(testFAST ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testFlowLast
    test/flowLast.cc)
target_link_libraries(testFlowLast ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testInitialize
    test/initialize.cc)
target_link_libraries(testInitialize ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testTrackLocal
    test/tracklocal.cc)
target_link_libraries(testTrackLocal ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testInitializeThirdParty
    test/initializethirdparty.cc)
target_link_libraries(testInitializeThirdParty ${OpenCV_LIBRARIES} ${PROJECT_NAME} )

add_executable(testLoadMap
    test/loadmap.cc)
target_link_libraries(testLoadMap ${OpenCV_LIBRARIES} ${PROJECT_NAME} )
