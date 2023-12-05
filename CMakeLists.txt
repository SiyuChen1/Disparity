cmake_minimum_required(VERSION 3.26)
project(Disparity)

set(CMAKE_CXX_STANDARD 17)

# Find Package
set( OpenCV_DIR "/home/siyuchen/lib/opencv_all/build" )

find_package(OpenCV REQUIRED)

add_executable(Disparity main.cpp)

if( OpenCV_FOUND )
    message("OpenCV version: ${OpenCV_VERSION}")
    # Additional Include Directories
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    # Additional Library Directories
    link_directories( ${OpenCV_LIB_DIR} )

    # Additional Dependencies
    target_link_libraries( Disparity ${OpenCV_LIBS} )
endif()

