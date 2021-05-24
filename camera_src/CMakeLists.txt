cmake_minimum_required(VERSION 2.8)
project( camera )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
add_compile_options(-pedantic  -Wall -Wextra -Wshadow -Wformat=2 -Wunused -march=native)


add_executable( camera camera.cpp )
target_link_libraries( camera ${OpenCV_LIBS} )


