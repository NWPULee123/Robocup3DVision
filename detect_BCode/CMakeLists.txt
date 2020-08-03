cmake_minimum_required (VERSION 2.8)

project (demo)

aux_source_directory (src SRC_LIST)

include_directories (include)

add_executable (demo ${SRC_LIST})

find_package( OpenCV REQUIRED )

target_link_libraries( demo ${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


