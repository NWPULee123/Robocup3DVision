cmake_minimum_required(VERSION 2.8)
project(calibration)

find_package(OpenCV REQUIRED)

aux_source_directory (src1 SRC_LIST1)
aux_source_directory (src2 SRC_LIST2)
aux_source_directory (src3 SRC_LIST3)

add_executable(create_board
	${SRC_LIST1}
)

add_executable(calibration
	${SRC_LIST2}
)
target_link_libraries(create_board
	${OpenCV_LIBS}
)

target_link_libraries(calibration
	${OpenCV_LIBS}
)

add_executable(ex_cali
	${SRC_LIST3}
)
target_link_libraries(ex_cali
	${OpenCV_LIBS}
)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
