
set(HCurlSphere_SRCS HCurlSphere.cpp)

add_executable(HCurlSphere ${HCurlSphere_SRCS} )

target_link_libraries(HCurlSphere pz)
target_include_directories(HCurlSphere PUBLIC ${PZ_INCLUDE_DIRS})