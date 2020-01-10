
set(HCurlProjection_SRCS HCurlProjection.cpp)

add_executable(HCurlProjection ${HCurlProjection_SRCS} )

target_link_libraries(HCurlProjection pz)
target_include_directories(HCurlProjection PUBLIC ${PZ_INCLUDE_DIRS})