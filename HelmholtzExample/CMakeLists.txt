
set(HelmholtzExample_SRCS Helmholtz.cpp)

add_executable(HelmholtzExample ${HelmholtzExample_SRCS} )

target_link_libraries(HelmholtzExample pz)
target_include_directories(HelmholtzExample PUBLIC ${PZ_INCLUDE_DIRS})