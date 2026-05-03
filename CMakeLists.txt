cmake_minimum_required(VERSION 3.18)
project(MyCudaSFMLProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(CMAKE_CUDA_COMPILER "C:/.../nvcc.exe")
    set(SFML_DIR "C:/mycomputer/SFML-3.0.0/lib/cmake/SFML")
endif()

add_executable(MyApp src/main.cpp src/simulation.cu src/particle.h src/fps.h)

# SFML 3 find & link
find_package(SFML COMPONENTS Graphics REQUIRED)
target_link_libraries(MyApp PRIVATE SFML::Graphics)

# CUDA options
set_target_properties(MyApp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
