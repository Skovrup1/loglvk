cmake_minimum_required(VERSION 3.20)

project(LOGLVK VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

file(GLOB_RECURSE LOGLVK_HEADERS "include/*.hpp")
file(GLOB_RECURSE LOGLVK_SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${LOGLVK_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${LOGLVK_HEADERS})

find_package(Vulkan REQUIRED)

add_subdirectory(modules/volk)
target_link_libraries(${PROJECT_NAME} PRIVATE volk)

add_subdirectory(modules/VulkanMemoryAllocator)
target_link_libraries(${PROJECT_NAME} PRIVATE GPUOpen::VulkanMemoryAllocator)

add_subdirectory(modules/vk-bootstrap)
target_link_libraries(${PROJECT_NAME} PRIVATE vk-bootstrap::vk-bootstrap)

add_subdirectory(modules/glm)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

add_subdirectory(modules/SDL)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)

add_subdirectory(modules/spdlog)
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)

#add_subdirectory(modules/fastgltf)
#target_link_libraries(${PROJECT_NAME} PRIVATE fastgltf::fastgltf)

target_compile_options(${PROJECT_NAME} PUBLIC
    -Wall
    -Wwrite-strings
    -Wstrict-prototypes
    -Wsign-compare
    -fno-common
    -Wtype-limits
    -Wformat-nonliteral
    -Wmissing-declarations
    #-fsanitize=address
    #-fno-omit-frame-pointer
    -pipe
    #-march=native
    #-O3
    -g
    -O0
)

#target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
