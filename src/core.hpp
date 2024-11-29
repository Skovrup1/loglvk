#pragma once

#include <limits>
#include <cstdint>

#include <volk.h>

#include <vk_mem_alloc.h>

#include <VkBootstrap.h>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using isize = std::ptrdiff_t;

using f32 = float;
using f64 = double;

#ifdef NDEBUG
constexpr bool DEBUG_BUILD = false;
#else
constexpr bool DEBUG_BUILD = true;
#endif

constexpr u64 MAX_TIMEOUT_DURATION = std::numeric_limits<u64>::max();

void vk_check(VkResult err);
