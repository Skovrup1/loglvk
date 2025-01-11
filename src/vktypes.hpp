#pragma once

#include "core.hpp"

struct AllocImage {
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
    VkExtent3D extent;
    VkFormat format;
};

struct AllocBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct GPUMeshBuffers {
    AllocBuffer index_buffer;
    AllocBuffer vertex_buffer;
    VkDeviceAddress vertex_device_address;
};

struct GPUDrawPushConstants {
    glm::mat4 mvp;
    VkDeviceAddress vertex_buffer;
};

namespace util {

void transition_image_color(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout);
void transition_image_depth(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout);

} // namespace util
