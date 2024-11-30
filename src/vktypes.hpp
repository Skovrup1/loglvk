#pragma once

#include "core.hpp"

namespace util {

void transition_image_color(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout);
void transition_image_depth(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout);

} // namespace util
