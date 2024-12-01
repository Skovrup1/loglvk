#include "core.hpp"

#include "deletion_queue.hpp"
#include "pipeline.hpp"
#include "shader.hpp"
#include "vktypes.hpp"
#include <span>

struct FrameData {
    VkSemaphore swapchain_semaphore, render_semaphore;
    VkFence render_fence;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    DeletionQueue deletion_queue;
};

u32 w_width = 800;
u32 w_height = 600;
SDL_Window *window = nullptr;

constexpr u32 FRAME_OVERLAP = 2;

VkInstance instance;
VkSurfaceKHR surface;
VkPhysicalDevice physical_device;
VkDevice device;
VkQueue graphics_queue;
u32 graphics_queue_family;

bool swapchain_ok = false;
VkSwapchainKHR swapchain;
VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
VkExtent2D swapchain_extent;
std::vector<VkImage> swapchain_images;
std::vector<VkImageView> swapchain_image_views;

VmaAllocator allocator;

AllocImage draw_image;
VkExtent2D render_extent;
AllocImage depth_image;

FrameData frames[FRAME_OVERLAP];
u32 frame_id;

VkCommandBuffer imm_command_buffer;
VkCommandPool imm_command_pool;
VkFence imm_fence;

VkDescriptorPool pool;
VkDescriptorSetLayout descriptor_layout;
VkDescriptorSet descriptor_set;

VkPipeline graphics_pipeline;
VkPipelineLayout pipeline_layout;

GPUMeshBuffers rectangle;

DeletionQueue deletion_queue;

void imm_submit(std::function<void(VkCommandBuffer cmd)> &&func) {
    vk_check(vkResetFences(device, 1, &imm_fence));
    vk_check(vkResetCommandBuffer(imm_command_buffer, 0));

    VkCommandBuffer cmd = imm_command_buffer;

    VkCommandBufferBeginInfo cmd_begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vk_check(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    func(cmd);

    vk_check(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };

    VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_info,
    };

    vk_check(vkQueueSubmit2(graphics_queue, 1, &submit, imm_fence));
    vk_check(
        vkWaitForFences(device, 1, &imm_fence, true, MAX_TIMEOUT_DURATION));
}

AllocatedBuffer create_buffer(size_t alloc_size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info{.sType =
                                       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = alloc_size,
                                   .usage = usage};

    VmaAllocationCreateInfo alloc_info{
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    AllocatedBuffer buffer;
    vk_check(vmaCreateBuffer(allocator, &buffer_info, &alloc_info,
                             &buffer.buffer, &buffer.allocation, &buffer.info));

    return buffer;
}

AllocatedBuffer create_buffer_staging(size_t alloc_size,
                                      VkBufferUsageFlags usage) {
    VkBufferCreateInfo buffer_info{.sType =
                                       VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = alloc_size,
                                   .usage = usage};

    VmaAllocationCreateInfo alloc_info{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    AllocatedBuffer buffer;
    vk_check(vmaCreateBuffer(allocator, &buffer_info, &alloc_info,
                             &buffer.buffer, &buffer.allocation, &buffer.info));

    return buffer;
}

void destroy_buffer(AllocatedBuffer buffer) {
    vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffers upload_mesh(std::span<u32> indices, std::span<Vertex> vertices) {
    usize const index_buffer_bsize = indices.size() * sizeof(u32);
    usize const vertex_buffer_bsize = vertices.size() * sizeof(Vertex);

    GPUMeshBuffers mesh;

    mesh.vertex_buffer = create_buffer(
        vertex_buffer_bsize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkBufferDeviceAddressInfo address_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = mesh.vertex_buffer.buffer,
    };
    mesh.vertex_device_address =
        vkGetBufferDeviceAddress(device, &address_info);

    mesh.index_buffer =
        create_buffer(index_buffer_bsize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    AllocatedBuffer staging =
        create_buffer_staging(vertex_buffer_bsize + index_buffer_bsize,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    void *data = staging.info.pMappedData;

    memcpy(data, vertices.data(), vertex_buffer_bsize);
    memcpy((std::byte *)data + vertex_buffer_bsize, indices.data(),
           vertex_buffer_bsize);

    imm_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertex_buffer_bsize,
        };

        vkCmdCopyBuffer(cmd, staging.buffer, mesh.vertex_buffer.buffer, 1,
                        &vertexCopy);

        VkBufferCopy indexCopy{
            .srcOffset = vertex_buffer_bsize,
            .dstOffset = 0,
            .size = index_buffer_bsize,
        };

        vkCmdCopyBuffer(cmd, staging.buffer, mesh.index_buffer.buffer, 1,
                        &indexCopy);
    });

    destroy_buffer(staging);

    return mesh;
}

bool init_logger() {
    auto logger = spdlog::stderr_color_mt("stderr");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::trace);

    return true;
}

bool init_window() {
    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("Engine", w_width, w_height,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        SPDLOG_CRITICAL("Failed to select SDL window. Error: {}\n",
                        SDL_GetError());
        return false;
    }

    return true;
}

bool init_vulkan() {
    vk_check(volkInitialize());

    u32 extension_len;
    char const *const *sdl_extensions =
        SDL_Vulkan_GetInstanceExtensions(&extension_len);
    std::vector<std::string_view> extensions(sdl_extensions,
                                             sdl_extensions + extension_len);

    for (auto const &str : extensions) {
        SPDLOG_DEBUG("Extension: {}", str);
    }

    vkb::InstanceBuilder builder;
    auto inst_ret = builder.set_app_name("Example Vulkan Application")
                        .request_validation_layers()
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .enable_extension(extensions.data()->data())
                        .build();
    if (!inst_ret) {
        SPDLOG_CRITICAL("Failed to create Vulkan instance. Error: {}\n",
                        inst_ret.error().message());
        return false;
    }
    vkb::Instance vkb_inst = inst_ret.value();
    instance = vkb_inst;
    volkLoadInstance(instance);

    SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface);

    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // VkPhysicalDeviceFeatures features{};
    // features.sparseResidencyBuffer = true;

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    auto phys_ret = selector.set_surface(surface)
                        .set_minimum_version(1, 3)
                        .set_required_features_12(features12)
                        .set_required_features_13(features13)
                        .select();
    if (!phys_ret) {
        SPDLOG_CRITICAL("Failed to select Vulkan Physical Device. Error: {}\n",
                        phys_ret.error().message());
        return false;
    }
    vkb::PhysicalDevice vkb_phys = phys_ret.value();
    physical_device = vkb_phys;

    vkb::DeviceBuilder device_builder{phys_ret.value()};
    // automatically propagate needed data from instance & physical device
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
        SPDLOG_CRITICAL("Failed to create Vulkan device. Error: {}\n",
                        dev_ret.error().message());
        return false;
    }
    vkb::Device vkb_device = dev_ret.value();
    device = vkb_device;
    volkLoadDevice(device);

    // Get the graphics queue with a helper function
    auto graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
    if (!graphics_queue_ret) {
        SPDLOG_CRITICAL("Failed to get graphics queue. Error: {}\n",
                        graphics_queue_ret.error().message());
        return false;
    }
    graphics_queue = graphics_queue_ret.value();

    auto graphics_queue_family_ret =
        vkb_device.get_queue_index(vkb::QueueType::graphics);
    if (!graphics_queue_family_ret) {
        SPDLOG_DEBUG("Failed to get graphics queue family. Error: {}\n",
                     graphics_queue_family_ret.error().message());
    }
    graphics_queue_family = graphics_queue_family_ret.value();

    return true;
}

void init_vma() {
    VmaVulkanFunctions vma_vulkan_func{};
    vma_vulkan_func.vkAllocateMemory = vkAllocateMemory;
    vma_vulkan_func.vkBindBufferMemory = vkBindBufferMemory;
    vma_vulkan_func.vkBindImageMemory = vkBindImageMemory;
    vma_vulkan_func.vkCreateBuffer = vkCreateBuffer;
    vma_vulkan_func.vkCreateImage = vkCreateImage;
    vma_vulkan_func.vkDestroyBuffer = vkDestroyBuffer;
    vma_vulkan_func.vkDestroyImage = vkDestroyImage;
    vma_vulkan_func.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vma_vulkan_func.vkFreeMemory = vkFreeMemory;
    vma_vulkan_func.vkGetBufferMemoryRequirements =
        vkGetBufferMemoryRequirements;
    vma_vulkan_func.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vma_vulkan_func.vkGetPhysicalDeviceMemoryProperties =
        vkGetPhysicalDeviceMemoryProperties;
    vma_vulkan_func.vkGetPhysicalDeviceProperties =
        vkGetPhysicalDeviceProperties;
    vma_vulkan_func.vkInvalidateMappedMemoryRanges =
        vkInvalidateMappedMemoryRanges;
    vma_vulkan_func.vkMapMemory = vkMapMemory;
    vma_vulkan_func.vkUnmapMemory = vkUnmapMemory;
    vma_vulkan_func.vkCmdCopyBuffer = vkCmdCopyBuffer;

    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = device;
    allocator_info.instance = instance;
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocator_info.pVulkanFunctions = &vma_vulkan_func;
    vmaCreateAllocator(&allocator_info, &allocator);
}

void destroy_swapchain(VkSwapchainKHR swapchain) {
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    for (usize i = 0; i < swapchain_image_views.size(); i++) {
        vkDestroyImageView(device, swapchain_image_views[i], nullptr);
    }
}

void create_swapchain(u32 width, u32 height, VkSwapchainKHR old_swapchain) {
    vkb::SwapchainBuilder builder{physical_device, device, surface};

    if (old_swapchain) {
        builder.set_old_swapchain(old_swapchain);
    }

    auto swap_ret = builder
                        .set_desired_format(VkSurfaceFormatKHR{
                            .format = swapchain_format,
                            .colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR})
                        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                        .set_desired_extent(width, height)
                        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                        .build();
    if (!swap_ret) {
        SPDLOG_CRITICAL("Failed to create Vulkan swapchain. Error: {}\n",
                        swap_ret.error().message());
    }
    vkb::Swapchain vkb_swapchain = swap_ret.value();

    if (old_swapchain) {
        destroy_swapchain(old_swapchain);
    }

    swapchain = vkb_swapchain;
    swapchain_extent = vkb_swapchain.extent;
    swapchain_images = vkb_swapchain.get_images().value();
    swapchain_image_views = vkb_swapchain.get_image_views().value();

    swapchain_ok = true;
}

void resize_swapchain() {
    vkDeviceWaitIdle(device);

    int w, h;
    SDL_GetWindowSize(window, &w, &h);

    w_width = w;
    w_height = h;

    if (w > 0 && h > 0) {
        create_swapchain(w, h, swapchain);
    }
}

void init_swapchain() {
    create_swapchain(w_width, w_height, swapchain);

    VkExtent3D extent{.width = w_width, .height = w_height, .depth = 1};

    draw_image = {.extent = extent, .format = VK_FORMAT_R16G16B16A16_SFLOAT};
    depth_image = {.extent = extent, .format = VK_FORMAT_D32_SFLOAT};

    VkImageUsageFlags draw_usage_flags = 0;
    draw_usage_flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    /*    VkImageUsageFlags depth_usage_flags = 0;
        depth_usage_flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        */

    VkImageCreateInfo draw_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = draw_image.format,
        .extent = draw_image.extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = draw_usage_flags,
    };

    /*    VkImageCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = depth_image.format,
            .extent = depth_image.extent,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = depth_usage_flags,
        };
        */

    VmaAllocationCreateInfo draw_alloc_info{
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};

    /*    VmaAllocationCreateInfo depth_alloc_info{
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
            */

    vk_check(vmaCreateImage(allocator, &draw_info, &draw_alloc_info,
                            &draw_image.image, &draw_image.allocation,
                            nullptr));

    /*    vk_check(vmaCreateImage(allocator, &depth_info, &depth_alloc_info,
                                &depth_image.image, &depth_image.allocation,
                                nullptr));
                                */

    VkImageViewCreateInfo draw_view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = draw_image.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = draw_image.format,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
    };

    /*VkImageViewCreateInfo depth_view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = depth_image.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = depth_image.format,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
    };
    */

    vk_check(
        vkCreateImageView(device, &draw_view_info, nullptr, &draw_image.view));

    /*vk_check(vkCreateImageView(device, &depth_view_info, nullptr,
                               &depth_image.view));
                               */
};

void init_commands() {
    VkCommandPoolCreateInfo cmd_pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_queue_family};

    for (usize i = 0; i < FRAME_OVERLAP; i++) {
        vk_check(vkCreateCommandPool(device, &cmd_pool_info, nullptr,
                                     &frames[i].command_pool));

        VkCommandBufferAllocateInfo cmd_alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = frames[i].command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        vk_check(vkAllocateCommandBuffers(device, &cmd_alloc_info,
                                          &frames[i].command_buffer));
    }

    vk_check(vkCreateCommandPool(device, &cmd_pool_info, nullptr,
                                 &imm_command_pool));

    VkCommandBufferAllocateInfo imm_cmd_alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = imm_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    vk_check(vkAllocateCommandBuffers(device, &imm_cmd_alloc_info,
                                      &imm_command_buffer));
}

void init_sync_structs() {
    VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VkSemaphoreCreateInfo semaphore_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    for (usize i = 0; i < FRAME_OVERLAP; i++) {
        vk_check(vkCreateFence(device, &fence_info, nullptr,
                               &frames[i].render_fence));
        vk_check(vkCreateSemaphore(device, &semaphore_info, nullptr,
                                   &frames[i].swapchain_semaphore));
        vk_check(vkCreateSemaphore(device, &semaphore_info, nullptr,
                                   &frames[i].render_semaphore));
    }

    vk_check(vkCreateFence(device, &fence_info, nullptr, &imm_fence));
}

void init_descriptors() {}

void init_compute_pipeline() {}

void init_graphic_pipeline() {
    std::string vert_path = "shaders/hello.vert.glsl.spv";
    VkShaderModule vert_shader = load_shader_module(device, vert_path.c_str());
    if (!vert_shader) {
        SPDLOG_CRITICAL("failed to load vert shader module from {}", vert_path);
    }

    std::string frag_path = "shaders/hello.frag.glsl.spv";
    VkShaderModule frag_shader = load_shader_module(device, frag_path.c_str());
    if (!frag_shader) {
        SPDLOG_CRITICAL("failed to load frag shader module from {}", frag_path);
    }

    VkPushConstantRange buffer_range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants),
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &buffer_range,
    };

    vk_check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                    &pipeline_layout));

    PipelineBuilder builder{};
    graphics_pipeline =
        builder.set_pipeline_layout(pipeline_layout)
            .set_shaders(vert_shader, frag_shader)
            .set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .set_polygon_mode(VK_POLYGON_MODE_FILL)
            .set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE)
            .set_multisampling_none()
            .disable_blending()
            .disable_depthtest()
            .set_color_attachment_format(draw_image.format)
            .set_depth_format(VK_FORMAT_UNDEFINED)
            .build(device);

    vkDestroyShaderModule(device, frag_shader, nullptr);
    vkDestroyShaderModule(device, vert_shader, nullptr);
}

void init_pipelines() {
    // init_compute_pipeline();
    init_graphic_pipeline();
}

void init_imgui() {}

void init_default_data() {
    std::array<Vertex, 4> rect_vertices;

    rect_vertices[0].position = {0.5, -0.5, 0};
    rect_vertices[1].position = {0.5, 0.5, 0};
    rect_vertices[2].position = {-0.5, -0.5, 0};
    rect_vertices[3].position = {-0.5, 0.5, 0};

    rect_vertices[0].color = {0, 0, 0, 1};
    rect_vertices[1].color = {0.5, 0.5, 0.5, 1};
    rect_vertices[2].color = {1, 0, 0, 1};
    rect_vertices[3].color = {0, 1, 0, 1};

    std::array<u32, 6> rect_indices;

    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;

    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;

    rectangle = upload_mesh(rect_indices, rect_vertices);

    deletion_queue.push_function([&]() {
        destroy_buffer(rectangle.index_buffer);
        destroy_buffer(rectangle.vertex_buffer);
    });
}

FrameData &get_current_frame() { return frames[frame_id % FRAME_OVERLAP]; }

void render_compute(VkCommandBuffer cmd) {
    float flash = std::abs(std::sin(frame_id / 120.f));
    VkClearColorValue clear_value = {{0.0f, 0.0f, flash, 1.0f}};

    VkImageSubresourceRange clear_range{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    vkCmdClearColorImage(cmd, draw_image.image, VK_IMAGE_LAYOUT_GENERAL,
                         &clear_value, 1, &clear_range);
}

void render_triangle(VkCommandBuffer cmd) {
    VkRenderingAttachmentInfo color_attach{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = draw_image.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };

    VkRenderingInfo render_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{VkOffset2D{0, 0}, render_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attach,
    };

    vkCmdBeginRendering(cmd, &render_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    VkViewport viewport{
        .x = 0,
        .y = 0,
        .width = static_cast<f32>(render_extent.width),
        .height = static_cast<f32>(render_extent.height),
        .minDepth = 0.f,
        .maxDepth = 1.f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {.x = 0, .y = 0},
        .extent = {.width = render_extent.width,
                   .height = render_extent.height},
    };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    GPUDrawPushConstants push_constants;
    push_constants.worldMatrix = glm::mat4{1.f};
    push_constants.vertexBuffer = rectangle.vertex_device_address;

    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(GPUDrawPushConstants), &push_constants);

    vkCmdBindIndexBuffer(cmd, rectangle.index_buffer.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
}

void copy_image_to_image(VkCommandBuffer cmd, VkImage src_image,
                         VkImage dst_image, VkExtent2D src_extent,
                         VkExtent2D dst_extent) {
    VkImageBlit2 region{
        .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
    };

    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = 0;

    region.srcOffsets[1].x = src_extent.width;
    region.srcOffsets[1].y = src_extent.height;
    region.srcOffsets[1].z = 1;

    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount = 1;
    region.dstSubresource.mipLevel = 0;

    region.dstOffsets[1].x = dst_extent.width;
    region.dstOffsets[1].y = dst_extent.height;
    region.dstOffsets[1].z = 1;

    VkBlitImageInfo2 blit_info{
        .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
        .srcImage = src_image,
        .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .dstImage = dst_image,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 1,
        .pRegions = &region,
        .filter = VK_FILTER_LINEAR,
    };

    vkCmdBlitImage2(cmd, &blit_info);
}

void render() {
    vk_check(vkWaitForFences(device, 1, &get_current_frame().render_fence, true,
                             MAX_TIMEOUT_DURATION));

    u32 swapchain_image_index;
    if (swapchain_ok) {
        VkResult r =
            vkAcquireNextImageKHR(device, swapchain, MAX_TIMEOUT_DURATION,
                                  get_current_frame().swapchain_semaphore,
                                  nullptr, &swapchain_image_index);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain_ok = false;
            SPDLOG_DEBUG("out of date, acquire\n");
        } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            SPDLOG_DEBUG("failed to acquire next image: {}\n",
                         string_vkresult(r));
        }
    }
    if (!swapchain_ok) {
        return;
    }

    vk_check(vkResetFences(device, 1, &get_current_frame().render_fence));

    // cleanup frames from previous iteration
    VkCommandBuffer cmd = get_current_frame().command_buffer;

    vk_check(vkResetCommandBuffer(cmd, 0));

    render_extent.width = draw_image.extent.width;
    render_extent.height = draw_image.extent.height;

    VkCommandBufferBeginInfo cmd_begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vk_check(vkBeginCommandBuffer(cmd, &cmd_begin_info));

    util::transition_image_color(cmd, draw_image.image,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL);

    // render_compute(cmd);

    util::transition_image_color(cmd, draw_image.image, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    render_triangle(cmd);

    util::transition_image_color(cmd, draw_image.image,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    util::transition_image_color(cmd, swapchain_images[swapchain_image_index],
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    copy_image_to_image(cmd, draw_image.image,
                        swapchain_images[swapchain_image_index], render_extent,
                        swapchain_extent);

    util::transition_image_color(cmd, swapchain_images[swapchain_image_index],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    vk_check(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmd_submit_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };

    VkSemaphoreSubmitInfo semaphore_wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = get_current_frame().swapchain_semaphore,
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };

    VkSemaphoreSubmitInfo semaphore_signal_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = get_current_frame().render_semaphore,
        .value = 1,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
    };

    VkSubmitInfo2 submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &semaphore_wait_info,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmd_submit_info,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &semaphore_signal_info,
    };

    vk_check(vkQueueSubmit2(graphics_queue, 1, &submit_info,
                            get_current_frame().render_fence));

    VkPresentInfoKHR present_info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &get_current_frame().render_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_image_index,
    };

    VkResult r = vkQueuePresentKHR(graphics_queue, &present_info);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_ok = false;
        SPDLOG_DEBUG("out of date acquire\n");
    } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        SPDLOG_DEBUG("failed to present an image: {}\n", string_vkresult(r));
        abort();
    }

    frame_id++;
}

void free_on_exit() {
    vkDeviceWaitIdle(device);

    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyPipeline(device, graphics_pipeline, nullptr);

    for (u32 i = 0; i < FRAME_OVERLAP; i++) {
        vkDestroyCommandPool(device, frames[i].command_pool, nullptr);

        vkDestroyFence(device, frames[i].render_fence, nullptr);
        vkDestroySemaphore(device, frames[i].render_semaphore, nullptr);
        vkDestroySemaphore(device, frames[i].swapchain_semaphore, nullptr);

        frames[i].deletion_queue.flush();
    }

    // vkDestroyDescriptorPool(device, pool, nullptr);
    // vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    
    deletion_queue.flush();

    vkDestroyFence(device, imm_fence, nullptr);

    vkDestroyCommandPool(device, imm_command_pool, nullptr);

    vkDestroyImageView(device, draw_image.view, nullptr);
    vmaDestroyImage(allocator, draw_image.image, draw_image.allocation);

    vkDestroyImageView(device, depth_image.view, nullptr);
    vmaDestroyImage(allocator, depth_image.image, depth_image.allocation);

    vmaDestroyAllocator(allocator);
}

int main() {
    init_logger();

    init_window();

    init_vulkan();

    init_vma();

    init_swapchain();

    init_commands();

    init_sync_structs();

    // init_descriptors();

    init_pipelines();

    // init_imgui();

    init_default_data();

    bool is_running = true;
    while (is_running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
                is_running = false;
                break;
            case SDL_EVENT_WINDOW_RESIZED:
                resize_swapchain();
            case SDL_EVENT_KEY_DOWN:
                switch (e.key.key) {
                case SDLK_ESCAPE:
                    is_running = false;
                    break;
                }
            }
        }

        render();
    }

    free_on_exit();
}
