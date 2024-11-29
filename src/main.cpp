#include "core.hpp"

#include <fstream>

struct FrameData {
    VkSemaphore swapchain_semaphore, render_semaphore;
    VkFence render_fence;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
};

struct AllocImage {
    VkImage image = nullptr;
    VkImageView view = nullptr;
    VmaAllocation allocation = nullptr;
    VkExtent3D extent;
    VkFormat format;
};

struct AllocBuffer {
    VkBuffer buffer;
    VmaAllocation alloc;
    VmaAllocationInfo alloc_info;
};

struct GPUPushConstants {
    glm::mat4 world_matrix;
    VkDeviceAddress vertex_buffer;
};

struct GPUMeshBuffers {
    AllocBuffer index_buffer;
    AllocBuffer vertex_buffer;
    VkDeviceAddress vertex_buffer_address;
};

struct Vertex {
    glm::vec3 pos;
    //    float uv_x;
    //   glm::vec3 normal;
    //  float uv_y;
    //    glm::vec4 color;
};

constexpr u32 W_WIDTH = 800;
constexpr u32 W_HEIGHT = 600;
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

VkPipeline pipeline;
VkPipelineLayout pipeline_layout;

GPUMeshBuffers mesh;

bool init_logger() {
    auto logger = spdlog::stderr_color_mt("stderr");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::trace);

    return true;
}

bool init_window() {
    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow("Engine", W_WIDTH, W_HEIGHT,
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

void init_swapchain() {
    create_swapchain(W_WIDTH, W_HEIGHT, swapchain);

    VkExtent3D extent{.width = W_WIDTH, .height = W_HEIGHT, .depth = 1};

    draw_image = {.extent = extent, .format = VK_FORMAT_R16G16B16A16_SFLOAT};
    depth_image = {.extent = extent, .format = VK_FORMAT_D32_SFLOAT};

    VkImageUsageFlags draw_usage_flags = 0;
    draw_usage_flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    draw_usage_flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageUsageFlags depth_usage_flags = 0;
    depth_usage_flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

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

    VkImageCreateInfo depth_info{
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

    VmaAllocationCreateInfo draw_alloc_info{
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};

    VmaAllocationCreateInfo depth_alloc_info{
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};

    vk_check(vmaCreateImage(allocator, &draw_info, &draw_alloc_info,
                            &draw_image.image, &draw_image.allocation,
                            nullptr));

    vk_check(vmaCreateImage(allocator, &depth_info, &depth_alloc_info,
                            &depth_image.image, &depth_image.allocation,
                            nullptr));

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

    VkImageViewCreateInfo depth_view_info{
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

    vk_check(
        vkCreateImageView(device, &draw_view_info, nullptr, &draw_image.view));

    vk_check(vkCreateImageView(device, &depth_view_info, nullptr,
                               &depth_image.view));
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

void init_descriptors() {
    VkDescriptorSetLayoutBinding bind{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };

    VkDescriptorSetLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &bind,
    };

    vk_check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                         &descriptor_layout));

    std::array<VkDescriptorPoolSize, 1> pool_sizes = {
        VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLER,
                             .descriptorCount = 1},
    };

    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 10,
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes = pool_sizes.data()};

    vk_check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool));

    VkDescriptorSetAllocateInfo set_alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_layout};

    vk_check(
        vkAllocateDescriptorSets(device, &set_alloc_info, &descriptor_set));

    // if we need frame descriptor_sets
    /*for (u32 i = 0; i < FRAME_OVERLAP; i++) {
        std::array<VkDescriptorPoolSize, 1> frame_sizes = {
            VkDescriptorPoolSize{.type = VK_DESCRIPTOR_TYPE_SAMPLER,
    .descriptorCount = 1},
        };
    }*/
}

VkShaderModule load_shader_module(VkDevice device, char const *filepath) {
    std::ifstream file{filepath, std::ios::ate | std::ios::binary};

    if (!file.is_open()) {
        return nullptr;
    }

    usize file_size = file.tellg();

    std::vector<u32> buffer(file_size / sizeof(u32));

    file.seekg(0);

    file.read(reinterpret_cast<char *>(buffer.data()), file_size);

    file.close();

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = buffer.size() * sizeof(u32);
    info.pCode = buffer.data();

    VkShaderModule module;
    vk_check(vkCreateShaderModule(device, &info, nullptr, &module));

    return module;
}

void init_pipelines() {
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

    VkPushConstantRange buffer_range{};
    buffer_range.offset = 0;
    buffer_range.size = sizeof(GPUPushConstants);
    buffer_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &buffer_range,
    };

    vk_check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                    &pipeline_layout));

    VkPipelineRenderingCreateInfo render_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &draw_image.format,
        .depthAttachmentFormat = depth_image.format};

    VkPipelineShaderStageCreateInfo shader_stages[]{
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_shader,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_shader,
         .pName = "main"},
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};

    VkPipelineViewportStateCreateInfo viewport_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1};

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.f};

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};

    VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    VkPipelineColorBlendAttachmentState color_blend_attach{};
    VkPipelineColorBlendStateCreateInfo color_blending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attach,
    };

    VkDynamicState state[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = std::size(state),
        .pDynamicStates = state,
    };

    VkGraphicsPipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &render_info,
        .stageCount = std::size(shader_stages),
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout,
    };

    vk_check(vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info,
                                       nullptr, &pipeline));

    vkDestroyShaderModule(device, frag_shader, nullptr);
    vkDestroyShaderModule(device, vert_shader, nullptr);
}

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

void init_default_data() {
    float const vertices[] = {
        // first triangle
        0.5f, 0.5f, 0.0f,   // top right
        0.5f, -0.5f, 0.0f,  // bottom right
        -0.5f, 0.5f, 0.0f,  // top left
                            // second triangle
                            // bottom right
        -0.5f, -0.5f, 0.0f, // bottom left
                            // top left
    };

    float const indices[] = {
        1, 2, 3, 2, 4, 3,
    };

    const usize vertex_buffer_bsize = std::size(vertices) * sizeof(Vertex);
    const usize index_buffer_bsize = std::size(indices) * sizeof(u32);

    {
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = vertex_buffer_bsize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };

        VmaAllocationCreateInfo alloc_info{
            .usage = VMA_MEMORY_USAGE_AUTO,
        };

        vk_check(vmaCreateBuffer(
            allocator, &buffer_info, &alloc_info, &mesh.vertex_buffer.buffer,
            &mesh.vertex_buffer.alloc, &mesh.vertex_buffer.alloc_info));
    }

    VkBufferDeviceAddressInfo device_address_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = mesh.vertex_buffer.buffer,
    };

    mesh.vertex_buffer_address =
        vkGetBufferDeviceAddress(device, &device_address_info);

    {
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = index_buffer_bsize,
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };

        VmaAllocationCreateInfo alloc_info{
            alloc_info.usage = VMA_MEMORY_USAGE_AUTO,
        };

        vk_check(vmaCreateBuffer(
            allocator, &buffer_info, &alloc_info, &mesh.index_buffer.buffer,
            &mesh.index_buffer.alloc, &mesh.index_buffer.alloc_info));
    }

    /*AllocBuffer staging;
    {
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = vertex_buffer_bsize + index_buffer_bsize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        };

        VmaAllocationCreateInfo alloc_create_info{
            .flags = 0,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };

        vk_check(vmaCreateBuffer(allocator, &buffer_info, &alloc_create_info,
                                 &staging.buffer, &staging.alloc,
                                 &staging.alloc_info));
    }*/


    /*
    void *mapped_data;
    vmaMapMemory(allocator, staging.alloc, &mapped_data);
    std::memcpy(mapped_data, vertices, vertex_buffer_bsize);
    std::memcpy(static_cast<char *>(mapped_data) + vertex_buffer_bsize, indices,
                index_buffer_bsize);
    vmaUnmapMemory(allocator, staging.alloc);

    vmaFlushAllocation(allocator, staging.alloc, 0,
                       vertex_buffer_bsize + index_buffer_bsize);
    */

    /*imm_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertex_copy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertex_buffer_bsize,
        };

        vkCmdCopyBuffer(cmd, staging.buffer, mesh.vertex_buffer.buffer, 1,
                        &vertex_copy);

        VkBufferCopy index_copy{
            .srcOffset = vertex_buffer_bsize,
            .dstOffset = 0,
            .size = index_buffer_bsize,
        };

        vkCmdCopyBuffer(cmd, staging.buffer, mesh.index_buffer.buffer, 1,
                        &index_copy);
    });
    */

    //vmaDestroyBuffer(allocator, staging.buffer, staging.alloc);
}

FrameData &get_current_frame() { return frames[frame_id % FRAME_OVERLAP]; }

void render_triangle(VkCommandBuffer cmd) {
    VkRenderingAttachmentInfo color_attach{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = draw_image.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingAttachmentInfo depth_attach{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = draw_image.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {.depth = 0.f}},
    };

    VkRenderingInfo render_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{VkOffset2D{0, 0}, render_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attach,
        .pDepthAttachment = &depth_attach,
    };

    vkCmdBeginRendering(cmd, &render_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

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

    VkDescriptorSet image_set;
    // do stuff

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout, 0, 1, &image_set, 0, nullptr);

    GPUPushConstants push_constants{
        .world_matrix = glm::mat4{},
        //.vertex_buffer = vertices,
    };
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(GPUPushConstants), &push_constants);

    // vkCmdBindIndexBuffer(cmd, vertices, 0, VK_INDEX_TYPE_UINT32);

    // vkCmdDrawIndexed(cmd, std::size(vertices), 1, 0, 0, 0);

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
            SPDLOG_DEBUG("failed to acquire next image\n");
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

    // draw_image
    // from: VK_IMAGE_LAYOUT_UNDEFINED
    // to: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    {
        VkImageMemoryBarrier2 image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            // where
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            // how
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = draw_image.image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                //.levelCount = VK_REMAINING_MIP_LEVELS,
                //.layerCount = VK_REMAINING_ARRAY_LAYERS
            },
        };

        VkDependencyInfo dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    // depth_image
    // from: VK_IMAGE_LAYOUT_UNDEFINED
    // to: VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
    {
        VkImageMemoryBarrier2 image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            // where
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            // how
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .image = depth_image.image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                //.levelCount = VK_REMAINING_MIP_LEVELS,
                //.layerCount = VK_REMAINING_ARRAY_LAYERS
            },
        };

        VkDependencyInfo dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    render_triangle(cmd);

    // draw_image
    // from: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    // to: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    {
        VkImageMemoryBarrier2 image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            // where
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            // how
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = draw_image.image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                //.levelCount = VK_REMAINING_MIP_LEVELS,
                //.layerCount = VK_REMAINING_ARRAY_LAYERS
            },
        };

        VkDependencyInfo dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    // swapchain_image
    // from VK_IMAGE_LAYOUT_UNDEFINED
    // to: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    {
        VkImageMemoryBarrier2 image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            // where
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            // how
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = swapchain_images[swapchain_image_index],
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                //.levelCount = VK_REMAINING_MIP_LEVELS,
                //.layerCount = VK_REMAINING_ARRAY_LAYERS
            },
        };

        VkDependencyInfo dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    copy_image_to_image(cmd, draw_image.image,
                        swapchain_images[swapchain_image_index], render_extent,
                        swapchain_extent);

    // swapchain_image
    // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    {
        VkImageMemoryBarrier2 image_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            // where
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            // how
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask =
                VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = swapchain_images[swapchain_image_index],
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                //.levelCount = VK_REMAINING_MIP_LEVELS,
                //.layerCount = VK_REMAINING_ARRAY_LAYERS
            },
        };

        VkDependencyInfo dep_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(cmd, &dep_info);
    }

    VkCommandBufferSubmitInfo cmd_submit_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd};

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
        SPDLOG_DEBUG("failed to present an image\n");
    }

    frame_id++;
}

void free_on_exit() {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);

    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);

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

    init_descriptors();

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
            case SDL_EVENT_KEY_DOWN:
                switch (e.key.key) {
                case SDLK_ESCAPE:
                    is_running = true;
                    break;
                }
            }
        }

        render();
    }

    free_on_exit();
}
