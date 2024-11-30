#include "shader.hpp"

#include <fstream>

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

