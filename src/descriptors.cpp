#include "descriptors.hpp"

void DescriptorAllocator::init(VkDevice device, u32 max_sets,
                               std::span<PoolSizeRatio> pool_ratios) {
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (const auto p_ratio : pool_ratios) {
        pool_sizes.push_back(VkDescriptorPoolSize{
            .type = p_ratio.type,
            .descriptorCount = static_cast<u32>(p_ratio.ratio * max_sets)});
    }

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.flags = 0;
    info.maxSets = max_sets;
    info.poolSizeCount = pool_sizes.size();
    info.pPoolSizes = pool_sizes.data();

    vkCreateDescriptorPool(device, &info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device) {
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device) {
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device,
                                              VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;

    VkDescriptorSet ds;
    vk_check(vkAllocateDescriptorSets(device, &info, &ds));

    return ds;
}

void DescriptorWriter::write_image(int binding, VkImageView image,
                                   VkSampler sampler, VkImageLayout layout,
                                   VkDescriptorType type) {
    VkDescriptorImageInfo info{};
    info.sampler = sampler;
    info.imageView = image;
    info.imageLayout = layout;

    image_infos.push_back(info);
    VkDescriptorImageInfo &info_ref = image_infos.back();

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.dstSet = VK_NULL_HANDLE;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &info_ref;

    writes.push_back(write);
}

void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, usize size,
                                    usize offset, VkDescriptorType type) {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer;
    info.offset = offset;
    info.range = size;

    buffer_infos.push_back(info);
    VkDescriptorBufferInfo &info_ref = buffer_infos.back();

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.dstSet = VK_NULL_HANDLE;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &info_ref;

    writes.push_back(write);
}

void DescriptorWriter::clear() {
    buffer_infos.clear();
    image_infos.clear();
    writes.clear();
}

void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set) {
    for (auto &write : writes) {
        write.dstSet = set;
    }

    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
}
