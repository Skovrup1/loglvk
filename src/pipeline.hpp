#pragma once

#include "core.hpp"

class PipelineBuilder {
  public:
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState color_blend_attachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipeline_layout;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineRenderingCreateInfo render_info;
    VkFormat color_attachment_format;

    PipelineBuilder() { clear(); }

    void clear();
    VkPipeline build(VkDevice device);

    PipelineBuilder &set_pipeline_layout(VkPipelineLayout layout);
    PipelineBuilder &set_shaders(VkShaderModule vertexShader,
                                 VkShaderModule fragmentShader);
    PipelineBuilder &set_input_topology(VkPrimitiveTopology topology);
    PipelineBuilder &set_polygon_mode(VkPolygonMode mode);
    PipelineBuilder &set_cull_mode(VkCullModeFlags cullMode,
                                   VkFrontFace frontFace);
    PipelineBuilder &set_multisampling_none();
    PipelineBuilder &disable_blending();
    PipelineBuilder &enable_blending_additive();
    PipelineBuilder &enable_blending_alphablend();
    PipelineBuilder &set_color_attachment_format(VkFormat format);
    PipelineBuilder &set_depth_format(VkFormat format);
    PipelineBuilder &disable_depthtest();
    PipelineBuilder &enable_depthtest(bool depthWriteEnable, VkCompareOp op);
};
