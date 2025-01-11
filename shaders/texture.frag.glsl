#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D display_texture;

void main() {
    frag_color = texture(display_texture, in_uv);
    //frag_color = vec4(1.f, in_uv, 1.f);
}

