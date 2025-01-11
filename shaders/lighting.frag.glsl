#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D display_texture;

void main() {
    float ambient_strength = 0.9;
    vec3 light_color = vec3(0.7f, 0.7f, 1.0f);

    vec3 ambient = ambient_strength * light_color;

    vec3 result = ambient * vec3(1.f);

    frag_color = vec4(result, 1.f);
}

