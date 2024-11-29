#version 450

#extension GL_EXT_buffer_reference : require

layout(location = 0) out vec3 out_color;
//layout(location = 1) out vec3 out_normal;
//layout(location = 2) out vec2 out_uv;

struct Vertex {
    vec3 position;
    //    float uv_x;
    //    vec3 normal;
    //    float uv_y;
    //    vec4 color;
};

layout(buffer_reference, std430) buffer VertexBuffer {
    Vertex vertices[];
};

layout(push_constant) uniform constants {
    mat4 render_matrix;
    VertexBuffer vertex_buffer;
} push_constants;

void main() {
    Vertex v = push_constants.vertex_buffer.vertices[gl_VertexIndex];

    gl_Position = vec4(v.position, 1.f);

    out_color = v.position;
    //  out_normal = v.normal;
    // out_uv.x = v.uv_x;
    //out_uv.y = v.uv_y;
}
