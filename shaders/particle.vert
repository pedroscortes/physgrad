#version 330 core
layout(location = 0) in vec4 position;  // xyz = position, w = mass

uniform mat4 projection;
uniform mat4 view;

out vec3 particle_color;

void main() {
    gl_Position = projection * view * vec4(position.xyz, 1.0);
    gl_PointSize = 2.0 + position.w * 10.0;  // Size based on mass

    // Color gradient based on distance from origin
    float dist = length(position.xyz);
    particle_color = mix(vec3(0.3, 0.5, 1.0), vec3(1.0, 0.3, 0.3), dist / 5.0);
}