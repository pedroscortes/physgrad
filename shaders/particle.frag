#version 330 core
in vec3 particle_color;
out vec4 frag_color;

void main() {
    // Circular point sprite
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float alpha = 1.0 - dist * 2.0;
    frag_color = vec4(particle_color, alpha);
}