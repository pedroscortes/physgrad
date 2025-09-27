#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "simulation.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

// Window dimensions
const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

// Camera parameters
struct Camera {
    float distance = 5.0f;
    float theta = 0.0f;  // Horizontal rotation
    float phi = 0.0f;    // Vertical rotation
    float fov = 45.0f;
};

Camera camera;
bool mouse_pressed = false;
double last_mouse_x = 0, last_mouse_y = 0;

// OpenGL handles
GLuint vbo = 0;
GLuint vao = 0;
GLuint shader_program = 0;
cudaGraphicsResource_t cuda_vbo_resource = nullptr;

// Simulation
std::unique_ptr<physgrad::Simulation> simulation;

// Shader sources (embedded for simplicity)
const char* vertex_shader_source = R"(
#version 330 core
layout(location = 0) in vec4 position;  // xyz = position, w = mass

uniform mat4 projection;
uniform mat4 view;

out vec3 particle_color;

void main() {
    gl_Position = projection * view * vec4(position.xyz, 1.0);
    gl_PointSize = 2.0 + position.w * 10.0;  // Size based on mass

    // Simple color gradient based on distance from origin
    float dist = length(position.xyz);
    particle_color = mix(vec3(0.3, 0.5, 1.0), vec3(1.0, 0.3, 0.3), dist / 5.0);
}
)";

const char* fragment_shader_source = R"(
#version 330 core
in vec3 particle_color;
out vec4 frag_color;

void main() {
    // Simple circular point
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;

    float alpha = 1.0 - dist * 2.0;
    frag_color = vec4(particle_color, alpha);
}
)";

// Compile shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed: " << info_log << std::endl;
        exit(1);
    }

    return shader;
}

// Initialize OpenGL
void initOpenGL() {
    // Enable depth testing and point sprites
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Compile and link shaders
    GLuint vertex_shader = compileShader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = compileShader(GL_FRAGMENT_SHADER, fragment_shader_source);

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);

    GLint success;
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shader_program, 512, nullptr, info_log);
        std::cerr << "Shader linking failed: " << info_log << std::endl;
        exit(1);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // Create VAO and VBO
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Allocate buffer for particle positions
    int num_particles = simulation->getBodies()->n;
    glBufferData(GL_ARRAY_BUFFER, num_particles * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    // Set vertex attributes
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                 cudaGraphicsRegisterFlagsWriteDiscard);

    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Failed to register VBO with CUDA\n";
        exit(1);
    }
}

// Update and render
void render(GLFWwindow* window) {
    // Update simulation
    simulation->step();

    // Map OpenGL buffer to CUDA
    float* d_vbo_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes,
                                         cuda_vbo_resource);

    // Get packed positions from simulation
    float* packed_positions = simulation->getPackedPositions();

    // Copy to VBO
    cudaMemcpy(d_vbo_ptr, packed_positions,
               simulation->getBodies()->n * 4 * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Unmap buffer
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up matrices
    glUseProgram(shader_program);

    // Projection matrix
    float aspect = float(WINDOW_WIDTH) / float(WINDOW_HEIGHT);
    float projection[16] = {0};
    float fov_rad = camera.fov * M_PI / 180.0f;
    float f = 1.0f / tanf(fov_rad / 2.0f);
    projection[0] = f / aspect;
    projection[5] = f;
    projection[10] = -1.01f;  // Near/far plane
    projection[11] = -1.0f;
    projection[14] = -0.02f;

    // View matrix (simple orbital camera)
    float view[16] = {0};
    float cos_theta = cosf(camera.theta);
    float sin_theta = sinf(camera.theta);
    float cos_phi = cosf(camera.phi);
    float sin_phi = sinf(camera.phi);

    float eye_x = camera.distance * sin_phi * cos_theta;
    float eye_y = camera.distance * cos_phi;
    float eye_z = camera.distance * sin_phi * sin_theta;

    // Simple look-at matrix
    view[0] = -sin_theta;
    view[1] = sin_phi * cos_theta;
    view[2] = cos_phi * cos_theta;
    view[3] = 0;

    view[4] = 0;
    view[5] = cos_phi;
    view[6] = -sin_phi;
    view[7] = 0;

    view[8] = -cos_theta;
    view[9] = -sin_phi * sin_theta;
    view[10] = -cos_phi * sin_theta;
    view[11] = 0;

    view[12] = eye_x;
    view[13] = eye_y;
    view[14] = eye_z;
    view[15] = 1;

    // Set uniforms
    GLint proj_loc = glGetUniformLocation(shader_program, "projection");
    GLint view_loc = glGetUniformLocation(shader_program, "view");
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view);

    // Draw particles
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, simulation->getBodies()->n);
    glBindVertexArray(0);

    // Show FPS and performance
    static int frame_count = 0;
    static double last_time = glfwGetTime();
    frame_count++;

    double current_time = glfwGetTime();
    if (current_time - last_time > 1.0) {
        double fps = frame_count / (current_time - last_time);
        float gflops = simulation->getGFLOPS();

        std::stringstream title;
        title << "PhysGrad - " << simulation->getBodies()->n << " bodies | "
              << "FPS: " << int(fps) << " | "
              << "GFLOPS: " << gflops << " | "
              << "Step time: " << simulation->getLastStepTime() << " ms";

        glfwSetWindowTitle(window, title.str().c_str());

        frame_count = 0;
        last_time = current_time;
    }
}

// Mouse callbacks
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouse_pressed = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mouse_pressed) {
        double dx = xpos - last_mouse_x;
        double dy = ypos - last_mouse_y;

        camera.theta += float(dx) * 0.01f;
        camera.phi += float(dy) * 0.01f;

        // Clamp phi to avoid flipping
        if (camera.phi < 0.1f) camera.phi = 0.1f;
        if (camera.phi > M_PI - 0.1f) camera.phi = M_PI - 0.1f;

        last_mouse_x = xpos;
        last_mouse_y = ypos;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.distance *= float(1.0 - yoffset * 0.1);
    if (camera.distance < 1.0f) camera.distance = 1.0f;
    if (camera.distance > 100.0f) camera.distance = 100.0f;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    physgrad::SimParams params;
    if (argc > 1) {
        params.num_bodies = std::atoi(argv[1]);
    }

    std::cout << "Starting PhysGrad N-body simulation\n";
    std::cout << "Bodies: " << params.num_bodies << "\n";

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                          "PhysGrad", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // OpenGL is ready (using GL_GLEXT_PROTOTYPES instead of GLEW)

    // Set callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize simulation
    simulation = std::make_unique<physgrad::Simulation>(params);

    // Initialize OpenGL
    initOpenGL();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        render(window);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    if (cuda_vbo_resource) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shader_program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}