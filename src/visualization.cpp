#include "visualization.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace physgrad {

// Shader source code definitions
namespace ShaderSources {
    const char* particle_vertex_shader = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 particle_pos;
        uniform float particle_radius;

        out vec3 FragPos;
        out vec3 Normal;

        void main() {
            vec3 world_pos = particle_pos + aPos * particle_radius;
            FragPos = world_pos;
            Normal = mat3(model) * aNormal;
            gl_Position = projection * view * model * vec4(world_pos, 1.0);
        }
    )";

    const char* particle_fragment_shader = R"(
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;

        uniform vec3 particle_color;
        uniform float particle_alpha;
        uniform vec3 light_pos;
        uniform vec3 view_pos;

        void main() {
            vec3 norm = normalize(Normal);
            vec3 light_dir = normalize(light_pos - FragPos);

            float ambient = 0.3;
            float diffuse = max(dot(norm, light_dir), 0.0) * 0.6;

            vec3 view_dir = normalize(view_pos - FragPos);
            vec3 reflect_dir = reflect(-light_dir, norm);
            float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32) * 0.5;

            vec3 result = (ambient + diffuse + specular) * particle_color;
            FragColor = vec4(result, particle_alpha);
        }
    )";

    const char* line_vertex_shader = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
    )";

    const char* line_fragment_shader = R"(
        #version 330 core
        out vec4 FragColor;

        uniform vec3 line_color;
        uniform float line_alpha;

        void main() {
            FragColor = vec4(line_color, line_alpha);
        }
    )";

    const char* grid_vertex_shader = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
        }
    )";

    const char* grid_fragment_shader = R"(
        #version 330 core
        out vec4 FragColor;

        uniform vec3 grid_color;

        void main() {
            FragColor = vec4(grid_color, 0.5);
        }
    )";
}

// Camera implementation
glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, target, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspect_ratio) const {
    return glm::perspective(glm::radians(fov), aspect_ratio, near_plane, far_plane);
}

void Camera::processInput(GLFWwindow* window, float delta_time) {
    float speed = 5.0f * delta_time;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        distance -= speed;
        if (distance < 0.1f) distance = 0.1f;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        distance += speed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        theta -= speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        theta += speed;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        phi += speed;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        phi -= speed;
    }

    updatePosition();
}

void Camera::processMouseMovement(double xpos, double ypos) {
    if (!mouse_dragging) {
        last_mouse_x = xpos;
        last_mouse_y = ypos;
        return;
    }

    float sensitivity = 0.005f;
    float xoffset = (xpos - last_mouse_x) * sensitivity;
    float yoffset = (last_mouse_y - ypos) * sensitivity;

    theta += xoffset;
    phi += yoffset;

    if (phi > 1.5f) phi = 1.5f;
    if (phi < -1.5f) phi = -1.5f;

    updatePosition();

    last_mouse_x = xpos;
    last_mouse_y = ypos;
}

void Camera::processMouseScroll(double yoffset) {
    distance -= yoffset * 0.5f;
    if (distance < 0.1f) distance = 0.1f;
    if (distance > 100.0f) distance = 100.0f;
    updatePosition();
}

void Camera::updatePosition() {
    position.x = target.x + distance * cos(phi) * cos(theta);
    position.y = target.y + distance * sin(phi);
    position.z = target.z + distance * cos(phi) * sin(theta);
}

// Particle implementation
void Particle::updateTrail() {
    trail.push_back(position);
    if (trail.size() > max_trail_length) {
        trail.erase(trail.begin());
    }
}

glm::vec3 Particle::getSpeedBasedColor() const {
    float speed = glm::length(velocity);
    float normalized_speed = std::min(speed / 5.0f, 1.0f);

    // Blue to red gradient based on speed
    return glm::vec3(normalized_speed, 0.0f, 1.0f - normalized_speed);
}

float Particle::getRadiusFromMass() const {
    return 0.05f + 0.1f * std::pow(mass / 10.0f, 1.0f/3.0f);
}

// ForceVector implementation
void ForceVector::updateFromForce(const glm::vec3& force_vec, const glm::vec3& origin) {
    position = origin;
    direction = glm::normalize(force_vec);
    magnitude = glm::length(force_vec);
    scale = glm::vec3(magnitude * scale_factor);
}

// Shader implementation
Shader::~Shader() {
    if (program_id != 0) {
        glDeleteProgram(program_id);
    }
}

bool Shader::loadFromStrings(const std::string& vertex_source, const std::string& fragment_source) {
    GLuint vertex_shader, fragment_shader;

    if (!compileShader(vertex_shader, GL_VERTEX_SHADER, vertex_source)) {
        return false;
    }

    if (!compileShader(fragment_shader, GL_FRAGMENT_SHADER, fragment_source)) {
        glDeleteShader(vertex_shader);
        return false;
    }

    program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader);
    glAttachShader(program_id, fragment_shader);
    glLinkProgram(program_id);

    GLint success;
    glGetProgramiv(program_id, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program_id, 512, nullptr, info_log);
        std::cerr << "Shader program linking failed: " << info_log << std::endl;
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return true;
}

void Shader::use() {
    glUseProgram(program_id);
}

void Shader::setUniform(const std::string& name, float value) {
    glUniform1f(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, int value) {
    glUniform1i(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, const glm::vec3& value) {
    glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void Shader::setUniform(const std::string& name, const glm::vec4& value) {
    glUniform4fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void Shader::setUniform(const std::string& name, const glm::mat4& value) {
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

GLint Shader::getUniformLocation(const std::string& name) {
    auto it = uniform_locations.find(name);
    if (it != uniform_locations.end()) {
        return it->second;
    }

    GLint location = glGetUniformLocation(program_id, name.c_str());
    uniform_locations[name] = location;
    return location;
}

bool Shader::compileShader(GLuint& shader_id, GLenum type, const std::string& source) {
    shader_id = glCreateShader(type);
    const char* source_cstr = source.c_str();
    glShaderSource(shader_id, 1, &source_cstr, nullptr);
    glCompileShader(shader_id);

    GLint success;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader_id, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed: " << info_log << std::endl;
        return false;
    }

    return true;
}

// Renderer implementation
Renderer::Renderer() = default;

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(int width, int height, const std::string& title) {
    // Check for display environment (detect headless mode)
    const char* display = std::getenv("DISPLAY");
    if (!display || std::strlen(display) == 0) {
        std::cerr << "Warning: No display environment detected (headless mode)." << std::endl;
        std::cerr << "Visualization disabled - running in headless mode." << std::endl;
        return false;
    }

    // Set error callback before initialization
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
    });

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);

    glfwSetMouseButtonCallback(window, mouseCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    setupImGui();
    setupShaders();
    setupBuffers();

    camera.updatePosition();

    return true;
}

void Renderer::shutdown() {
    if (sphere_VAO) glDeleteVertexArrays(1, &sphere_VAO);
    if (sphere_VBO) glDeleteBuffers(1, &sphere_VBO);
    if (sphere_EBO) glDeleteBuffers(1, &sphere_EBO);
    if (line_VAO) glDeleteVertexArrays(1, &line_VAO);
    if (line_VBO) glDeleteBuffers(1, &line_VBO);
    if (grid_VAO) glDeleteVertexArrays(1, &grid_VAO);
    if (grid_VBO) glDeleteBuffers(1, &grid_VBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool Renderer::shouldClose() const {
    return window && glfwWindowShouldClose(window);
}

void Renderer::beginFrame() {
    float current_frame = static_cast<float>(glfwGetTime());
    frame_time = current_frame - last_frame;
    last_frame = current_frame;

    updatePerformanceStats();

    glfwPollEvents();
    camera.processInput(window, frame_time);

    glClearColor(settings.background_color.r, settings.background_color.g, settings.background_color.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Renderer::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

void Renderer::renderParticles(const std::vector<Particle>& particles) {
    if (!settings.show_particles || !particle_shader) return;

    particle_shader->use();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glm::mat4 projection = camera.getProjectionMatrix(static_cast<float>(width) / height);
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 model = glm::mat4(1.0f);

    particle_shader->setUniform("model", model);
    particle_shader->setUniform("view", view);
    particle_shader->setUniform("projection", projection);
    particle_shader->setUniform("light_pos", glm::vec3(5.0f, 5.0f, 5.0f));
    particle_shader->setUniform("view_pos", camera.position);

    glBindVertexArray(sphere_VAO);

    for (const auto& particle : particles) {
        if (!particle.visible) continue;

        glm::vec3 color = particle.color;
        if (glm::length(particle.velocity) > 0.1f) {
            color = particle.getSpeedBasedColor();
        }

        particle_shader->setUniform("particle_pos", particle.position);
        particle_shader->setUniform("particle_radius", particle.getRadiusFromMass() * settings.particle_scale);
        particle_shader->setUniform("particle_color", color);
        particle_shader->setUniform("particle_alpha", particle.alpha);

        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(sphere_indices.size()), GL_UNSIGNED_INT, 0);
    }

    glBindVertexArray(0);
}

void Renderer::renderUI() {
    ImGui::Begin("PhysGrad Controls");

    if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show Particles", &settings.show_particles);
        ImGui::Checkbox("Show Force Vectors", &settings.show_force_vectors);
        ImGui::Checkbox("Show Trails", &settings.show_trails);
        ImGui::Checkbox("Show Grid", &settings.show_grid);
        ImGui::Checkbox("Show Axes", &settings.show_coordinate_axes);

        ImGui::SliderFloat("Particle Scale", &settings.particle_scale, 0.1f, 5.0f);
        ImGui::SliderFloat("Force Scale", &settings.force_scale, 0.01f, 1.0f);
        ImGui::SliderInt("Trail Length", &settings.trail_length, 10, 200);
    }

    if (ImGui::CollapsingHeader("Camera")) {
        float pos[3] = {camera.position.x, camera.position.y, camera.position.z};
        ImGui::InputFloat3("Position", pos);

        float target[3] = {camera.target.x, camera.target.y, camera.target.z};
        ImGui::InputFloat3("Target", target);

        ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f);
        ImGui::SliderFloat("Distance", &camera.distance, 0.1f, 50.0f);

        if (ImGui::Button("Reset Camera")) {
            camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
            camera.target = glm::vec3(0.0f, 0.0f, 0.0f);
            camera.distance = 5.0f;
            camera.theta = 0.0f;
            camera.phi = 0.0f;
            camera.updatePosition();
        }
    }

    ImGui::End();
}

void Renderer::setupImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Renderer::createSphere(float radius, int segments) {
    sphere_vertices.clear();
    sphere_indices.clear();

    for (int i = 0; i <= segments; ++i) {
        float phi = M_PI * i / segments;
        for (int j = 0; j <= segments; ++j) {
            float theta = 2.0f * M_PI * j / segments;

            float x = radius * sin(phi) * cos(theta);
            float y = radius * cos(phi);
            float z = radius * sin(phi) * sin(theta);

            sphere_vertices.push_back(glm::vec3(x, y, z));
            sphere_vertices.push_back(glm::vec3(x, y, z)); // Normal (same as position for unit sphere)
        }
    }

    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < segments; ++j) {
            int first = i * (segments + 1) + j;
            int second = first + segments + 1;

            sphere_indices.push_back(first);
            sphere_indices.push_back(second);
            sphere_indices.push_back(first + 1);

            sphere_indices.push_back(second);
            sphere_indices.push_back(second + 1);
            sphere_indices.push_back(first + 1);
        }
    }
}

void Renderer::setupShaders() {
    particle_shader = std::make_unique<Shader>();
    particle_shader->loadFromStrings(ShaderSources::particle_vertex_shader,
                                   ShaderSources::particle_fragment_shader);

    line_shader = std::make_unique<Shader>();
    line_shader->loadFromStrings(ShaderSources::line_vertex_shader,
                               ShaderSources::line_fragment_shader);

    grid_shader = std::make_unique<Shader>();
    grid_shader->loadFromStrings(ShaderSources::grid_vertex_shader,
                               ShaderSources::grid_fragment_shader);
}

void Renderer::setupBuffers() {
    createSphere(1.0f, 24);

    glGenVertexArrays(1, &sphere_VAO);
    glGenBuffers(1, &sphere_VBO);
    glGenBuffers(1, &sphere_EBO);

    glBindVertexArray(sphere_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO);
    glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(glm::vec3),
                sphere_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(unsigned int),
                sphere_indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &line_VAO);
    glGenBuffers(1, &line_VBO);

    glBindVertexArray(line_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, line_VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void Renderer::updatePerformanceStats() {
    frame_count++;
    static float last_time = 0.0f;
    float current_time = static_cast<float>(glfwGetTime());

    if (current_time - last_time >= 1.0f) {
        fps = frame_count / (current_time - last_time);
        frame_count = 0;
        last_time = current_time;
    }
}

// Static callbacks
void Renderer::mouseCallback(GLFWwindow* window, int button, int action, int mods) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        renderer->camera.mouse_dragging = (action == GLFW_PRESS);
    }
}

void Renderer::cursorCallback(GLFWwindow* window, double xpos, double ypos) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->camera.processMouseMovement(xpos, ypos);
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    renderer->camera.processMouseScroll(yoffset);
}

void Renderer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// VisualizationManager implementation
VisualizationManager::VisualizationManager() = default;

VisualizationManager::~VisualizationManager() = default;

bool VisualizationManager::initialize(int width, int height) {
    renderer = std::make_unique<Renderer>();
    return renderer->initialize(width, height, "PhysGrad - Real-time Physics Visualization");
}

void VisualizationManager::shutdown() {
    if (renderer) {
        renderer->shutdown();
        renderer.reset();
    }
}

void VisualizationManager::updateFromSimulation(const std::vector<float>& pos_x,
                                               const std::vector<float>& pos_y,
                                               const std::vector<float>& pos_z,
                                               const std::vector<float>& vel_x,
                                               const std::vector<float>& vel_y,
                                               const std::vector<float>& vel_z,
                                               const std::vector<float>& masses) {
    size_t n = pos_x.size();
    if (particles.size() != n) {
        particles.resize(n);
    }

    for (size_t i = 0; i < n; ++i) {
        particles[i].position = glm::vec3(pos_x[i], pos_y[i], pos_z[i]);
        particles[i].velocity = glm::vec3(vel_x[i], vel_y[i], vel_z[i]);
        particles[i].mass = masses[i];
        particles[i].updateTrail();
    }
}

void VisualizationManager::render() {
    if (!renderer) return;

    renderer->beginFrame();

    if (renderer->getSettings().show_grid) {
        renderer->renderGrid(10.0f, 20);
    }

    if (renderer->getSettings().show_coordinate_axes) {
        renderer->renderCoordinateAxes(2.0f);
    }

    renderer->renderParticles(particles);

    renderer->renderUI();

    ImGui::Begin("Simulation Info");
    ImGui::Text("Particles: %zu", particles.size());
    ImGui::Text("FPS: %.1f", renderer->getFPS());
    ImGui::Text("Frame Time: %.3f ms", renderer->getDeltaTime() * 1000.0f);

    ImGui::Separator();
    ImGui::Text("Energy");
    ImGui::Text("Kinetic: %.6f", kinetic_energy);
    ImGui::Text("Potential: %.6f", potential_energy);
    ImGui::Text("Total: %.6f", total_energy);

    ImGui::Separator();
    ImGui::Checkbox("Running", &simulation_running);
    ImGui::SameLine();
    if (ImGui::Button("Step")) {
        single_step = true;
    }
    ImGui::SliderFloat("Speed", &time_scale, 0.0f, 5.0f);

    // Collision information
    if (collision_contacts > 0 || collision_broad_pairs > 0) {
        ImGui::Separator();
        ImGui::Text("Collision Detection");
        ImGui::Text("Broad Phase Pairs: %d", collision_broad_pairs);
        ImGui::Text("Narrow Phase Tests: %d", collision_narrow_tests);
        ImGui::Text("Active Contacts: %d", collision_contacts);
    }

    ImGui::End();

    // Interactive Controls Panel
    ImGui::Begin("Physics Controls");

    ImGui::Text("Physics Parameters");
    ImGui::Separator();

    ImGui::Checkbox("Enable Gravity", &interactive_params.enable_gravity);
    if (interactive_params.enable_gravity) {
        ImGui::SliderFloat("Gravity Strength", &interactive_params.gravity_strength, 0.0f, 20.0f);
    }

    ImGui::Checkbox("Enable Collisions", &interactive_params.enable_collisions);
    if (interactive_params.enable_collisions) {
        ImGui::SliderFloat("Contact Stiffness", &interactive_params.contact_stiffness, 50.0f, 2000.0f);
        ImGui::SliderFloat("Contact Damping", &interactive_params.contact_damping, 0.1f, 20.0f);
        ImGui::SliderFloat("Restitution", &interactive_params.restitution, 0.0f, 1.0f);
        ImGui::SliderFloat("Friction", &interactive_params.friction, 0.0f, 1.0f);
    }

    ImGui::SliderFloat("Air Damping", &interactive_params.air_damping, 0.95f, 1.0f);
    ImGui::SliderFloat("Mass Scale", &interactive_params.particle_mass_scale, 0.1f, 5.0f);

    ImGui::Separator();
    ImGui::Text("Visualization");
    ImGui::Checkbox("Show Force Vectors", &interactive_params.show_force_vectors);
    ImGui::Checkbox("Show Particle Trails", &interactive_params.show_particle_trails);
    ImGui::Checkbox("Velocity-based Colors", &interactive_params.show_velocity_colors);
    ImGui::Checkbox("Pause on Collision", &interactive_params.pause_on_collision);

    ImGui::Separator();
    ImGui::Text("Scenario Controls");
    if (ImGui::Button("Reset Particles")) {
        // Signal to reset particles (we'll handle this in the demo)
    }
    if (ImGui::Button("Add Random Particle")) {
        // Signal to add a particle
    }
    if (ImGui::Button("Clear All")) {
        // Signal to clear all particles
    }

    ImGui::End();

    // Mathematical Information Panel
    ImGui::Begin("Mathematical Info");

    ImGui::Text("Physics Equations");
    ImGui::Separator();

    ImGui::Text("Newton's 2nd Law:");
    ImGui::Text("F = ma");
    ImGui::Text("a = F/m");

    ImGui::Separator();
    ImGui::Text("Gravitational Force:");
    ImGui::Text("F = mg");
    ImGui::Text("where g = %.2f m/s²", interactive_params.gravity_strength);

    ImGui::Separator();
    ImGui::Text("Contact Force (Spring-Damper):");
    ImGui::Text("F = kx + cv");
    ImGui::Text("k = %.1f (stiffness)", interactive_params.contact_stiffness);
    ImGui::Text("c = %.1f (damping)", interactive_params.contact_damping);

    ImGui::Separator();
    ImGui::Text("Energy Conservation:");
    ImGui::Text("E_total = E_kinetic + E_potential");
    ImGui::Text("Kinetic: %.6f", kinetic_energy);
    ImGui::Text("Potential: %.6f", potential_energy);
    ImGui::Text("Total: %.6f", total_energy);

    ImGui::Separator();
    ImGui::Text("Collision Response:");
    ImGui::Text("Coefficient of Restitution: %.2f", interactive_params.restitution);
    ImGui::Text("v_separation = -e * v_approach");
    ImGui::Text("Friction coefficient: %.2f", interactive_params.friction);

    ImGui::End();

    renderer->endFrame();
}

bool VisualizationManager::shouldClose() const {
    return renderer ? renderer->shouldClose() : true;
}

void VisualizationManager::updateEnergy(float kinetic, float potential) {
    kinetic_energy = kinetic;
    potential_energy = potential;
    total_energy = kinetic + potential;
}

void VisualizationManager::setCollisionStats(int broad_pairs, int narrow_tests, int contacts) {
    collision_broad_pairs = broad_pairs;
    collision_narrow_tests = narrow_tests;
    collision_contacts = contacts;
}

void VisualizationManager::updateForces(const std::vector<float>& force_x,
                                       const std::vector<float>& force_y,
                                       const std::vector<float>& force_z) {
    force_vectors.clear();
    size_t min_size = std::min(force_x.size(), std::min(force_y.size(), force_z.size()));
    for (size_t i = 0; i < min_size; ++i) {
        if (i < particles.size()) {
            ForceVector fv;
            fv.position = particles[i].position;
            fv.direction = glm::vec3(force_x[i], force_y[i], force_z[i]);
            fv.magnitude = glm::length(fv.direction);
            if (fv.magnitude > 0.0f) {
                fv.direction = glm::normalize(fv.direction);
                fv.scale_factor = 0.1f;
                fv.color = glm::vec3(1.0f, 0.5f, 0.0f);
                force_vectors.push_back(fv);
            }
        }
    }
}

void Renderer::renderGrid(float size, int divisions) {
    if (!grid_shader) return;

    grid_shader->use();
    grid_shader->setUniform("view", camera.getViewMatrix());
    grid_shader->setUniform("projection", camera.getProjectionMatrix(1280.0f / 720.0f));
    grid_shader->setUniform("color", settings.grid_color);

    std::vector<glm::vec3> grid_vertices;
    float step = size / divisions;
    float half_size = size * 0.5f;

    for (int i = 0; i <= divisions; ++i) {
        float pos = -half_size + i * step;

        grid_vertices.push_back(glm::vec3(pos, 0, -half_size));
        grid_vertices.push_back(glm::vec3(pos, 0, half_size));

        grid_vertices.push_back(glm::vec3(-half_size, 0, pos));
        grid_vertices.push_back(glm::vec3(half_size, 0, pos));
    }

    glBindVertexArray(grid_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, grid_VBO);
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.size() * sizeof(glm::vec3), grid_vertices.data(), GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_LINES, 0, grid_vertices.size());
    glBindVertexArray(0);
}

void Renderer::renderCoordinateAxes(float length) {
    if (!line_shader) return;

    line_shader->use();
    line_shader->setUniform("view", camera.getViewMatrix());
    line_shader->setUniform("projection", camera.getProjectionMatrix(1280.0f / 720.0f));

    glBindVertexArray(line_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, line_VBO);

    std::vector<glm::vec3> axis_vertices = {
        glm::vec3(0, 0, 0), glm::vec3(length, 0, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, length, 0),
        glm::vec3(0, 0, 0), glm::vec3(0, 0, length)
    };

    glBufferData(GL_ARRAY_BUFFER, axis_vertices.size() * sizeof(glm::vec3), axis_vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    glLineWidth(3.0f);

    line_shader->setUniform("color", settings.axis_x_color);
    glDrawArrays(GL_LINES, 0, 2);

    line_shader->setUniform("color", settings.axis_y_color);
    glDrawArrays(GL_LINES, 2, 2);

    line_shader->setUniform("color", settings.axis_z_color);
    glDrawArrays(GL_LINES, 4, 2);

    glLineWidth(1.0f);
    glBindVertexArray(0);
}

} // namespace physgrad