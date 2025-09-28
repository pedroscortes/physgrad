#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace physgrad {

struct Camera {
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};

    float fov = 45.0f;
    float near_plane = 0.1f;
    float far_plane = 1000.0f;

    bool mouse_dragging = false;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    float distance = 5.0f;
    float theta = 0.0f;  // horizontal angle
    float phi = 0.0f;    // vertical angle

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect_ratio) const;
    void processInput(GLFWwindow* window, float delta_time);
    void processMouseMovement(double xpos, double ypos);
    void processMouseScroll(double yoffset);
    void updatePosition();
};

struct RenderObject {
    glm::vec3 position{0.0f};
    glm::vec3 scale{1.0f};
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    bool visible = true;
};

struct Particle : public RenderObject {
    float mass = 1.0f;
    glm::vec3 velocity{0.0f};
    glm::vec3 force{0.0f};
    std::vector<glm::vec3> trail;
    size_t max_trail_length = 100;

    void updateTrail();
    glm::vec3 getSpeedBasedColor() const;
    float getRadiusFromMass() const;
};

struct ForceVector : public RenderObject {
    glm::vec3 direction{0.0f};
    float magnitude = 1.0f;
    float scale_factor = 1.0f;

    void updateFromForce(const glm::vec3& force_vec, const glm::vec3& origin);
};

struct VisualizationSettings {
    bool show_particles = true;
    bool show_force_vectors = true;
    bool show_trails = true;
    bool show_grid = true;
    bool show_coordinate_axes = true;
    bool show_energy_info = true;
    bool show_performance_info = true;

    float particle_scale = 1.0f;
    float force_scale = 0.1f;
    float trail_alpha = 0.5f;
    int trail_length = 50;

    glm::vec3 background_color{0.1f, 0.1f, 0.15f};
    glm::vec3 grid_color{0.3f, 0.3f, 0.3f};
    glm::vec3 axis_x_color{1.0f, 0.0f, 0.0f};
    glm::vec3 axis_y_color{0.0f, 1.0f, 0.0f};
    glm::vec3 axis_z_color{0.0f, 0.0f, 1.0f};
};

class Shader {
private:
    GLuint program_id = 0;
    std::unordered_map<std::string, GLint> uniform_locations;

public:
    Shader() = default;
    ~Shader();

    bool loadFromStrings(const std::string& vertex_source, const std::string& fragment_source);
    bool loadFromFiles(const std::string& vertex_path, const std::string& fragment_path);

    void use();
    void setUniform(const std::string& name, float value);
    void setUniform(const std::string& name, int value);
    void setUniform(const std::string& name, const glm::vec3& value);
    void setUniform(const std::string& name, const glm::vec4& value);
    void setUniform(const std::string& name, const glm::mat4& value);

    GLuint getID() const { return program_id; }

private:
    GLint getUniformLocation(const std::string& name);
    bool compileShader(GLuint& shader_id, GLenum type, const std::string& source);
};

class Renderer {
private:
    GLFWwindow* window = nullptr;
    Camera camera;
    VisualizationSettings settings;

    // Rendering resources
    std::unique_ptr<Shader> particle_shader;
    std::unique_ptr<Shader> line_shader;
    std::unique_ptr<Shader> grid_shader;

    GLuint sphere_VAO = 0, sphere_VBO = 0, sphere_EBO = 0;
    GLuint line_VAO = 0, line_VBO = 0;
    GLuint grid_VAO = 0, grid_VBO = 0;

    std::vector<glm::vec3> sphere_vertices;
    std::vector<unsigned int> sphere_indices;

    // Performance tracking
    float frame_time = 0.0f;
    float last_frame = 0.0f;
    int frame_count = 0;
    float fps = 0.0f;

public:
    Renderer();
    ~Renderer();

    bool initialize(int width = 1280, int height = 720, const std::string& title = "PhysGrad Visualization");
    void shutdown();

    bool shouldClose() const;
    void beginFrame();
    void endFrame();

    void renderParticles(const std::vector<Particle>& particles);
    void renderForceVectors(const std::vector<ForceVector>& forces);
    void renderTrails(const std::vector<Particle>& particles);
    void renderGrid(float size = 10.0f, int divisions = 20);
    void renderCoordinateAxes(float length = 1.0f);

    void renderUI();
    void renderEnergyInfo(float kinetic, float potential, float total);
    void renderPerformanceInfo();

    Camera& getCamera() { return camera; }
    VisualizationSettings& getSettings() { return settings; }
    GLFWwindow* getWindow() { return window; }

    float getDeltaTime() const { return frame_time; }
    float getFPS() const { return fps; }

private:
    void setupImGui();
    void createSphere(float radius = 1.0f, int segments = 24);
    void setupShaders();
    void setupBuffers();
    void updatePerformanceStats();

    static void mouseCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

class VisualizationManager {
private:
    std::unique_ptr<Renderer> renderer;
    std::vector<Particle> particles;
    std::vector<ForceVector> force_vectors;

    // Energy tracking
    float kinetic_energy = 0.0f;
    float potential_energy = 0.0f;
    float total_energy = 0.0f;

    // Simulation state
    bool simulation_running = false;
    bool single_step = false;
    float time_scale = 1.0f;

    // Collision statistics
    int collision_broad_pairs = 0;
    int collision_narrow_tests = 0;
    int collision_contacts = 0;

    // Interactive parameters
    struct InteractiveParams {
        float gravity_strength = 9.8f;
        float air_damping = 0.999f;
        float contact_stiffness = 500.0f;
        float contact_damping = 5.0f;
        float restitution = 0.7f;
        float friction = 0.2f;
        float particle_mass_scale = 1.0f;
        bool show_force_vectors = true;
        bool show_particle_trails = false;
        bool show_velocity_colors = true;
        bool enable_gravity = true;
        bool enable_collisions = true;
        int max_particles = 50;
        bool pause_on_collision = false;
    } interactive_params;

public:
    VisualizationManager();
    ~VisualizationManager();

    bool initialize(int width = 1280, int height = 720);
    void shutdown();

    void updateFromSimulation(const std::vector<float>& pos_x,
                             const std::vector<float>& pos_y,
                             const std::vector<float>& pos_z,
                             const std::vector<float>& vel_x,
                             const std::vector<float>& vel_y,
                             const std::vector<float>& vel_z,
                             const std::vector<float>& masses);

    void updateForces(const std::vector<float>& force_x,
                     const std::vector<float>& force_y,
                     const std::vector<float>& force_z);

    void updateEnergy(float kinetic, float potential);
    void setCollisionStats(int broad_pairs, int narrow_tests, int contacts);

    // Interactive parameter access
    const InteractiveParams& getInteractiveParams() const { return interactive_params; }
    InteractiveParams& getInteractiveParams() { return interactive_params; }

    void render();
    bool shouldClose() const;

    bool isSimulationRunning() const { return simulation_running; }
    bool shouldSingleStep() const { return single_step; }
    float getTimeScale() const { return time_scale; }

    void resetSingleStep() { single_step = false; }

    Renderer& getRenderer() { return *renderer; }
};

// Shader source code
namespace ShaderSources {
    extern const char* particle_vertex_shader;
    extern const char* particle_fragment_shader;
    extern const char* line_vertex_shader;
    extern const char* line_fragment_shader;
    extern const char* grid_vertex_shader;
    extern const char* grid_fragment_shader;
}

} // namespace physgrad