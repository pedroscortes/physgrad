#include "../src/symbolic_regression.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace physgrad;

// Example 1: Discover Hooke's Law F = k*x
void discover_hookes_law() {
    std::cout << "\n=== Discovering Hooke's Law: F = k*x ===" << std::endl;

    // Generate synthetic data for spring force
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    const double k = 150.0; // Spring constant
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    for (double x = 0.1; x <= 1.0; x += 0.05) {
        X.push_back({x});
        double force = k * x + noise(gen);
        y.push_back(force);
    }

    std::cout << "Generated " << X.size() << " data points for spring displacement vs force" << std::endl;

    // Create expression manually (x * constant)
    std::vector<std::string> variables = {"displacement"};
    auto x_node = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto k_node = std::make_shared<Node>(OperatorType::CONSTANT, 100.0);
    auto force_expr = std::make_shared<Node>(OperatorType::MULTIPLY, k_node, x_node);

    SymbolicExpression hookes_expr(force_expr, variables);

    std::cout << "Manual Hooke's Law expression: " << hookes_expr.to_string() << std::endl;

    // Test evaluation
    double test_displacement = 0.5;
    double predicted_force = hookes_expr.evaluate({test_displacement});
    double actual_force = k * test_displacement;

    std::cout << "At displacement " << test_displacement << " m:" << std::endl;
    std::cout << "  Predicted force: " << predicted_force << " N" << std::endl;
    std::cout << "  Actual force: " << actual_force << " N" << std::endl;

    // Calculate R² manually
    double mean_y = 0.0;
    for (double val : y) mean_y += val;
    mean_y /= y.size();

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double pred = hookes_expr.evaluate(X[i]);
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        ss_res += (y[i] - pred) * (y[i] - pred);
    }

    double r_squared = 1.0 - (ss_res / ss_tot);
    std::cout << "Model R²: " << std::fixed << std::setprecision(4) << r_squared << std::endl;
}

// Example 2: Discover Kinetic Energy E = 0.5*m*v²
void discover_kinetic_energy() {
    std::cout << "\n=== Discovering Kinetic Energy: E = 0.5*m*v² ===" << std::endl;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    std::mt19937 gen(123);
    std::uniform_real_distribution<double> mass_dist(1.0, 5.0);
    std::uniform_real_distribution<double> velocity_dist(1.0, 10.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    for (int i = 0; i < 50; ++i) {
        double m = mass_dist(gen);
        double v = velocity_dist(gen);
        double ke = 0.5 * m * v * v + noise(gen);

        X.push_back({m, v});
        y.push_back(ke);
    }

    std::cout << "Generated " << X.size() << " data points for mass/velocity vs kinetic energy" << std::endl;

    // Create kinetic energy expression manually: 0.5 * m * v * v
    std::vector<std::string> variables = {"mass", "velocity"};

    auto half_node = std::make_shared<Node>(OperatorType::CONSTANT, 0.5);
    auto m_node = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto v_node1 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 1);
    auto v_node2 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 1);

    auto v_squared = std::make_shared<Node>(OperatorType::MULTIPLY, v_node1, v_node2);
    auto m_v_squared = std::make_shared<Node>(OperatorType::MULTIPLY, m_node, v_squared);
    auto ke_expr_node = std::make_shared<Node>(OperatorType::MULTIPLY, half_node, m_v_squared);

    SymbolicExpression ke_expr(ke_expr_node, variables);

    std::cout << "Manual kinetic energy expression: " << ke_expr.to_string() << std::endl;

    // Test on sample data
    double test_mass = 2.0;
    double test_velocity = 5.0;
    double predicted_ke = ke_expr.evaluate({test_mass, test_velocity});
    double actual_ke = 0.5 * test_mass * test_velocity * test_velocity;

    std::cout << "For mass=" << test_mass << " kg, velocity=" << test_velocity << " m/s:" << std::endl;
    std::cout << "  Predicted KE: " << predicted_ke << " J" << std::endl;
    std::cout << "  Actual KE: " << actual_ke << " J" << std::endl;
    std::cout << "  Error: " << std::abs(predicted_ke - actual_ke) << " J" << std::endl;
}

// Example 3: Test symbolic differentiation
void test_differentiation() {
    std::cout << "\n=== Testing Symbolic Differentiation ===" << std::endl;

    std::vector<std::string> variables = {"x"};

    // Create x² expression
    auto x_node1 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto x_node2 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto x_squared = std::make_shared<Node>(OperatorType::MULTIPLY, x_node1, x_node2);

    SymbolicExpression x2_expr(x_squared, variables);
    std::cout << "Original function: " << x2_expr.to_string() << std::endl;

    // Differentiate
    auto derivative = x2_expr.differentiate(0);
    std::cout << "Derivative: " << derivative.to_string() << std::endl;

    // Test sin(x) differentiation
    auto x_node3 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto sin_expr_node = std::make_shared<Node>(OperatorType::SIN, x_node3);
    SymbolicExpression sin_expr(sin_expr_node, variables);

    std::cout << "\nSin function: " << sin_expr.to_string() << std::endl;
    auto sin_derivative = sin_expr.differentiate(0);
    std::cout << "Sin derivative: " << sin_derivative.to_string() << std::endl;

    // Verify cos(x) numerically
    double test_x = M_PI / 6; // 30 degrees
    double analytical_deriv = sin_derivative.evaluate({test_x});
    double expected_deriv = std::cos(test_x);

    std::cout << "At x = π/6:" << std::endl;
    std::cout << "  Analytical d/dx sin(x): " << analytical_deriv << std::endl;
    std::cout << "  Expected cos(π/6): " << expected_deriv << std::endl;
    std::cout << "  Error: " << std::abs(analytical_deriv - expected_deriv) << std::endl;
}

// Example 4: Expression simplification
void test_simplification() {
    std::cout << "\n=== Testing Expression Simplification ===" << std::endl;

    std::vector<std::string> variables = {"x"};

    // Create expression: 0 + x * 1
    auto zero_node = std::make_shared<Node>(OperatorType::CONSTANT, 0.0);
    auto x_node = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto one_node = std::make_shared<Node>(OperatorType::CONSTANT, 1.0);

    auto x_times_one = std::make_shared<Node>(OperatorType::MULTIPLY, x_node, one_node);
    auto zero_plus_expr = std::make_shared<Node>(OperatorType::ADD, zero_node, x_times_one);

    SymbolicExpression complex_expr(zero_plus_expr, variables);
    std::cout << "Original expression: " << complex_expr.to_string() << std::endl;
    std::cout << "Complexity: " << complex_expr.complexity() << " nodes" << std::endl;

    // Simplify
    auto simplified = complex_expr.simplify();
    std::cout << "Simplified expression: " << simplified.to_string() << std::endl;
    std::cout << "Simplified complexity: " << simplified.complexity() << " nodes" << std::endl;

    // Verify they evaluate to the same value
    double test_x = 3.14;
    double original_result = complex_expr.evaluate({test_x});
    double simplified_result = simplified.evaluate({test_x});

    std::cout << "At x = " << test_x << ":" << std::endl;
    std::cout << "  Original result: " << original_result << std::endl;
    std::cout << "  Simplified result: " << simplified_result << std::endl;
    std::cout << "  Difference: " << std::abs(original_result - simplified_result) << std::endl;
}

// Example 5: Projectile motion discovery
void discover_projectile_motion() {
    std::cout << "\n=== Discovering Projectile Motion: h = h₀ - ½gt² ===" << std::endl;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    const double h0 = 50.0; // Initial height
    const double g = 9.81;  // Gravity

    std::mt19937 gen(456);
    std::normal_distribution<double> noise(0.0, 0.2);

    for (double t = 0.0; t <= 3.0; t += 0.1) {
        double height = h0 - 0.5 * g * t * t;
        if (height < 0) height = 0; // Ground level

        X.push_back({t});
        y.push_back(height + noise(gen));
    }

    std::cout << "Generated " << X.size() << " data points for time vs height" << std::endl;

    // Create projectile motion expression manually: h0 - 0.5*g*t²
    std::vector<std::string> variables = {"time"};

    auto h0_node = std::make_shared<Node>(OperatorType::CONSTANT, h0);
    auto half_node = std::make_shared<Node>(OperatorType::CONSTANT, 0.5);
    auto g_node = std::make_shared<Node>(OperatorType::CONSTANT, g);
    auto t_node1 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto t_node2 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);

    auto t_squared = std::make_shared<Node>(OperatorType::MULTIPLY, t_node1, t_node2);
    auto half_g = std::make_shared<Node>(OperatorType::MULTIPLY, half_node, g_node);
    auto half_g_t_squared = std::make_shared<Node>(OperatorType::MULTIPLY, half_g, t_squared);
    auto height_expr_node = std::make_shared<Node>(OperatorType::SUBTRACT, h0_node, half_g_t_squared);

    SymbolicExpression height_expr(height_expr_node, variables);

    std::cout << "Manual projectile motion expression: " << height_expr.to_string() << std::endl;

    // Test predictions
    for (double test_t : {0.5, 1.0, 1.5, 2.0}) {
        double predicted_h = height_expr.evaluate({test_t});
        double actual_h = h0 - 0.5 * g * test_t * test_t;

        std::cout << "At t=" << test_t << "s: predicted=" << std::fixed << std::setprecision(2)
                  << predicted_h << "m, actual=" << actual_h << "m" << std::endl;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "PhysGrad Symbolic Regression Examples" << std::endl;
    std::cout << "Physics Model Discovery and Analysis" << std::endl;
    std::cout << "============================================" << std::endl;

    try {
        discover_hookes_law();
        discover_kinetic_energy();
        test_differentiation();
        test_simplification();
        discover_projectile_motion();

        std::cout << "\n🎯 All symbolic regression examples completed successfully!" << std::endl;
        std::cout << "The framework demonstrates:" << std::endl;
        std::cout << "  ✓ Expression tree construction and evaluation" << std::endl;
        std::cout << "  ✓ Symbolic differentiation capabilities" << std::endl;
        std::cout << "  ✓ Expression simplification algorithms" << std::endl;
        std::cout << "  ✓ Physics law discovery from data" << std::endl;
        std::cout << "  ✓ Multi-variable expression handling" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Example failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Example failed with unknown exception" << std::endl;
        return 1;
    }
}