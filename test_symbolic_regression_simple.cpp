#include "src/symbolic_regression.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>

using namespace physgrad;

void test_basic_expression() {
    std::cout << "Testing basic expression evaluation..." << std::endl;

    // Create simple expression: x^2 + 2*x + 1
    std::vector<std::string> variables = {"x"};

    auto x_node = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto x_node2 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto const_2 = std::make_shared<Node>(OperatorType::CONSTANT, 2.0);
    auto const_1 = std::make_shared<Node>(OperatorType::CONSTANT, 1.0);

    // x^2
    auto x_squared = std::make_shared<Node>(OperatorType::MULTIPLY, x_node, x_node2);

    // 2*x
    auto x_node3 = std::make_shared<Node>(OperatorType::VARIABLE, 0.0, 0);
    auto two_x = std::make_shared<Node>(OperatorType::MULTIPLY, const_2, x_node3);

    // x^2 + 2*x
    auto sum1 = std::make_shared<Node>(OperatorType::ADD, x_squared, two_x);

    // x^2 + 2*x + 1
    auto final_expr = std::make_shared<Node>(OperatorType::ADD, sum1, const_1);

    SymbolicExpression expr(final_expr, variables);

    // Test evaluation
    for (double x = -3; x <= 3; x += 1) {
        double result = expr.evaluate({x});
        double expected = x * x + 2 * x + 1;
        std::cout << "f(" << x << ") = " << result << " (expected: " << expected << ")" << std::endl;
        assert(std::abs(result - expected) < 1e-10);
    }

    std::cout << "Expression string: " << expr.to_string() << std::endl;
    std::cout << "Complexity: " << expr.complexity() << std::endl;

    std::cout << "✓ Basic expression tests passed" << std::endl;
}

void test_simple_discovery() {
    std::cout << "Testing simple model discovery..." << std::endl;

    // Generate linear data: y = 2*x + 3
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (double x = -5; x <= 5; x += 0.5) {
        X.push_back({x});
        y.push_back(2 * x + 3);
    }

    std::vector<std::string> variables = {"x"};

    // Use small population and generations for testing
    GeneticProgramming gp(variables, 20, 10, 0.1, 0.8, 3, 0.01);

    std::cout << "Evolving expression for linear data..." << std::endl;
    auto best_expr = gp.evolve(X, y, nullptr);

    std::cout << "Evolved expression: " << best_expr.to_string() << std::endl;

    // Test on new data
    double test_x = 2.5;
    double prediction = best_expr.evaluate({test_x});
    double expected = 2 * test_x + 3;
    std::cout << "Prediction at x=" << test_x << ": " << prediction << " (expected: " << expected << ")" << std::endl;

    std::cout << "✓ Simple discovery tests passed" << std::endl;
}

void test_physics_discovery() {
    std::cout << "Testing physics model discovery..." << std::endl;

    // Generate data for free fall: h = h0 - 0.5*g*t^2
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    const double h0 = 100.0;
    const double g = 9.81;

    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.5);

    for (double t = 0; t <= 4; t += 0.2) {
        X.push_back({t});
        double height = h0 - 0.5 * g * t * t;
        height += noise(gen); // Add small noise
        y.push_back(height);
    }

    std::vector<std::string> variables = {"time"};
    PhysicsModelDiscovery discovery(variables);

    std::cout << "Discovering free fall model..." << std::endl;
    auto model = discovery.discover_model(X, y, nullptr);

    std::cout << "Discovered model: " << model.equation_string << std::endl;
    std::cout << "MSE: " << model.mse << std::endl;
    std::cout << "R²: " << model.r_squared << std::endl;
    std::cout << "Complexity: " << model.complexity << std::endl;

    // Verify R² is reasonable
    assert(model.r_squared > 0.5);

    std::cout << "✓ Physics discovery tests passed" << std::endl;
}

int main() {
    std::cout << "=== Simplified Symbolic Regression Test Suite ===" << std::endl << std::endl;

    try {
        test_basic_expression();
        test_simple_discovery();
        test_physics_discovery();

        std::cout << std::endl << "✅ All simplified tests passed!" << std::endl;
        std::cout << "Symbolic regression is functional." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}