#pragma once

#include <vector>
#include <string>
#include <random>
#include <functional>
#include <memory>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <queue>
#include <set>
#include <thread>
#include <future>
#include <mutex>

namespace physgrad {

enum class OperatorType {
    CONSTANT,
    VARIABLE,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER,
    SIN,
    COS,
    EXP,
    LOG,
    SQRT,
    ABS,
    TANH
};

struct Node {
    OperatorType type;
    double value;              // For constants
    size_t variable_index;     // For variables
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node(OperatorType t, double v = 0.0, size_t var_idx = 0)
        : type(t), value(v), variable_index(var_idx), left(nullptr), right(nullptr) {}

    Node(OperatorType t, std::shared_ptr<Node> l, std::shared_ptr<Node> r = nullptr)
        : type(t), value(0.0), variable_index(0), left(l), right(r) {}

    bool is_leaf() const {
        return type == OperatorType::CONSTANT || type == OperatorType::VARIABLE;
    }

    bool is_unary() const {
        return type == OperatorType::SIN || type == OperatorType::COS ||
               type == OperatorType::EXP || type == OperatorType::LOG ||
               type == OperatorType::SQRT || type == OperatorType::ABS ||
               type == OperatorType::TANH;
    }

    int arity() const {
        if (is_leaf()) return 0;
        if (is_unary()) return 1;
        return 2;
    }

    std::shared_ptr<Node> clone() const {
        auto new_node = std::make_shared<Node>(type, value, variable_index);
        if (left) new_node->left = left->clone();
        if (right) new_node->right = right->clone();
        return new_node;
    }
};

using ExpressionTree = std::shared_ptr<Node>;

class SymbolicExpression {
private:
    ExpressionTree root_;
    std::vector<std::string> variable_names_;

public:
    SymbolicExpression(ExpressionTree root, const std::vector<std::string>& var_names)
        : root_(root), variable_names_(var_names) {}

    // Copy constructor
    SymbolicExpression(const SymbolicExpression& other)
        : root_(other.root_), variable_names_(other.variable_names_) {}

    // Assignment operator
    SymbolicExpression& operator=(const SymbolicExpression& other) {
        if (this != &other) {
            root_ = other.root_;
            variable_names_ = other.variable_names_;
        }
        return *this;
    }

    // Move constructor
    SymbolicExpression(SymbolicExpression&& other) noexcept
        : root_(std::move(other.root_)), variable_names_(std::move(other.variable_names_)) {}

    // Move assignment
    SymbolicExpression& operator=(SymbolicExpression&& other) noexcept {
        if (this != &other) {
            root_ = std::move(other.root_);
            variable_names_ = std::move(other.variable_names_);
        }
        return *this;
    }

    double evaluate(const std::vector<double>& variables) const {
        return evaluate_node(root_, variables);
    }

    std::vector<double> evaluate_batch(const std::vector<std::vector<double>>& variable_sets) const {
        std::vector<double> results;
        results.reserve(variable_sets.size());

        for (const auto& variables : variable_sets) {
            results.push_back(evaluate(variables));
        }

        return results;
    }

    std::string to_string() const {
        return node_to_string(root_);
    }

    std::string to_latex() const {
        return node_to_latex(root_);
    }

    size_t complexity() const {
        return count_nodes(root_);
    }

    size_t depth() const {
        return compute_depth(root_);
    }

    std::vector<double> get_constants() const {
        std::vector<double> constants;
        collect_constants(root_, constants);
        return constants;
    }

    void set_constants(const std::vector<double>& new_constants) {
        size_t index = 0;
        update_constants(root_, new_constants, index);
    }

    ExpressionTree get_tree() const { return root_; }

    SymbolicExpression differentiate(size_t variable_index) const {
        auto derivative_tree = differentiate_node(root_, variable_index);
        return SymbolicExpression(derivative_tree, variable_names_);
    }

    SymbolicExpression simplify() const {
        auto simplified_tree = simplify_node(root_);
        return SymbolicExpression(simplified_tree, variable_names_);
    }

private:
    double evaluate_node(const ExpressionTree& node, const std::vector<double>& variables) const {
        if (!node) return 0.0;

        switch (node->type) {
            case OperatorType::CONSTANT:
                return node->value;

            case OperatorType::VARIABLE:
                if (node->variable_index < variables.size()) {
                    return variables[node->variable_index];
                }
                return 0.0;

            case OperatorType::ADD:
                return evaluate_node(node->left, variables) + evaluate_node(node->right, variables);

            case OperatorType::SUBTRACT:
                return evaluate_node(node->left, variables) - evaluate_node(node->right, variables);

            case OperatorType::MULTIPLY:
                return evaluate_node(node->left, variables) * evaluate_node(node->right, variables);

            case OperatorType::DIVIDE: {
                double denominator = evaluate_node(node->right, variables);
                if (std::abs(denominator) < 1e-12) return 1e6; // Avoid division by zero
                return evaluate_node(node->left, variables) / denominator;
            }

            case OperatorType::POWER: {
                double base = evaluate_node(node->left, variables);
                double exponent = evaluate_node(node->right, variables);
                if (base < 0 && std::abs(exponent - std::round(exponent)) > 1e-10) {
                    return std::nan(""); // Avoid complex numbers
                }
                return std::pow(base, exponent);
            }

            case OperatorType::SIN:
                return std::sin(evaluate_node(node->left, variables));

            case OperatorType::COS:
                return std::cos(evaluate_node(node->left, variables));

            case OperatorType::EXP: {
                double arg = evaluate_node(node->left, variables);
                if (arg > 700) return 1e6; // Prevent overflow
                return std::exp(arg);
            }

            case OperatorType::LOG: {
                double arg = evaluate_node(node->left, variables);
                if (arg <= 0) return -1e6; // Handle invalid log
                return std::log(arg);
            }

            case OperatorType::SQRT: {
                double arg = evaluate_node(node->left, variables);
                if (arg < 0) return std::nan(""); // Avoid complex numbers
                return std::sqrt(arg);
            }

            case OperatorType::ABS:
                return std::abs(evaluate_node(node->left, variables));

            case OperatorType::TANH:
                return std::tanh(evaluate_node(node->left, variables));

            default:
                return 0.0;
        }
    }

    std::string node_to_string(const ExpressionTree& node) const {
        if (!node) return "0";

        switch (node->type) {
            case OperatorType::CONSTANT:
                return std::to_string(node->value);

            case OperatorType::VARIABLE:
                if (node->variable_index < variable_names_.size()) {
                    return variable_names_[node->variable_index];
                }
                return "x" + std::to_string(node->variable_index);

            case OperatorType::ADD:
                return "(" + node_to_string(node->left) + " + " + node_to_string(node->right) + ")";

            case OperatorType::SUBTRACT:
                return "(" + node_to_string(node->left) + " - " + node_to_string(node->right) + ")";

            case OperatorType::MULTIPLY:
                return "(" + node_to_string(node->left) + " * " + node_to_string(node->right) + ")";

            case OperatorType::DIVIDE:
                return "(" + node_to_string(node->left) + " / " + node_to_string(node->right) + ")";

            case OperatorType::POWER:
                return "(" + node_to_string(node->left) + " ^ " + node_to_string(node->right) + ")";

            case OperatorType::SIN:
                return "sin(" + node_to_string(node->left) + ")";

            case OperatorType::COS:
                return "cos(" + node_to_string(node->left) + ")";

            case OperatorType::EXP:
                return "exp(" + node_to_string(node->left) + ")";

            case OperatorType::LOG:
                return "log(" + node_to_string(node->left) + ")";

            case OperatorType::SQRT:
                return "sqrt(" + node_to_string(node->left) + ")";

            case OperatorType::ABS:
                return "abs(" + node_to_string(node->left) + ")";

            case OperatorType::TANH:
                return "tanh(" + node_to_string(node->left) + ")";

            default:
                return "unknown";
        }
    }

    std::string node_to_latex(const ExpressionTree& node) const {
        if (!node) return "0";

        switch (node->type) {
            case OperatorType::CONSTANT:
                return std::to_string(node->value);

            case OperatorType::VARIABLE:
                if (node->variable_index < variable_names_.size()) {
                    return variable_names_[node->variable_index];
                }
                return "x_{" + std::to_string(node->variable_index) + "}";

            case OperatorType::ADD:
                return "(" + node_to_latex(node->left) + " + " + node_to_latex(node->right) + ")";

            case OperatorType::SUBTRACT:
                return "(" + node_to_latex(node->left) + " - " + node_to_latex(node->right) + ")";

            case OperatorType::MULTIPLY:
                return "(" + node_to_latex(node->left) + " \\cdot " + node_to_latex(node->right) + ")";

            case OperatorType::DIVIDE:
                return "\\frac{" + node_to_latex(node->left) + "}{" + node_to_latex(node->right) + "}";

            case OperatorType::POWER:
                return "(" + node_to_latex(node->left) + ")^{" + node_to_latex(node->right) + "}";

            case OperatorType::SIN:
                return "\\sin(" + node_to_latex(node->left) + ")";

            case OperatorType::COS:
                return "\\cos(" + node_to_latex(node->left) + ")";

            case OperatorType::EXP:
                return "e^{" + node_to_latex(node->left) + "}";

            case OperatorType::LOG:
                return "\\ln(" + node_to_latex(node->left) + ")";

            case OperatorType::SQRT:
                return "\\sqrt{" + node_to_latex(node->left) + "}";

            case OperatorType::ABS:
                return "|" + node_to_latex(node->left) + "|";

            case OperatorType::TANH:
                return "\\tanh(" + node_to_latex(node->left) + ")";

            default:
                return "unknown";
        }
    }

    size_t count_nodes(const ExpressionTree& node) const {
        if (!node) return 0;
        size_t count = 1;
        if (node->left) count += count_nodes(node->left);
        if (node->right) count += count_nodes(node->right);
        return count;
    }

    size_t compute_depth(const ExpressionTree& node) const {
        if (!node) return 0;
        size_t left_depth = node->left ? compute_depth(node->left) : 0;
        size_t right_depth = node->right ? compute_depth(node->right) : 0;
        return 1 + std::max(left_depth, right_depth);
    }

    void collect_constants(const ExpressionTree& node, std::vector<double>& constants) const {
        if (!node) return;
        if (node->type == OperatorType::CONSTANT) {
            constants.push_back(node->value);
        }
        if (node->left) collect_constants(node->left, constants);
        if (node->right) collect_constants(node->right, constants);
    }

    void update_constants(ExpressionTree& node, const std::vector<double>& constants, size_t& index) {
        if (!node) return;
        if (node->type == OperatorType::CONSTANT && index < constants.size()) {
            node->value = constants[index++];
        }
        if (node->left) update_constants(node->left, constants, index);
        if (node->right) update_constants(node->right, constants, index);
    }

    ExpressionTree differentiate_node(const ExpressionTree& node, size_t var_index) const {
        if (!node) return std::make_shared<Node>(OperatorType::CONSTANT, 0.0);

        switch (node->type) {
            case OperatorType::CONSTANT:
                return std::make_shared<Node>(OperatorType::CONSTANT, 0.0);

            case OperatorType::VARIABLE:
                if (node->variable_index == var_index) {
                    return std::make_shared<Node>(OperatorType::CONSTANT, 1.0);
                } else {
                    return std::make_shared<Node>(OperatorType::CONSTANT, 0.0);
                }

            case OperatorType::ADD:
                return std::make_shared<Node>(OperatorType::ADD,
                    differentiate_node(node->left, var_index),
                    differentiate_node(node->right, var_index));

            case OperatorType::SUBTRACT:
                return std::make_shared<Node>(OperatorType::SUBTRACT,
                    differentiate_node(node->left, var_index),
                    differentiate_node(node->right, var_index));

            case OperatorType::MULTIPLY: {
                // Product rule: (uv)' = u'v + uv'
                auto u_prime = differentiate_node(node->left, var_index);
                auto v_prime = differentiate_node(node->right, var_index);
                auto term1 = std::make_shared<Node>(OperatorType::MULTIPLY, u_prime, node->right->clone());
                auto term2 = std::make_shared<Node>(OperatorType::MULTIPLY, node->left->clone(), v_prime);
                return std::make_shared<Node>(OperatorType::ADD, term1, term2);
            }

            case OperatorType::SIN: {
                // d/dx sin(u) = cos(u) * u'
                auto cos_u = std::make_shared<Node>(OperatorType::COS, node->left->clone());
                auto u_prime = differentiate_node(node->left, var_index);
                return std::make_shared<Node>(OperatorType::MULTIPLY, cos_u, u_prime);
            }

            case OperatorType::COS: {
                // d/dx cos(u) = -sin(u) * u'
                auto sin_u = std::make_shared<Node>(OperatorType::SIN, node->left->clone());
                auto u_prime = differentiate_node(node->left, var_index);
                auto neg_one = std::make_shared<Node>(OperatorType::CONSTANT, -1.0);
                auto neg_sin_u = std::make_shared<Node>(OperatorType::MULTIPLY, neg_one, sin_u);
                return std::make_shared<Node>(OperatorType::MULTIPLY, neg_sin_u, u_prime);
            }

            case OperatorType::EXP: {
                // d/dx exp(u) = exp(u) * u'
                auto exp_u = std::make_shared<Node>(OperatorType::EXP, node->left->clone());
                auto u_prime = differentiate_node(node->left, var_index);
                return std::make_shared<Node>(OperatorType::MULTIPLY, exp_u, u_prime);
            }

            default:
                // For other operations, return a zero derivative for simplicity
                return std::make_shared<Node>(OperatorType::CONSTANT, 0.0);
        }
    }

    ExpressionTree simplify_node(const ExpressionTree& node) const {
        if (!node) return nullptr;

        auto left = node->left ? simplify_node(node->left) : nullptr;
        auto right = node->right ? simplify_node(node->right) : nullptr;

        // Apply simplification rules
        switch (node->type) {
            case OperatorType::ADD:
                // 0 + x = x, x + 0 = x
                if (left && left->type == OperatorType::CONSTANT && left->value == 0.0) return right;
                if (right && right->type == OperatorType::CONSTANT && right->value == 0.0) return left;
                // Constant folding
                if (left && right && left->type == OperatorType::CONSTANT && right->type == OperatorType::CONSTANT) {
                    return std::make_shared<Node>(OperatorType::CONSTANT, left->value + right->value);
                }
                break;

            case OperatorType::MULTIPLY:
                // 0 * x = 0, x * 0 = 0
                if ((left && left->type == OperatorType::CONSTANT && left->value == 0.0) ||
                    (right && right->type == OperatorType::CONSTANT && right->value == 0.0)) {
                    return std::make_shared<Node>(OperatorType::CONSTANT, 0.0);
                }
                // 1 * x = x, x * 1 = x
                if (left && left->type == OperatorType::CONSTANT && left->value == 1.0) return right;
                if (right && right->type == OperatorType::CONSTANT && right->value == 1.0) return left;
                // Constant folding
                if (left && right && left->type == OperatorType::CONSTANT && right->type == OperatorType::CONSTANT) {
                    return std::make_shared<Node>(OperatorType::CONSTANT, left->value * right->value);
                }
                break;

            default:
                // For other operations, no simplification is applied
                break;
        }

        // Create new node with simplified children
        auto simplified = std::make_shared<Node>(node->type, node->value, node->variable_index);
        simplified->left = left;
        simplified->right = right;
        return simplified;
    }
};

struct Individual {
    SymbolicExpression expression;
    double fitness;
    double complexity_penalty;
    double adjusted_fitness;

    Individual(const SymbolicExpression& expr)
        : expression(expr), fitness(std::numeric_limits<double>::infinity()),
          complexity_penalty(0.0), adjusted_fitness(std::numeric_limits<double>::infinity()) {}

    // Copy constructor
    Individual(const Individual& other)
        : expression(other.expression), fitness(other.fitness),
          complexity_penalty(other.complexity_penalty), adjusted_fitness(other.adjusted_fitness) {}

    // Assignment operator
    Individual& operator=(const Individual& other) {
        if (this != &other) {
            expression = other.expression;
            fitness = other.fitness;
            complexity_penalty = other.complexity_penalty;
            adjusted_fitness = other.adjusted_fitness;
        }
        return *this;
    }

    // Move constructor
    Individual(Individual&& other) noexcept
        : expression(std::move(other.expression)), fitness(other.fitness),
          complexity_penalty(other.complexity_penalty), adjusted_fitness(other.adjusted_fitness) {}

    // Move assignment
    Individual& operator=(Individual&& other) noexcept {
        if (this != &other) {
            expression = std::move(other.expression);
            fitness = other.fitness;
            complexity_penalty = other.complexity_penalty;
            adjusted_fitness = other.adjusted_fitness;
        }
        return *this;
    }

    void compute_adjusted_fitness(double complexity_weight = 0.01) {
        complexity_penalty = complexity_weight * expression.complexity();
        adjusted_fitness = fitness + complexity_penalty;
    }
};

class GeneticProgramming {
private:
    std::vector<Individual> population_;
    std::vector<std::string> variable_names_;
    std::vector<OperatorType> function_set_;
    std::mt19937 generator_;
    std::mutex population_mutex_;

    // GP parameters
    size_t population_size_;
    size_t max_generations_;
    double mutation_rate_;
    double crossover_rate_;
    size_t max_depth_;
    double complexity_weight_;

public:
    GeneticProgramming(const std::vector<std::string>& var_names,
                      size_t pop_size = 100,
                      size_t max_gen = 50,
                      double mut_rate = 0.1,
                      double cross_rate = 0.9,
                      size_t max_d = 7,
                      double complexity_w = 0.01)
        : variable_names_(var_names), generator_(std::random_device{}()),
          population_size_(pop_size), max_generations_(max_gen),
          mutation_rate_(mut_rate), crossover_rate_(cross_rate),
          max_depth_(max_d), complexity_weight_(complexity_w) {

        // Initialize function set
        function_set_ = {
            OperatorType::ADD, OperatorType::SUBTRACT, OperatorType::MULTIPLY, OperatorType::DIVIDE,
            OperatorType::SIN, OperatorType::COS, OperatorType::EXP, OperatorType::LOG,
            OperatorType::POWER, OperatorType::SQRT, OperatorType::ABS, OperatorType::TANH
        };

        population_.reserve(population_size_);
    }

    SymbolicExpression evolve(const std::vector<std::vector<double>>& X,
                             const std::vector<double>& y,
                             std::function<void(size_t, double)> progress_callback = nullptr) {

        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }

        // Initialize population
        initialize_population();

        double best_fitness = std::numeric_limits<double>::infinity();
        SymbolicExpression best_expression = population_[0].expression;

        for (size_t generation = 0; generation < max_generations_; ++generation) {
            // Evaluate fitness
            evaluate_population(X, y);

            // Find best individual
            auto best_it = std::min_element(population_.begin(), population_.end(),
                [](const Individual& a, const Individual& b) {
                    return a.adjusted_fitness < b.adjusted_fitness;
                });

            if (best_it->adjusted_fitness < best_fitness) {
                best_fitness = best_it->adjusted_fitness;
                best_expression = best_it->expression;
            }

            if (progress_callback) {
                progress_callback(generation, best_fitness);
            }

            // Early stopping if perfect fit
            if (best_fitness < 1e-10) break;

            // Create next generation
            evolve_population();
        }

        return best_expression;
    }

    std::vector<SymbolicExpression> evolve_pareto_front(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        std::function<void(size_t, std::vector<Individual>&)> progress_callback = nullptr) {

        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }

        initialize_population();

        std::vector<Individual> pareto_front;

        for (size_t generation = 0; generation < max_generations_; ++generation) {
            evaluate_population(X, y);

            // Update Pareto front
            update_pareto_front(pareto_front);

            if (progress_callback) {
                progress_callback(generation, pareto_front);
            }

            // Multi-objective selection and evolution
            evolve_population_pareto();
        }

        std::vector<SymbolicExpression> result;
        for (const auto& ind : pareto_front) {
            result.push_back(ind.expression);
        }

        return result;
    }

private:
    void initialize_population() {
        population_.clear();
        population_.reserve(population_size_);

        for (size_t i = 0; i < population_size_; ++i) {
            auto tree = generate_random_tree(max_depth_);
            if (!tree) {
                // Ensure we always have a valid tree
                tree = std::make_shared<Node>(OperatorType::CONSTANT, 1.0);
            }
            SymbolicExpression expr(tree, variable_names_);
            population_.emplace_back(expr);
        }
    }

    ExpressionTree generate_random_tree(size_t max_depth, size_t current_depth = 0) {
        std::uniform_real_distribution<double> const_dist(-10.0, 10.0);
        std::uniform_int_distribution<size_t> var_dist(0, variable_names_.size() - 1);

        // Force leaf nodes at maximum depth
        if (current_depth >= max_depth) {
            std::uniform_int_distribution<int> leaf_type(0, 1);
            if (leaf_type(generator_) == 0) {
                return std::make_shared<Node>(OperatorType::CONSTANT, const_dist(generator_));
            } else {
                return std::make_shared<Node>(OperatorType::VARIABLE, 0.0, var_dist(generator_));
            }
        }

        // Choose random function or terminal
        std::uniform_real_distribution<double> grow_prob(0.0, 1.0);
        double terminal_prob = static_cast<double>(current_depth) / max_depth;

        if (grow_prob(generator_) < terminal_prob) {
            // Generate terminal
            std::uniform_int_distribution<int> terminal_type(0, 1);
            if (terminal_type(generator_) == 0) {
                return std::make_shared<Node>(OperatorType::CONSTANT, const_dist(generator_));
            } else {
                return std::make_shared<Node>(OperatorType::VARIABLE, 0.0, var_dist(generator_));
            }
        } else {
            // Generate function
            std::uniform_int_distribution<size_t> func_dist(0, function_set_.size() - 1);
            OperatorType op = function_set_[func_dist(generator_)];

            auto node = std::make_shared<Node>(op);

            if (node->is_unary()) {
                node->left = generate_random_tree(max_depth, current_depth + 1);
            } else {
                node->left = generate_random_tree(max_depth, current_depth + 1);
                node->right = generate_random_tree(max_depth, current_depth + 1);
            }

            return node;
        }
    }

    void evaluate_population(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        // Evaluate sequentially to avoid threading issues
        for (auto& individual : population_) {
            evaluate_individual(individual, X, y);
        }
    }

    void evaluate_individual(Individual& individual, const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) {
        double mse = 0.0;
        size_t valid_count = 0;

        for (size_t i = 0; i < X.size(); ++i) {
            try {
                double prediction = individual.expression.evaluate(X[i]);
                if (std::isfinite(prediction)) {
                    double error = prediction - y[i];
                    mse += error * error;
                    valid_count++;
                } else {
                    mse += 1e6; // Heavy penalty for invalid outputs
                }
            } catch (...) {
                mse += 1e6; // Heavy penalty for evaluation errors
            }
        }

        if (valid_count > 0) {
            individual.fitness = mse / valid_count;
        } else {
            individual.fitness = 1e6;
        }

        individual.compute_adjusted_fitness(complexity_weight_);
    }

    void evolve_population() {
        std::vector<Individual> new_population;
        new_population.reserve(population_size_);

        // Elitism: keep best 10%
        std::sort(population_.begin(), population_.end(),
            [](const Individual& a, const Individual& b) {
                return a.adjusted_fitness < b.adjusted_fitness;
            });

        size_t elite_count = population_size_ / 10;
        for (size_t i = 0; i < elite_count; ++i) {
            new_population.push_back(population_[i]);
        }

        // Generate rest through crossover and mutation
        while (new_population.size() < population_size_) {
            std::uniform_real_distribution<double> prob(0.0, 1.0);

            if (prob(generator_) < crossover_rate_) {
                auto parent1 = tournament_selection();
                auto parent2 = tournament_selection();
                auto child = crossover(parent1.expression, parent2.expression);
                new_population.emplace_back(child);
            } else {
                auto parent = tournament_selection();
                auto child = mutate(parent.expression);
                new_population.emplace_back(child);
            }
        }

        population_ = std::move(new_population);
    }

    void evolve_population_pareto() {
        // Multi-objective evolution using NSGA-II-like approach
        std::vector<Individual> new_population;
        new_population.reserve(population_size_);

        // Non-dominated sorting
        auto fronts = non_dominated_sort();

        for (const auto& front : fronts) {
            if (new_population.size() + front.size() <= population_size_) {
                for (size_t idx : front) {
                    new_population.push_back(population_[idx]);
                }
            } else {
                // Fill remaining slots with crowding distance
                auto selected = crowding_distance_selection(front, population_size_ - new_population.size());
                for (size_t idx : selected) {
                    new_population.push_back(population_[idx]);
                }
                break;
            }
        }

        // Generate offspring
        while (new_population.size() < population_size_) {
            auto parent1 = tournament_selection_pareto();
            auto parent2 = tournament_selection_pareto();
            auto child = crossover(parent1.expression, parent2.expression);
            new_population.emplace_back(child);
        }

        population_ = std::move(new_population);
    }

    Individual tournament_selection(size_t tournament_size = 3) {
        std::uniform_int_distribution<size_t> dist(0, population_.size() - 1);

        Individual best = population_[dist(generator_)];
        for (size_t i = 1; i < tournament_size; ++i) {
            Individual candidate = population_[dist(generator_)];
            if (candidate.adjusted_fitness < best.adjusted_fitness) {
                best = candidate;
            }
        }

        return best;
    }

    Individual tournament_selection_pareto(size_t tournament_size = 3) {
        std::uniform_int_distribution<size_t> dist(0, population_.size() - 1);

        Individual best = population_[dist(generator_)];
        for (size_t i = 1; i < tournament_size; ++i) {
            Individual candidate = population_[dist(generator_)];
            if (dominates(candidate, best)) {
                best = candidate;
            }
        }

        return best;
    }

    SymbolicExpression crossover(const SymbolicExpression& parent1, const SymbolicExpression& parent2) {
        auto tree1 = parent1.get_tree()->clone();
        auto tree2 = parent2.get_tree()->clone();

        // Find random crossover points
        auto nodes1 = collect_nodes(tree1);
        auto nodes2 = collect_nodes(tree2);

        if (nodes1.empty() || nodes2.empty()) return parent1;

        std::uniform_int_distribution<size_t> dist1(0, nodes1.size() - 1);
        std::uniform_int_distribution<size_t> dist2(0, nodes2.size() - 1);

        size_t point1 = dist1(generator_);
        size_t point2 = dist2(generator_);

        // Perform subtree crossover
        auto subtree = (*nodes2[point2])->clone();
        *nodes1[point1] = subtree;

        return SymbolicExpression(tree1, variable_names_);
    }

    SymbolicExpression mutate(const SymbolicExpression& parent) {
        auto tree = parent.get_tree()->clone();
        auto nodes = collect_nodes(tree);

        if (nodes.empty()) return parent;

        std::uniform_int_distribution<size_t> node_dist(0, nodes.size() - 1);
        std::uniform_real_distribution<double> mut_type(0.0, 1.0);

        size_t node_index = node_dist(generator_);
        auto& node_ptr = nodes[node_index];

        if (mut_type(generator_) < 0.5) {
            // Point mutation
            if ((*node_ptr)->type == OperatorType::CONSTANT) {
                std::normal_distribution<double> const_mut((*node_ptr)->value, std::abs((*node_ptr)->value) * 0.1 + 0.1);
                (*node_ptr)->value = const_mut(generator_);
            } else if ((*node_ptr)->type == OperatorType::VARIABLE) {
                std::uniform_int_distribution<size_t> var_dist(0, variable_names_.size() - 1);
                (*node_ptr)->variable_index = var_dist(generator_);
            } else {
                // Change operator
                std::uniform_int_distribution<size_t> op_dist(0, function_set_.size() - 1);
                (*node_ptr)->type = function_set_[op_dist(generator_)];
            }
        } else {
            // Subtree mutation
            size_t remaining_depth = max_depth_ - compute_node_depth(tree, node_ptr);
            auto new_subtree = generate_random_tree(std::max(size_t(1), remaining_depth));
            *node_ptr = new_subtree;
        }

        return SymbolicExpression(tree, variable_names_);
    }

    std::vector<ExpressionTree*> collect_nodes(ExpressionTree& tree) {
        std::vector<ExpressionTree*> nodes;
        collect_nodes_recursive(tree, nodes);
        return nodes;
    }

    void collect_nodes_recursive(ExpressionTree& node, std::vector<ExpressionTree*>& nodes) {
        if (!node) return;
        nodes.push_back(&node);
        if (node->left) collect_nodes_recursive(node->left, nodes);
        if (node->right) collect_nodes_recursive(node->right, nodes);
    }

    size_t compute_node_depth(const ExpressionTree& tree, const ExpressionTree* target) {
        return compute_node_depth_recursive(tree, target, 0);
    }

    size_t compute_node_depth_recursive(const ExpressionTree& node, const ExpressionTree* target, size_t current_depth) {
        if (!node) return 0;
        if (node.get() == target->get()) return current_depth;

        size_t left_depth = 0, right_depth = 0;
        if (node->left) {
            left_depth = compute_node_depth_recursive(node->left, target, current_depth + 1);
        }
        if (node->right) {
            right_depth = compute_node_depth_recursive(node->right, target, current_depth + 1);
        }

        return std::max(left_depth, right_depth);
    }

    std::vector<std::vector<size_t>> non_dominated_sort() {
        std::vector<std::vector<size_t>> fronts;
        std::vector<size_t> domination_count(population_.size(), 0);
        std::vector<std::vector<size_t>> dominated_solutions(population_.size());

        std::vector<size_t> front;

        // Find first front
        for (size_t i = 0; i < population_.size(); ++i) {
            for (size_t j = 0; j < population_.size(); ++j) {
                if (i != j) {
                    if (dominates(population_[i], population_[j])) {
                        dominated_solutions[i].push_back(j);
                    } else if (dominates(population_[j], population_[i])) {
                        domination_count[i]++;
                    }
                }
            }
            if (domination_count[i] == 0) {
                front.push_back(i);
            }
        }

        fronts.push_back(front);

        // Find subsequent fronts
        size_t front_index = 0;
        while (!fronts[front_index].empty()) {
            std::vector<size_t> next_front;
            for (size_t i : fronts[front_index]) {
                for (size_t j : dominated_solutions[i]) {
                    domination_count[j]--;
                    if (domination_count[j] == 0) {
                        next_front.push_back(j);
                    }
                }
            }
            if (!next_front.empty()) {
                fronts.push_back(next_front);
            }
            front_index++;
        }

        return fronts;
    }

    bool dominates(const Individual& a, const Individual& b) {
        return (a.fitness <= b.fitness && a.expression.complexity() <= b.expression.complexity()) &&
               (a.fitness < b.fitness || a.expression.complexity() < b.expression.complexity());
    }

    std::vector<size_t> crowding_distance_selection(const std::vector<size_t>& front, size_t count) {
        if (count >= front.size()) return front;

        std::vector<std::pair<size_t, double>> distances;
        for (size_t idx : front) {
            distances.push_back({idx, 0.0});
        }

        // Sort and calculate crowding distance (simplified)
        std::sort(distances.begin(), distances.end(),
            [this](const auto& a, const auto& b) {
                return population_[a.first].fitness < population_[b.first].fitness;
            });

        // Boundary solutions get infinite distance
        distances[0].second = std::numeric_limits<double>::infinity();
        distances.back().second = std::numeric_limits<double>::infinity();

        for (size_t i = 1; i < distances.size() - 1; ++i) {
            distances[i].second = population_[distances[i+1].first].fitness - population_[distances[i-1].first].fitness;
        }

        // Sort by crowding distance and select top ones
        std::sort(distances.begin(), distances.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second;
            });

        std::vector<size_t> selected;
        for (size_t i = 0; i < count; ++i) {
            selected.push_back(distances[i].first);
        }

        return selected;
    }

    void update_pareto_front(std::vector<Individual>& pareto_front) {
        // Add non-dominated solutions from current population
        for (const auto& ind : population_) {
            bool dominated = false;
            std::vector<size_t> to_remove;

            for (size_t i = 0; i < pareto_front.size(); ++i) {
                if (dominates(pareto_front[i], ind)) {
                    dominated = true;
                    break;
                } else if (dominates(ind, pareto_front[i])) {
                    to_remove.push_back(i);
                }
            }

            if (!dominated) {
                // Remove dominated solutions
                for (int i = to_remove.size() - 1; i >= 0; --i) {
                    pareto_front.erase(pareto_front.begin() + to_remove[i]);
                }
                pareto_front.push_back(ind);
            }
        }
    }
};

class PhysicsModelDiscovery {
private:
    std::vector<std::string> variable_names_;
    GeneticProgramming gp_;

public:
    PhysicsModelDiscovery(const std::vector<std::string>& variables)
        : variable_names_(variables), gp_(variables, 200, 100, 0.15, 0.85, 8, 0.02) {}

    struct DiscoveredModel {
        SymbolicExpression expression;
        double mse;
        double r_squared;
        size_t complexity;
        std::string equation_string;
        std::string latex_string;

        DiscoveredModel(const SymbolicExpression& expr, double error, double r2)
            : expression(expr), mse(error), r_squared(r2), complexity(expr.complexity()),
              equation_string(expr.to_string()), latex_string(expr.to_latex()) {}
    };

    DiscoveredModel discover_model(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y,
                                  std::function<void(size_t, double)> progress_callback = nullptr) {

        auto best_expression = gp_.evolve(X, y, progress_callback);

        // Evaluate final model
        double mse = compute_mse(best_expression, X, y);
        double r_squared = compute_r_squared(best_expression, X, y);

        return DiscoveredModel(best_expression, mse, r_squared);
    }

    std::vector<DiscoveredModel> discover_pareto_models(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        std::function<void(size_t, std::vector<SymbolicExpression>&)> progress_callback = nullptr) {

        auto pareto_expressions = gp_.evolve_pareto_front(X, y,
            [progress_callback](size_t gen, std::vector<Individual>& front) {
                if (progress_callback) {
                    std::vector<SymbolicExpression> expressions;
                    for (const auto& ind : front) {
                        expressions.push_back(ind.expression);
                    }
                    progress_callback(gen, expressions);
                }
            });

        std::vector<DiscoveredModel> models;
        for (const auto& expr : pareto_expressions) {
            double mse = compute_mse(expr, X, y);
            double r_squared = compute_r_squared(expr, X, y);
            models.emplace_back(expr, mse, r_squared);
        }

        // Sort by complexity for presentation
        std::sort(models.begin(), models.end(),
            [](const DiscoveredModel& a, const DiscoveredModel& b) {
                return a.complexity < b.complexity;
            });

        return models;
    }

private:
    double compute_mse(const SymbolicExpression& expr,
                      const std::vector<std::vector<double>>& X,
                      const std::vector<double>& y) {
        double mse = 0.0;
        size_t valid_count = 0;

        for (size_t i = 0; i < X.size(); ++i) {
            try {
                double prediction = expr.evaluate(X[i]);
                if (std::isfinite(prediction)) {
                    double error = prediction - y[i];
                    mse += error * error;
                    valid_count++;
                }
            } catch (...) {
                // Skip invalid evaluations
            }
        }

        return valid_count > 0 ? mse / valid_count : std::numeric_limits<double>::infinity();
    }

    double compute_r_squared(const SymbolicExpression& expr,
                           const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) {
        if (y.empty()) return 0.0;

        // Compute mean of y
        double y_mean = 0.0;
        for (double val : y) y_mean += val;
        y_mean /= y.size();

        double ss_tot = 0.0;  // Total sum of squares
        double ss_res = 0.0;  // Residual sum of squares
        size_t valid_count = 0;

        for (size_t i = 0; i < X.size(); ++i) {
            try {
                double prediction = expr.evaluate(X[i]);
                if (std::isfinite(prediction)) {
                    ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
                    ss_res += (y[i] - prediction) * (y[i] - prediction);
                    valid_count++;
                }
            } catch (...) {
                // Skip invalid evaluations
            }
        }

        if (valid_count == 0 || ss_tot == 0.0) return 0.0;
        return 1.0 - (ss_res / ss_tot);
    }
};

} // namespace physgrad