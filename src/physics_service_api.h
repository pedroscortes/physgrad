#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <thread>
#include <future>
#include <chrono>
#include <queue>
#include <atomic>
#include <random>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>

namespace physgrad {

// Forward declarations
class PhysicsSimulation;
class UncertaintyQuantification;
class SymbolicRegression;
class AutomaticExperimentDesign;

// Request/Response structures
struct APIRequest {
    std::string request_id;
    std::string service_type;
    std::string method;
    std::map<std::string, std::string> parameters;
    std::vector<double> input_data;
    std::chrono::system_clock::time_point timestamp;

    APIRequest() : timestamp(std::chrono::system_clock::now()) {
        generate_request_id();
    }

private:
    void generate_request_id() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);

        std::stringstream ss;
        ss << "req_";
        for (int i = 0; i < 8; ++i) {
            ss << std::hex << dis(gen);
        }
        request_id = ss.str();
    }
};

struct APIResponse {
    std::string request_id;
    int status_code;
    std::string status_message;
    std::map<std::string, std::string> metadata;
    std::vector<double> result_data;
    std::chrono::system_clock::time_point timestamp;
    double processing_time_ms;

    APIResponse() : timestamp(std::chrono::system_clock::now()), processing_time_ms(0.0) {}
};

enum class ServiceType {
    PHYSICS_SIMULATION,
    UNCERTAINTY_QUANTIFICATION,
    SYMBOLIC_REGRESSION,
    EXPERIMENT_DESIGN,
    MODEL_DISCOVERY,
    PARAMETER_OPTIMIZATION
};

enum class ServiceStatus {
    HEALTHY,
    DEGRADED,
    UNHEALTHY,
    MAINTENANCE
};

struct ServiceMetrics {
    size_t total_requests{0};
    size_t successful_requests{0};
    size_t failed_requests{0};
    double average_response_time_ms{0.0};
    size_t active_connections{0};
    std::chrono::system_clock::time_point last_request_time;
    ServiceStatus status{ServiceStatus::HEALTHY};

    void update_response_time(double new_time_ms) {
        double current_avg = average_response_time_ms;
        size_t total = total_requests;
        average_response_time_ms = (current_avg * (total - 1) + new_time_ms) / total;
    }

    double get_success_rate() const {
        size_t total = total_requests;
        if (total == 0) return 0.0;
        return static_cast<double>(successful_requests) / total * 100.0;
    }
};

class ServiceRegistry {
private:
    std::map<ServiceType, ServiceMetrics> service_metrics_;
    std::map<std::string, ServiceType> service_name_map_;
    std::mutex registry_mutex_;

public:
    ServiceRegistry() {
        // Initialize service mappings
        service_name_map_["physics_simulation"] = ServiceType::PHYSICS_SIMULATION;
        service_name_map_["uncertainty_quantification"] = ServiceType::UNCERTAINTY_QUANTIFICATION;
        service_name_map_["symbolic_regression"] = ServiceType::SYMBOLIC_REGRESSION;
        service_name_map_["experiment_design"] = ServiceType::EXPERIMENT_DESIGN;
        service_name_map_["model_discovery"] = ServiceType::MODEL_DISCOVERY;
        service_name_map_["parameter_optimization"] = ServiceType::PARAMETER_OPTIMIZATION;

        // Initialize metrics for each service
        for (const auto& pair : service_name_map_) {
            service_metrics_[pair.second] = ServiceMetrics{};
        }
    }

    ServiceType get_service_type(const std::string& service_name) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = service_name_map_.find(service_name);
        if (it != service_name_map_.end()) {
            return it->second;
        }
        throw std::invalid_argument("Unknown service: " + service_name);
    }

    void record_request(ServiceType service, double processing_time_ms, bool success) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto& metrics = service_metrics_[service];

        metrics.total_requests++;
        if (success) {
            metrics.successful_requests++;
        } else {
            metrics.failed_requests++;
        }

        metrics.update_response_time(processing_time_ms);
        metrics.last_request_time = std::chrono::system_clock::now();
    }

    ServiceMetrics get_metrics(ServiceType service) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return service_metrics_[service];
    }

    std::map<ServiceType, ServiceMetrics> get_all_metrics() {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return service_metrics_;
    }

    void set_service_status(ServiceType service, ServiceStatus status) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        service_metrics_[service].status = status;
    }
};

class RateLimiter {
private:
    std::map<std::string, std::queue<std::chrono::system_clock::time_point>> client_requests_;
    std::mutex limiter_mutex_;
    size_t max_requests_per_minute_;
    size_t max_concurrent_requests_;
    size_t current_concurrent_{0};

public:
    RateLimiter(size_t max_requests_per_minute = 100, size_t max_concurrent = 10)
        : max_requests_per_minute_(max_requests_per_minute),
          max_concurrent_requests_(max_concurrent) {}

    bool allow_request(const std::string& client_id) {
        std::lock_guard<std::mutex> lock(limiter_mutex_);

        // Check concurrent limit
        if (current_concurrent_ >= max_concurrent_requests_) {
            return false;
        }

        auto now = std::chrono::system_clock::now();
        auto& client_queue = client_requests_[client_id];

        // Remove old requests (older than 1 minute)
        while (!client_queue.empty() &&
               std::chrono::duration_cast<std::chrono::minutes>(now - client_queue.front()).count() >= 1) {
            client_queue.pop();
        }

        // Check rate limit
        if (client_queue.size() >= max_requests_per_minute_) {
            return false;
        }

        // Allow request
        client_queue.push(now);
        current_concurrent_++;
        return true;
    }

    void release_request() {
        std::lock_guard<std::mutex> lock(limiter_mutex_);
        current_concurrent_--;
    }

    size_t get_current_load() const {
        return current_concurrent_; // Simple read, may not be perfectly thread-safe but good enough for demo
    }
};

class LoadBalancer {
private:
    std::vector<std::string> server_endpoints_;
    size_t round_robin_counter_{0};
    std::map<std::string, double> server_load_;
    std::mutex load_mutex_;

public:
    LoadBalancer(const std::vector<std::string>& endpoints) : server_endpoints_(endpoints) {
        for (const auto& endpoint : endpoints) {
            server_load_[endpoint] = 0.0;
        }
    }

    std::string get_next_server() {
        if (server_endpoints_.empty()) {
            throw std::runtime_error("No servers available");
        }

        std::lock_guard<std::mutex> lock(load_mutex_);
        // Simple round-robin for now
        size_t index = round_robin_counter_++ % server_endpoints_.size();
        return server_endpoints_[index];
    }

    std::string get_least_loaded_server() {
        std::lock_guard<std::mutex> lock(load_mutex_);

        if (server_endpoints_.empty()) {
            throw std::runtime_error("No servers available");
        }

        auto min_it = std::min_element(server_load_.begin(), server_load_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        return min_it->first;
    }

    void update_server_load(const std::string& server, double load) {
        std::lock_guard<std::mutex> lock(load_mutex_);
        server_load_[server] = load;
    }

    std::vector<std::string> get_healthy_servers() {
        std::lock_guard<std::mutex> lock(load_mutex_);
        std::vector<std::string> healthy;

        for (const auto& pair : server_load_) {
            if (pair.second < 0.8) { // Less than 80% load considered healthy
                healthy.push_back(pair.first);
            }
        }

        return healthy;
    }
};

class RequestQueue {
private:
    std::queue<APIRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool shutdown_{false};
    size_t max_queue_size_;

public:
    RequestQueue(size_t max_size = 1000) : max_queue_size_(max_size) {}

    bool enqueue(const APIRequest& request) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if (request_queue_.size() >= max_queue_size_) {
            return false; // Queue full
        }

        request_queue_.push(request);
        queue_cv_.notify_one();
        return true;
    }

    bool dequeue(APIRequest& request, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if (queue_cv_.wait_for(lock, timeout, [this] { return !request_queue_.empty() || shutdown_; })) {
            if (!request_queue_.empty()) {
                request = request_queue_.front();
                request_queue_.pop();
                return true;
            }
        }

        return false;
    }

    size_t size() const {
        return request_queue_.size(); // Simple read for demo purposes
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_ = true;
        queue_cv_.notify_all();
    }
};

class PhysicsServiceAPI {
private:
    std::unique_ptr<ServiceRegistry> registry_;
    std::unique_ptr<RateLimiter> rate_limiter_;
    std::unique_ptr<LoadBalancer> load_balancer_;
    std::unique_ptr<RequestQueue> request_queue_;

    std::vector<std::thread> worker_threads_;
    bool running_{false};
    std::mutex running_mutex_;

    // Service handlers
    std::map<ServiceType, std::function<APIResponse(const APIRequest&)>> service_handlers_;

    mutable std::mutex api_mutex_;

public:
    PhysicsServiceAPI(const std::vector<std::string>& server_endpoints = {"localhost:8080"},
                     size_t worker_count = std::thread::hardware_concurrency())
        : registry_(std::make_unique<ServiceRegistry>()),
          rate_limiter_(std::make_unique<RateLimiter>(1000, 50)),
          load_balancer_(std::make_unique<LoadBalancer>(server_endpoints)),
          request_queue_(std::make_unique<RequestQueue>(10000)) {

        initialize_service_handlers();
        start_workers(worker_count);
    }

    ~PhysicsServiceAPI() {
        shutdown();
    }

    APIResponse process_request(const APIRequest& request, const std::string& client_id = "default") {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Rate limiting
        if (!rate_limiter_->allow_request(client_id)) {
            APIResponse response;
            response.request_id = request.request_id;
            response.status_code = 429; // Too Many Requests
            response.status_message = "Rate limit exceeded";
            return response;
        }

        // RAII for rate limiter release
        struct RateLimiterGuard {
            RateLimiter* limiter;
            ~RateLimiterGuard() { limiter->release_request(); }
        } guard{rate_limiter_.get()};

        try {
            ServiceType service_type = registry_->get_service_type(request.service_type);

            // Find appropriate handler
            auto handler_it = service_handlers_.find(service_type);
            if (handler_it == service_handlers_.end()) {
                APIResponse response;
                response.request_id = request.request_id;
                response.status_code = 501; // Not Implemented
                response.status_message = "Service not implemented: " + request.service_type;
                return response;
            }

            // Process request
            APIResponse response = handler_it->second(request);
            response.request_id = request.request_id;

            auto end_time = std::chrono::high_resolution_clock::now();
            response.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            // Record metrics
            bool success = (response.status_code >= 200 && response.status_code < 300);
            registry_->record_request(service_type, response.processing_time_ms, success);

            return response;

        } catch (const std::exception& e) {
            APIResponse response;
            response.request_id = request.request_id;
            response.status_code = 500; // Internal Server Error
            response.status_message = std::string("Internal error: ") + e.what();

            auto end_time = std::chrono::high_resolution_clock::now();
            response.processing_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            return response;
        }
    }

    std::future<APIResponse> process_request_async(const APIRequest& request, const std::string& client_id = "default") {
        return std::async(std::launch::async, [this, request, client_id]() {
            return process_request(request, client_id);
        });
    }

    bool enqueue_request(const APIRequest& request) {
        return request_queue_->enqueue(request);
    }

    ServiceMetrics get_service_metrics(const std::string& service_name) {
        ServiceType service_type = registry_->get_service_type(service_name);
        return registry_->get_metrics(service_type);
    }

    std::map<std::string, ServiceMetrics> get_all_metrics() {
        auto metrics_map = registry_->get_all_metrics();
        std::map<std::string, ServiceMetrics> result;

        for (const auto& pair : metrics_map) {
            std::string service_name = get_service_name(pair.first);
            result[service_name] = pair.second;
        }

        return result;
    }

    APIResponse health_check() {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Service healthy";
        response.metadata["service"] = "PhysGrad Physics-as-a-Service";
        response.metadata["version"] = "1.0.0";
        response.metadata["uptime"] = std::to_string(get_uptime_seconds());
        response.metadata["queue_size"] = std::to_string(request_queue_->size());
        response.metadata["load"] = std::to_string(rate_limiter_->get_current_load());

        return response;
    }

    std::string get_openapi_spec() {
        std::stringstream spec;
        spec << R"({
  "openapi": "3.0.0",
  "info": {
    "title": "PhysGrad Physics-as-a-Service API",
    "description": "Cloud-native physics simulation and analysis platform",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "https://api.physgrad.com/v1",
      "description": "Production server"
    }
  ],
  "paths": {
    "/physics/simulate": {
      "post": {
        "summary": "Run physics simulation",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SimulationRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Simulation completed successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/APIResponse"
                }
              }
            }
          }
        }
      }
    },
    "/uncertainty/quantify": {
      "post": {
        "summary": "Perform uncertainty quantification",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/UncertaintyRequest"
              }
            }
          }
        }
      }
    },
    "/symbolic/discover": {
      "post": {
        "summary": "Discover physics models using symbolic regression",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/SymbolicRequest"
              }
            }
          }
        }
      }
    },
    "/health": {
      "get": {
        "summary": "Health check endpoint",
        "responses": {
          "200": {
            "description": "Service is healthy"
          }
        }
      }
    },
    "/metrics": {
      "get": {
        "summary": "Get service metrics",
        "responses": {
          "200": {
            "description": "Service metrics"
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "APIResponse": {
        "type": "object",
        "properties": {
          "request_id": { "type": "string" },
          "status_code": { "type": "integer" },
          "status_message": { "type": "string" },
          "result_data": {
            "type": "array",
            "items": { "type": "number" }
          },
          "processing_time_ms": { "type": "number" }
        }
      }
    }
  }
})";
        return spec.str();
    }

private:
    void initialize_service_handlers() {
        // Physics simulation handler
        service_handlers_[ServiceType::PHYSICS_SIMULATION] = [this](const APIRequest& request) -> APIResponse {
            return handle_physics_simulation(request);
        };

        // Uncertainty quantification handler
        service_handlers_[ServiceType::UNCERTAINTY_QUANTIFICATION] = [this](const APIRequest& request) -> APIResponse {
            return handle_uncertainty_quantification(request);
        };

        // Symbolic regression handler
        service_handlers_[ServiceType::SYMBOLIC_REGRESSION] = [this](const APIRequest& request) -> APIResponse {
            return handle_symbolic_regression(request);
        };

        // Experiment design handler
        service_handlers_[ServiceType::EXPERIMENT_DESIGN] = [this](const APIRequest& request) -> APIResponse {
            return handle_experiment_design(request);
        };

        // Model discovery handler
        service_handlers_[ServiceType::MODEL_DISCOVERY] = [this](const APIRequest& request) -> APIResponse {
            return handle_model_discovery(request);
        };

        // Parameter optimization handler
        service_handlers_[ServiceType::PARAMETER_OPTIMIZATION] = [this](const APIRequest& request) -> APIResponse {
            return handle_parameter_optimization(request);
        };
    }

    APIResponse handle_physics_simulation(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Physics simulation completed";

        // Extract parameters
        auto method_it = request.parameters.find("method");
        std::string method = (method_it != request.parameters.end()) ? method_it->second : "default";

        // Simulate physics computation
        simulate_computation_load(50); // 50ms simulation

        // Generate sample physics results
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<double> dist(0.0, 1.0);

        response.result_data.reserve(100);
        for (int i = 0; i < 100; ++i) {
            response.result_data.push_back(dist(gen));
        }

        response.metadata["method"] = method;
        response.metadata["data_points"] = "100";
        response.metadata["simulation_type"] = "particle_dynamics";

        return response;
    }

    APIResponse handle_uncertainty_quantification(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Uncertainty quantification completed";

        simulate_computation_load(100); // 100ms computation

        // Generate uncertainty metrics
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        response.result_data = {
            dist(gen), // mean
            dist(gen) * 0.1, // std_dev
            dist(gen), // confidence_interval_lower
            dist(gen) // confidence_interval_upper
        };

        response.metadata["analysis_type"] = "monte_carlo";
        response.metadata["samples"] = "10000";
        response.metadata["confidence_level"] = "95";

        return response;
    }

    APIResponse handle_symbolic_regression(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Symbolic regression completed";

        simulate_computation_load(200); // 200ms computation

        // Generate discovered equation metrics
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        response.result_data = {
            dist(gen) * 100, // mse
            dist(gen), // r_squared
            static_cast<double>(gen() % 50 + 5) // complexity
        };

        response.metadata["discovered_equation"] = "2.34 * x^2 + 1.56 * x + 0.78";
        response.metadata["generations"] = "50";
        response.metadata["population_size"] = "100";

        return response;
    }

    APIResponse handle_experiment_design(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Experiment design completed";

        simulate_computation_load(75); // 75ms computation

        // Generate experiment parameters
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> param_dist(-10.0, 10.0);

        response.result_data.reserve(20);
        for (int i = 0; i < 20; ++i) {
            response.result_data.push_back(param_dist(gen));
        }

        response.metadata["design_method"] = "bayesian_optimization";
        response.metadata["parameter_count"] = "5";
        response.metadata["experiment_count"] = "4";

        return response;
    }

    APIResponse handle_model_discovery(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Model discovery completed";

        simulate_computation_load(150); // 150ms computation

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> metric_dist(0.0, 1.0);

        response.result_data = {
            metric_dist(gen), // accuracy
            metric_dist(gen), // precision
            metric_dist(gen), // recall
            metric_dist(gen) * 100 // complexity_score
        };

        response.metadata["discovered_models"] = "3";
        response.metadata["best_model"] = "E = m*c^2";
        response.metadata["discovery_method"] = "genetic_programming";

        return response;
    }

    APIResponse handle_parameter_optimization(const APIRequest& request) {
        APIResponse response;
        response.status_code = 200;
        response.status_message = "Parameter optimization completed";

        simulate_computation_load(125); // 125ms computation

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> param_dist(-5.0, 5.0);

        // Generate optimized parameters
        response.result_data.reserve(10);
        for (int i = 0; i < 10; ++i) {
            response.result_data.push_back(param_dist(gen));
        }

        response.metadata["optimization_method"] = "differential_evolution";
        response.metadata["iterations"] = "100";
        response.metadata["convergence"] = "true";

        return response;
    }

    void simulate_computation_load(int milliseconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    }

    void start_workers(size_t worker_count) {
        std::lock_guard<std::mutex> lock(running_mutex_);
        running_ = true;

        for (size_t i = 0; i < worker_count; ++i) {
            worker_threads_.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    void worker_loop() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(running_mutex_);
                if (!running_) break;
            }
            APIRequest request;
            if (request_queue_->dequeue(request, std::chrono::milliseconds(100))) {
                APIResponse response = process_request(request);
                // In a real implementation, you would send the response back to the client
                // For now, we just process it
            }
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(running_mutex_);
            running_ = false;
        }
        request_queue_->shutdown();

        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::string get_service_name(ServiceType service) {
        switch (service) {
            case ServiceType::PHYSICS_SIMULATION: return "physics_simulation";
            case ServiceType::UNCERTAINTY_QUANTIFICATION: return "uncertainty_quantification";
            case ServiceType::SYMBOLIC_REGRESSION: return "symbolic_regression";
            case ServiceType::EXPERIMENT_DESIGN: return "experiment_design";
            case ServiceType::MODEL_DISCOVERY: return "model_discovery";
            case ServiceType::PARAMETER_OPTIMIZATION: return "parameter_optimization";
            default: return "unknown";
        }
    }

    long get_uptime_seconds() {
        static auto start_time = std::chrono::system_clock::now();
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    }
};

// Utility functions for API integration
class APIClient {
private:
    std::string base_url_;
    std::string api_key_;
    std::map<std::string, std::string> default_headers_;

public:
    APIClient(const std::string& base_url, const std::string& api_key = "")
        : base_url_(base_url), api_key_(api_key) {
        default_headers_["Content-Type"] = "application/json";
        default_headers_["User-Agent"] = "PhysGrad-Client/1.0";
        if (!api_key_.empty()) {
            default_headers_["Authorization"] = "Bearer " + api_key_;
        }
    }

    APIResponse simulate_physics(const std::map<std::string, std::string>& parameters,
                                const std::vector<double>& input_data = {}) {
        APIRequest request;
        request.service_type = "physics_simulation";
        request.method = "POST";
        request.parameters = parameters;
        request.input_data = input_data;

        return send_request(request);
    }

    APIResponse quantify_uncertainty(const std::map<std::string, std::string>& parameters,
                                   const std::vector<double>& input_data = {}) {
        APIRequest request;
        request.service_type = "uncertainty_quantification";
        request.method = "POST";
        request.parameters = parameters;
        request.input_data = input_data;

        return send_request(request);
    }

    APIResponse discover_model(const std::map<std::string, std::string>& parameters,
                              const std::vector<double>& input_data = {}) {
        APIRequest request;
        request.service_type = "symbolic_regression";
        request.method = "POST";
        request.parameters = parameters;
        request.input_data = input_data;

        return send_request(request);
    }

    APIResponse design_experiment(const std::map<std::string, std::string>& parameters,
                                 const std::vector<double>& input_data = {}) {
        APIRequest request;
        request.service_type = "experiment_design";
        request.method = "POST";
        request.parameters = parameters;
        request.input_data = input_data;

        return send_request(request);
    }

private:
    APIResponse send_request(const APIRequest& request) {
        // In a real implementation, this would make an HTTP request
        // For demonstration, we'll simulate a local service call
        static PhysicsServiceAPI local_service;
        return local_service.process_request(request);
    }
};

} // namespace physgrad