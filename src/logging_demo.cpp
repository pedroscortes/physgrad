#include "logging_system.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>

using namespace physgrad;

class LoggingDemo {
public:
    static void demonstrateLoggingSystem() {
        std::cout << "PhysGrad Logging and Error Handling Demo\n";
        std::cout << "========================================\n\n";

        testBasicLogging();
        testPerformanceLogging();
        testErrorHandling();
        testValidation();
        testFileLogging();
    }

private:
    static void testBasicLogging() {
        std::cout << "1. BASIC LOGGING FUNCTIONALITY:\n";
        std::cout << "-------------------------------\n";

        Logger& logger = Logger::getInstance();

        // Configure logger
        logger.setLogLevel(LogLevel::DEBUG);
        logger.setIncludeTimestamp(true);
        logger.setIncludeFileLocation(true);
        logger.setDestination(LogDestination::CONSOLE);

        std::cout << "Testing different log levels:\n\n";

        // Test different log levels
        LOG_DEBUG("DEMO", "This is a debug message");
        LOG_INFO("DEMO", "This is an info message");
        LOG_WARNING("DEMO", "This is a warning message");
        LOG_ERROR("DEMO", "This is an error message");
        LOG_FATAL("DEMO", "This is a fatal message");

        std::cout << "\nTesting log level filtering (set to WARNING and above):\n\n";

        // Test level filtering
        logger.setLogLevel(LogLevel::WARNING);
        LOG_DEBUG("DEMO", "This debug message should not appear");
        LOG_INFO("DEMO", "This info message should not appear");
        LOG_WARNING("DEMO", "This warning message should appear");
        LOG_ERROR("DEMO", "This error message should appear");

        // Reset for other tests
        logger.setLogLevel(LogLevel::DEBUG);
        std::cout << "\n";
    }

    static void testPerformanceLogging() {
        std::cout << "2. PERFORMANCE LOGGING:\n";
        std::cout << "----------------------\n";

        Logger& logger = Logger::getInstance();

        // Test automatic performance timing
        {
            PERF_TIMER("Matrix multiplication");
            // Simulate some work
            volatile double sum = 0;
            for (int i = 0; i < 1000000; ++i) {
                sum += i * 0.001;
            }
        }

        // Test manual performance logging
        {
            PerformanceTimer timer("Custom operation", "CUSTOM", false);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            timer.log_elapsed("Processing 1000 elements");
        }

        // Test specific performance logging methods
        logger.logMemoryUsage("GPU Memory", 512 * 1024 * 1024, 1024 * 1024 * 1024);
        logger.logOptimizationStep(42, 0.0123f, 0.001f, "Adam");
        logger.logSimulationStep(100, 1.23f, "Stable integration");

        std::cout << "\n";
    }

    static void testErrorHandling() {
        std::cout << "3. ERROR HANDLING SYSTEM:\n";
        std::cout << "------------------------\n";

        // Clear any previous errors
        ErrorHandler::clearErrorHistory();

        // Record different types of errors
        RECORD_ERROR(ErrorHandler::ErrorCode::CUDA_ERROR,
                    "Failed to allocate GPU memory",
                    "Memory allocation in simulation init",
                    "Reduce problem size or use a GPU with more memory");

        RECORD_ERROR(ErrorHandler::ErrorCode::CONVERGENCE_ERROR,
                    "Optimization failed to converge after 1000 iterations",
                    "Adam optimizer with learning rate 0.1",
                    "Try reducing learning rate or changing optimizer");

        RECORD_ERROR(ErrorHandler::ErrorCode::VALIDATION_ERROR,
                    "Invalid time step value",
                    "Time step = -0.001",
                    "Use positive time step values");

        // Display error history
        std::cout << "Error history:\n";
        auto errors = ErrorHandler::getErrorHistory();
        for (size_t i = 0; i < errors.size(); ++i) {
            const auto& error = errors[i];
            std::cout << "  " << (i + 1) << ". "
                     << ErrorHandler::getErrorCodeString(error.code) << ": "
                     << error.message << "\n";
            if (!error.context.empty()) {
                std::cout << "     Context: " << error.context << "\n";
            }
            if (!error.suggestion.empty()) {
                std::cout << "     Suggestion: " << error.suggestion << "\n";
            }
        }

        std::cout << "\nHas errors: " << (ErrorHandler::hasErrors() ? "Yes" : "No") << "\n";

        auto last_error = ErrorHandler::getLastError();
        std::cout << "Last error: " << ErrorHandler::getErrorCodeString(last_error.code)
                 << " - " << last_error.message << "\n\n";
    }

    static void testValidation() {
        std::cout << "4. VALIDATION HELPERS:\n";
        std::cout << "---------------------\n";

        // Clear errors for clean test
        ErrorHandler::clearErrorHistory();

        std::cout << "Testing parameter validation:\n";

        // Valid values
        bool result1 = VALIDATE_FLOAT(0.5f, 0.0f, 1.0f, "learning_rate");
        std::cout << "  Valid learning rate: " << (result1 ? "PASS" : "FAIL") << "\n";

        // Invalid values
        bool result2 = VALIDATE_FLOAT(-0.1f, 0.0f, 1.0f, "learning_rate");
        std::cout << "  Invalid learning rate: " << (result2 ? "FAIL" : "PASS") << "\n";

        bool result3 = ValidationHelper::validateInt(1000, 1, 10000, "num_bodies");
        std::cout << "  Valid num_bodies: " << (result3 ? "PASS" : "FAIL") << "\n";

        bool result4 = ValidationHelper::validateInt(-10, 1, 10000, "num_bodies");
        std::cout << "  Invalid num_bodies: " << (result4 ? "FAIL" : "PASS") << "\n";

        // Pointer validation
        int valid_value = 42;
        bool result5 = VALIDATE_POINTER(&valid_value, "data_ptr");
        std::cout << "  Valid pointer: " << (result5 ? "PASS" : "FAIL") << "\n";

        bool result6 = VALIDATE_POINTER(nullptr, "null_ptr");
        std::cout << "  Null pointer: " << (result6 ? "FAIL" : "PASS") << "\n";

        // Array size validation
        bool result7 = ValidationHelper::validateArraySize(100, 1, 1000, "positions");
        std::cout << "  Valid array size: " << (result7 ? "PASS" : "FAIL") << "\n";

        bool result8 = ValidationHelper::validateArraySize(10000, 1, 1000, "positions");
        std::cout << "  Invalid array size: " << (result8 ? "FAIL" : "PASS") << "\n";

        std::cout << "\nValidation generated " << ErrorHandler::getErrorHistory().size()
                 << " errors as expected.\n\n";
    }

    static void testFileLogging() {
        std::cout << "5. FILE LOGGING:\n";
        std::cout << "---------------\n";

        Logger& logger = Logger::getInstance();
        std::string log_file = "physgrad_test.log";

        // Configure file logging
        logger.setDestination(LogDestination::BOTH);
        logger.setLogFile(log_file);
        logger.setIncludeThreadId(true);

        std::cout << "Writing logs to both console and file: " << log_file << "\n\n";

        // Write various log messages
        LOG_INFO("FILE_TEST", "Starting file logging test");

        // Simulate multithreaded logging
        std::vector<std::thread> threads;
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([i]() {
                for (int j = 0; j < 3; ++j) {
                    std::ostringstream oss;
                    oss << "Thread " << i << " message " << j;
                    LOG_INFO("THREAD", oss.str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            });
        }

        // Wait for threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        LOG_INFO("FILE_TEST", "Completed file logging test");

        logger.flush();

        // Check if file was created and has content
        if (std::filesystem::exists(log_file)) {
            auto file_size = std::filesystem::file_size(log_file);
            std::cout << "✓ Log file created successfully (" << file_size << " bytes)\n";

            // Read and display first few lines
            std::ifstream log_stream(log_file);
            std::string line;
            int line_count = 0;
            std::cout << "\nFirst few lines of log file:\n";
            while (std::getline(log_stream, line) && line_count < 5) {
                std::cout << "  " << line << "\n";
                line_count++;
            }

            // Clean up
            log_stream.close();
            std::filesystem::remove(log_file);
            std::cout << "✓ Log file cleaned up\n";
        } else {
            std::cout << "✗ Log file was not created\n";
        }

        // Reset to console only
        logger.setDestination(LogDestination::CONSOLE);
        logger.setIncludeThreadId(false);

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "LOGGING SYSTEM BENEFITS:\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "✅ Structured logging with multiple levels\n";
        std::cout << "✅ Performance monitoring and timing\n";
        std::cout << "✅ Comprehensive error tracking and history\n";
        std::cout << "✅ Parameter validation with detailed feedback\n";
        std::cout << "✅ Thread-safe file and console logging\n";
        std::cout << "✅ Configurable output formatting\n";
        std::cout << "✅ Easy-to-use macros for development\n";
        std::cout << "\nBest practices:\n";
        std::cout << "- Use appropriate log levels for different information\n";
        std::cout << "- Enable file logging for production systems\n";
        std::cout << "- Use performance timers to identify bottlenecks\n";
        std::cout << "- Validate inputs early with detailed error messages\n";
        std::cout << "- Review error history to identify recurring issues\n";
    }
};

int main() {
    try {
        LoggingDemo::demonstrateLoggingSystem();
    } catch (const std::exception& e) {
        std::cerr << "Demo error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}