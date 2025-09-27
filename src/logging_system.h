#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>

namespace physgrad {

// Log levels
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

// Log destinations
enum class LogDestination {
    CONSOLE,
    FILE,
    BOTH
};

// Log entry structure
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string category;
    std::string message;
    std::string file;
    int line;
    std::thread::id thread_id;
};

// Error handling utilities
class ErrorHandler {
public:
    enum class ErrorCode {
        SUCCESS = 0,
        CUDA_ERROR,
        MEMORY_ERROR,
        FILE_ERROR,
        VALIDATION_ERROR,
        CONFIGURATION_ERROR,
        OPTIMIZATION_ERROR,
        INTEGRATION_ERROR,
        CONVERGENCE_ERROR,
        UNKNOWN_ERROR
    };

    struct ErrorInfo {
        ErrorCode code;
        std::string message;
        std::string context;
        std::string suggestion;
        std::chrono::system_clock::time_point timestamp;
    };

private:
    static std::vector<ErrorInfo> error_history;
    static std::mutex error_mutex;

public:
    static void recordError(ErrorCode code, const std::string& message,
                          const std::string& context = "",
                          const std::string& suggestion = "");
    static std::vector<ErrorInfo> getErrorHistory();
    static void clearErrorHistory();
    static std::string getErrorCodeString(ErrorCode code);
    static bool hasErrors();
    static ErrorInfo getLastError();
};

// Main logging system
class Logger {
private:
    static std::unique_ptr<Logger> instance;
    static std::mutex logger_mutex;

    LogLevel min_level;
    LogDestination destination;
    std::string log_file_path;
    std::ofstream log_file;
    bool enabled;
    bool include_timestamp;
    bool include_thread_id;
    bool include_file_location;
    std::mutex file_mutex;

    Logger();

public:
    static Logger& getInstance();

    // Configuration
    void setLogLevel(LogLevel level);
    void setDestination(LogDestination dest);
    void setLogFile(const std::string& filepath);
    void setEnabled(bool enable);
    void setIncludeTimestamp(bool include);
    void setIncludeThreadId(bool include);
    void setIncludeFileLocation(bool include);

    // Logging methods
    void log(LogLevel level, const std::string& category, const std::string& message,
             const std::string& file = "", int line = 0);

    void debug(const std::string& category, const std::string& message,
               const std::string& file = "", int line = 0);
    void info(const std::string& category, const std::string& message,
              const std::string& file = "", int line = 0);
    void warning(const std::string& category, const std::string& message,
                 const std::string& file = "", int line = 0);
    void error(const std::string& category, const std::string& message,
               const std::string& file = "", int line = 0);
    void fatal(const std::string& category, const std::string& message,
               const std::string& file = "", int line = 0);

    // Performance logging
    void logPerformance(const std::string& operation, double duration_ms,
                       const std::string& details = "");

    // Memory logging
    void logMemoryUsage(const std::string& component, size_t bytes_used,
                       size_t bytes_total = 0);

    // Optimization logging
    void logOptimizationStep(int iteration, float loss, float learning_rate,
                           const std::string& optimizer = "");

    // Simulation logging
    void logSimulationStep(int step, float time, const std::string& details = "");

    // Utility methods
    void flush();
    std::string getLogLevelString(LogLevel level) const;

private:
    void writeLog(const LogEntry& entry);
    std::string formatLogEntry(const LogEntry& entry) const;
    bool shouldLog(LogLevel level) const;
};

// Performance timer utility
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string operation_name;
    std::string category;
    bool auto_log;

public:
    PerformanceTimer(const std::string& operation, const std::string& cat = "PERF",
                    bool auto_log_on_destroy = true);
    ~PerformanceTimer();

    void reset();
    double elapsed_ms() const;
    void log_elapsed(const std::string& details = "");
};

// Validation helpers
class ValidationHelper {
public:
    static bool validateFloat(float value, float min_val, float max_val,
                            const std::string& param_name);
    static bool validateInt(int value, int min_val, int max_val,
                          const std::string& param_name);
    static bool validatePointer(const void* ptr, const std::string& ptr_name);
    static bool validateArraySize(size_t size, size_t min_size, size_t max_size,
                                const std::string& array_name);
    static bool validateCudaError(int cuda_error, const std::string& operation);
};

// Macros for convenient logging
#define LOG_DEBUG(category, message) \
    physgrad::Logger::getInstance().debug(category, message, __FILE__, __LINE__)

#define LOG_INFO(category, message) \
    physgrad::Logger::getInstance().info(category, message, __FILE__, __LINE__)

#define LOG_WARNING(category, message) \
    physgrad::Logger::getInstance().warning(category, message, __FILE__, __LINE__)

#define LOG_ERROR(category, message) \
    physgrad::Logger::getInstance().error(category, message, __FILE__, __LINE__)

#define LOG_FATAL(category, message) \
    physgrad::Logger::getInstance().fatal(category, message, __FILE__, __LINE__)

#define RECORD_ERROR(code, message, context, suggestion) \
    physgrad::ErrorHandler::recordError(code, message, context, suggestion)

#define VALIDATE_FLOAT(value, min_val, max_val, name) \
    physgrad::ValidationHelper::validateFloat(value, min_val, max_val, name)

#define VALIDATE_POINTER(ptr, name) \
    physgrad::ValidationHelper::validatePointer(ptr, name)

#define PERF_TIMER(operation) \
    physgrad::PerformanceTimer __timer(operation)

#define PERF_TIMER_CATEGORY(operation, category) \
    physgrad::PerformanceTimer __timer(operation, category)

// Implementation
inline Logger::Logger()
    : min_level(LogLevel::INFO), destination(LogDestination::CONSOLE),
      enabled(true), include_timestamp(true), include_thread_id(false),
      include_file_location(false) {}

inline Logger& Logger::getInstance() {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (!instance) {
        instance = std::unique_ptr<Logger>(new Logger());
    }
    return *instance;
}

inline void Logger::setLogLevel(LogLevel level) {
    min_level = level;
}

inline void Logger::setDestination(LogDestination dest) {
    destination = dest;
}

inline void Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(file_mutex);
    if (log_file.is_open()) {
        log_file.close();
    }
    log_file_path = filepath;
    log_file.open(filepath, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << filepath << std::endl;
    }
}

inline void Logger::setEnabled(bool enable) {
    enabled = enable;
}

inline void Logger::setIncludeTimestamp(bool include) {
    include_timestamp = include;
}

inline void Logger::setIncludeThreadId(bool include) {
    include_thread_id = include;
}

inline void Logger::setIncludeFileLocation(bool include) {
    include_file_location = include;
}

inline void Logger::log(LogLevel level, const std::string& category, const std::string& message,
                       const std::string& file, int line) {
    if (!enabled || !shouldLog(level)) return;

    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.category = category;
    entry.message = message;
    entry.file = file;
    entry.line = line;
    entry.thread_id = std::this_thread::get_id();

    writeLog(entry);
}

inline void Logger::debug(const std::string& category, const std::string& message,
                         const std::string& file, int line) {
    log(LogLevel::DEBUG, category, message, file, line);
}

inline void Logger::info(const std::string& category, const std::string& message,
                        const std::string& file, int line) {
    log(LogLevel::INFO, category, message, file, line);
}

inline void Logger::warning(const std::string& category, const std::string& message,
                           const std::string& file, int line) {
    log(LogLevel::WARNING, category, message, file, line);
}

inline void Logger::error(const std::string& category, const std::string& message,
                         const std::string& file, int line) {
    log(LogLevel::ERROR, category, message, file, line);
}

inline void Logger::fatal(const std::string& category, const std::string& message,
                         const std::string& file, int line) {
    log(LogLevel::FATAL, category, message, file, line);
}

inline void Logger::logPerformance(const std::string& operation, double duration_ms,
                                  const std::string& details) {
    std::ostringstream oss;
    oss << operation << " completed in " << duration_ms << "ms";
    if (!details.empty()) {
        oss << " (" << details << ")";
    }
    info("PERFORMANCE", oss.str());
}

inline void Logger::logMemoryUsage(const std::string& component, size_t bytes_used,
                                  size_t bytes_total) {
    std::ostringstream oss;
    oss << component << " using " << bytes_used / (1024 * 1024) << " MB";
    if (bytes_total > 0) {
        oss << " / " << bytes_total / (1024 * 1024) << " MB ("
            << std::fixed << std::setprecision(1)
            << (100.0 * bytes_used / bytes_total) << "%)";
    }
    info("MEMORY", oss.str());
}

inline void Logger::logOptimizationStep(int iteration, float loss, float learning_rate,
                                       const std::string& optimizer) {
    std::ostringstream oss;
    oss << "Iteration " << iteration << ": loss=" << loss << ", lr=" << learning_rate;
    if (!optimizer.empty()) {
        oss << ", optimizer=" << optimizer;
    }
    info("OPTIMIZATION", oss.str());
}

inline void Logger::logSimulationStep(int step, float time, const std::string& details) {
    std::ostringstream oss;
    oss << "Step " << step << " at time " << time;
    if (!details.empty()) {
        oss << " (" << details << ")";
    }
    debug("SIMULATION", oss.str());
}

inline void Logger::flush() {
    std::lock_guard<std::mutex> lock(file_mutex);
    if (log_file.is_open()) {
        log_file.flush();
    }
    std::cout.flush();
    std::cerr.flush();
}

inline std::string Logger::getLogLevelString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

inline void Logger::writeLog(const LogEntry& entry) {
    std::string formatted = formatLogEntry(entry);

    if (destination == LogDestination::CONSOLE || destination == LogDestination::BOTH) {
        if (entry.level >= LogLevel::ERROR) {
            std::cerr << formatted << std::endl;
        } else {
            std::cout << formatted << std::endl;
        }
    }

    if (destination == LogDestination::FILE || destination == LogDestination::BOTH) {
        std::lock_guard<std::mutex> lock(file_mutex);
        if (log_file.is_open()) {
            log_file << formatted << std::endl;
            log_file.flush();
        }
    }
}

inline std::string Logger::formatLogEntry(const LogEntry& entry) const {
    std::ostringstream oss;

    if (include_timestamp) {
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;

        oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    }

    oss << "[" << getLogLevelString(entry.level) << "] ";
    oss << "[" << entry.category << "] ";

    if (include_thread_id) {
        oss << "[T:" << entry.thread_id << "] ";
    }

    oss << entry.message;

    if (include_file_location && !entry.file.empty() && entry.line > 0) {
        // Extract just the filename from the full path
        size_t last_slash = entry.file.find_last_of("/\\");
        std::string filename = (last_slash == std::string::npos) ?
            entry.file : entry.file.substr(last_slash + 1);
        oss << " (" << filename << ":" << entry.line << ")";
    }

    return oss.str();
}

inline bool Logger::shouldLog(LogLevel level) const {
    return level >= min_level;
}

// Static member definitions
inline std::unique_ptr<Logger> Logger::instance = nullptr;
inline std::mutex Logger::logger_mutex;

// PerformanceTimer implementation
inline PerformanceTimer::PerformanceTimer(const std::string& operation, const std::string& cat,
                                         bool auto_log_on_destroy)
    : operation_name(operation), category(cat), auto_log(auto_log_on_destroy) {
    start_time = std::chrono::high_resolution_clock::now();
}

inline PerformanceTimer::~PerformanceTimer() {
    if (auto_log) {
        log_elapsed();
    }
}

inline void PerformanceTimer::reset() {
    start_time = std::chrono::high_resolution_clock::now();
}

inline double PerformanceTimer::elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0;
}

inline void PerformanceTimer::log_elapsed(const std::string& details) {
    Logger::getInstance().logPerformance(operation_name, elapsed_ms(), details);
}

// ErrorHandler implementation
inline std::vector<ErrorHandler::ErrorInfo> ErrorHandler::error_history;
inline std::mutex ErrorHandler::error_mutex;

inline void ErrorHandler::recordError(ErrorCode code, const std::string& message,
                                     const std::string& context, const std::string& suggestion) {
    std::lock_guard<std::mutex> lock(error_mutex);

    ErrorInfo info;
    info.code = code;
    info.message = message;
    info.context = context;
    info.suggestion = suggestion;
    info.timestamp = std::chrono::system_clock::now();

    error_history.push_back(info);

    // Also log the error
    std::ostringstream oss;
    oss << getErrorCodeString(code) << ": " << message;
    if (!context.empty()) {
        oss << " (Context: " << context << ")";
    }
    if (!suggestion.empty()) {
        oss << " Suggestion: " << suggestion;
    }

    Logger::getInstance().error("ERROR_HANDLER", oss.str());
}

inline std::vector<ErrorHandler::ErrorInfo> ErrorHandler::getErrorHistory() {
    std::lock_guard<std::mutex> lock(error_mutex);
    return error_history;
}

inline void ErrorHandler::clearErrorHistory() {
    std::lock_guard<std::mutex> lock(error_mutex);
    error_history.clear();
}

inline std::string ErrorHandler::getErrorCodeString(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "SUCCESS";
        case ErrorCode::CUDA_ERROR: return "CUDA_ERROR";
        case ErrorCode::MEMORY_ERROR: return "MEMORY_ERROR";
        case ErrorCode::FILE_ERROR: return "FILE_ERROR";
        case ErrorCode::VALIDATION_ERROR: return "VALIDATION_ERROR";
        case ErrorCode::CONFIGURATION_ERROR: return "CONFIGURATION_ERROR";
        case ErrorCode::OPTIMIZATION_ERROR: return "OPTIMIZATION_ERROR";
        case ErrorCode::INTEGRATION_ERROR: return "INTEGRATION_ERROR";
        case ErrorCode::CONVERGENCE_ERROR: return "CONVERGENCE_ERROR";
        case ErrorCode::UNKNOWN_ERROR: return "UNKNOWN_ERROR";
        default: return "UNDEFINED_ERROR";
    }
}

inline bool ErrorHandler::hasErrors() {
    std::lock_guard<std::mutex> lock(error_mutex);
    return !error_history.empty();
}

inline ErrorHandler::ErrorInfo ErrorHandler::getLastError() {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (error_history.empty()) {
        return {ErrorCode::SUCCESS, "", "", "", std::chrono::system_clock::now()};
    }
    return error_history.back();
}

// ValidationHelper implementation
inline bool ValidationHelper::validateFloat(float value, float min_val, float max_val,
                                          const std::string& param_name) {
    if (value < min_val || value > max_val) {
        std::ostringstream oss;
        oss << "Parameter " << param_name << " = " << value
            << " is out of range [" << min_val << ", " << max_val << "]";
        RECORD_ERROR(ErrorHandler::ErrorCode::VALIDATION_ERROR, oss.str(),
                    "Float validation", "Check parameter bounds");
        return false;
    }
    return true;
}

inline bool ValidationHelper::validateInt(int value, int min_val, int max_val,
                                        const std::string& param_name) {
    if (value < min_val || value > max_val) {
        std::ostringstream oss;
        oss << "Parameter " << param_name << " = " << value
            << " is out of range [" << min_val << ", " << max_val << "]";
        RECORD_ERROR(ErrorHandler::ErrorCode::VALIDATION_ERROR, oss.str(),
                    "Integer validation", "Check parameter bounds");
        return false;
    }
    return true;
}

inline bool ValidationHelper::validatePointer(const void* ptr, const std::string& ptr_name) {
    if (ptr == nullptr) {
        std::ostringstream oss;
        oss << "Null pointer: " << ptr_name;
        RECORD_ERROR(ErrorHandler::ErrorCode::VALIDATION_ERROR, oss.str(),
                    "Pointer validation", "Check pointer initialization");
        return false;
    }
    return true;
}

inline bool ValidationHelper::validateArraySize(size_t size, size_t min_size, size_t max_size,
                                              const std::string& array_name) {
    if (size < min_size || size > max_size) {
        std::ostringstream oss;
        oss << "Array " << array_name << " size " << size
            << " is out of range [" << min_size << ", " << max_size << "]";
        RECORD_ERROR(ErrorHandler::ErrorCode::VALIDATION_ERROR, oss.str(),
                    "Array size validation", "Check array dimensions");
        return false;
    }
    return true;
}

inline bool ValidationHelper::validateCudaError(int cuda_error, const std::string& operation) {
    if (cuda_error != 0) {  // cudaSuccess = 0
        std::ostringstream oss;
        oss << "CUDA error " << cuda_error << " in operation: " << operation;
        RECORD_ERROR(ErrorHandler::ErrorCode::CUDA_ERROR, oss.str(),
                    "CUDA operation", "Check CUDA installation and GPU status");
        return false;
    }
    return true;
}

} // namespace physgrad