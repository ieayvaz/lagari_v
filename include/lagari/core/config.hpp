#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>
#include <functional>
#include <any>

namespace lagari {

/**
 * @brief Configuration management class
 * 
 * Provides hierarchical configuration access with:
 * - YAML file loading
 * - Type-safe value retrieval with defaults
 * - Hot-reload capability for runtime parameters
 * - Command-line override support
 */
class Config {
public:
    Config() = default;
    ~Config() = default;

    // Non-copyable, moveable
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = default;
    Config& operator=(Config&&) = default;

    /**
     * @brief Load configuration from YAML file
     * 
     * @param path Path to YAML file
     * @return true if loaded successfully
     */
    bool load(const std::string& path);

    /**
     * @brief Reload configuration from previously loaded file
     * 
     * Only reloads values marked as hot-reloadable.
     * 
     * @return true if reloaded successfully
     */
    bool reload();

    /**
     * @brief Get string value
     * 
     * @param key Dot-separated key path (e.g., "camera.width")
     * @param default_value Value to return if key not found
     * @return Configuration value or default
     */
    std::string get_string(const std::string& key, 
                           const std::string& default_value = "") const;

    /**
     * @brief Get integer value
     */
    int get_int(const std::string& key, int default_value = 0) const;

    /**
     * @brief Get unsigned integer value
     */
    uint32_t get_uint(const std::string& key, uint32_t default_value = 0) const;

    /**
     * @brief Get floating-point value
     */
    float get_float(const std::string& key, float default_value = 0.0f) const;

    /**
     * @brief Get double value
     */
    double get_double(const std::string& key, double default_value = 0.0) const;

    /**
     * @brief Get boolean value
     */
    bool get_bool(const std::string& key, bool default_value = false) const;

    /**
     * @brief Get vector of strings
     */
    std::vector<std::string> get_string_list(const std::string& key) const;

    /**
     * @brief Get vector of integers
     */
    std::vector<int> get_int_list(const std::string& key) const;

    /**
     * @brief Get vector of floats
     */
    std::vector<float> get_float_list(const std::string& key) const;

    /**
     * @brief Check if key exists
     */
    bool has(const std::string& key) const;

    /**
     * @brief Set value (for runtime updates or command-line overrides)
     * 
     * @tparam T Value type
     * @param key Dot-separated key path
     * @param value Value to set
     */
    template<typename T>
    void set(const std::string& key, const T& value);

    /**
     * @brief Override value from command line
     * 
     * Command-line overrides take precedence over file values.
     * 
     * @param key Key path
     * @param value Value string (will be parsed)
     */
    void override(const std::string& key, const std::string& value);

    /**
     * @brief Parse command-line arguments
     * 
     * Supports --key=value and --key value formats.
     * 
     * @param argc Argument count
     * @param argv Argument values
     */
    void parse_args(int argc, char* argv[]);

    /**
     * @brief Register callback for configuration changes
     * 
     * @param key Key to watch (or "*" for all changes)
     * @param callback Function to call on change
     */
    using ChangeCallback = std::function<void(const std::string& key)>;
    void on_change(const std::string& key, ChangeCallback callback);

    /**
     * @brief Get the configuration file path
     */
    const std::string& file_path() const { return file_path_; }

    /**
     * @brief Get sub-configuration for a section
     * 
     * @param section Section name
     * @return Pointer to section config, or nullptr if not found
     */
    std::shared_ptr<Config> section(const std::string& section) const;

private:
    std::string file_path_;
    
    // Internal storage
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    mutable std::mutex mutex_;
    
    // Change callbacks
    std::unordered_map<std::string, std::vector<ChangeCallback>> callbacks_;
    
    // Command-line overrides (take precedence)
    std::unordered_map<std::string, std::string> overrides_;

    void notify_change(const std::string& key);
};

/**
 * @brief Global configuration instance
 */
Config& global_config();

}  // namespace lagari
