#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace lagari {

// ============================================================================
// Implementation details
// ============================================================================

Config::~Config() = default;
Config::Config() = default;

struct Config::Impl {
    YAML::Node root;
    
    YAML::Node navigate(const std::string& key) const {
        std::vector<std::string> parts;
        std::stringstream ss(key);
        std::string part;
        while (std::getline(ss, part, '.')) {
            parts.push_back(part);
        }
        
        // Clone the root to avoid modifying the original tree
        // yaml-cpp's [] operator can modify the tree even on const nodes
        YAML::Node current = YAML::Clone(root);
        
        for (size_t i = 0; i < parts.size(); ++i) {
            const auto& p = parts[i];
            if(!current) {
                return YAML::Node();
            }
            if(!current.IsMap()) {
                return YAML::Node();
            }
            
            // Check if key exists before accessing
            if (!current[p]) {
                return YAML::Node(YAML::NodeType::Undefined);
            }
            current = current[p];
        }

        return current;
    }
};

// ============================================================================
// Config Implementation
// ============================================================================
bool Config::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        impl_ = std::make_unique<Impl>();
        impl_->root = YAML::LoadFile(path);
        file_path_ = path;
        
        LOG_INFO("Loaded configuration from: {}", path);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to load config from {}: {}", path, e.what());
        return false;
    }
}

bool Config::reload() {
    if (file_path_.empty()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        YAML::Node new_root = YAML::LoadFile(file_path_);
        impl_->root = new_root;
        
        // Notify all watchers
        notify_change("*");
        
        LOG_INFO("Reloaded configuration from: {}", file_path_);
        return true;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to reload config: {}", e.what());
        return false;
    }
}

std::string Config::get_string(const std::string& key, 
                                const std::string& default_value) const {
    // Check overrides first
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        return it->second;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<std::string>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

int Config::get_int(const std::string& key, int default_value) const {
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {}
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<int>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

uint32_t Config::get_uint(const std::string& key, uint32_t default_value) const {
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        try {
            return static_cast<uint32_t>(std::stoul(it->second));
        } catch (...) {}
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<uint32_t>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

float Config::get_float(const std::string& key, float default_value) const {
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {}
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<float>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

double Config::get_double(const std::string& key, double default_value) const {
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        try {
            return std::stod(it->second);
        } catch (...) {}
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<double>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

bool Config::get_bool(const std::string& key, bool default_value) const {
    auto it = overrides_.find(key);
    if (it != overrides_.end()) {
        std::string val = it->second;
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        return val == "true" || val == "1" || val == "yes";
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return default_value;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsScalar()) {
            return node.as<bool>();
        }
    } catch (const YAML::Exception&) {}
    
    return default_value;
}

std::vector<std::string> Config::get_string_list(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> result;
    if (!impl_) return result;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsSequence()) {
            for (const auto& item : node) {
                result.push_back(item.as<std::string>());
            }
        }
    } catch (const YAML::Exception&) {}
    
    return result;
}

std::vector<int> Config::get_int_list(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<int> result;
    if (!impl_) return result;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsSequence()) {
            for (const auto& item : node) {
                result.push_back(item.as<int>());
            }
        }
    } catch (const YAML::Exception&) {}
    
    return result;
}

std::vector<float> Config::get_float_list(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<float> result;
    if (!impl_) return result;
    
    try {
        auto node = impl_->navigate(key);
        if (node && node.IsSequence()) {
            for (const auto& item : node) {
                result.push_back(item.as<float>());
            }
        }
    } catch (const YAML::Exception&) {}
    
    return result;
}

bool Config::has(const std::string& key) const {
    if (overrides_.find(key) != overrides_.end()) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!impl_) return false;
    
    auto node = impl_->navigate(key);
    return node && node.IsDefined();
}

template<typename T>
void Config::set(const std::string& key, const T& value) {
    // For now, just store as override
    std::stringstream ss;
    ss << value;
    overrides_[key] = ss.str();
    notify_change(key);
}

// Explicit instantiations
template void Config::set<int>(const std::string&, const int&);
template void Config::set<float>(const std::string&, const float&);
template void Config::set<double>(const std::string&, const double&);
template void Config::set<bool>(const std::string&, const bool&);
template void Config::set<std::string>(const std::string&, const std::string&);

void Config::override(const std::string& key, const std::string& value) {
    overrides_[key] = value;
    notify_change(key);
}

void Config::parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg.substr(0, 2) != "--") {
            continue;
        }
        
        arg = arg.substr(2);
        auto eq_pos = arg.find('=');
        
        std::string key, value;
        
        if (eq_pos != std::string::npos) {
            key = arg.substr(0, eq_pos);
            value = arg.substr(eq_pos + 1);
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
            key = arg;
            value = argv[++i];
        } else {
            // Boolean flag
            key = arg;
            value = "true";
        }
        
        // Convert dashes to dots for nested keys
        std::replace(key.begin(), key.end(), '-', '.');
        
        override(key, value);
        LOG_DEBUG("Config override: {} = {}", key, value);
    }
}

void Config::on_change(const std::string& key, ChangeCallback callback) {
    callbacks_[key].push_back(callback);
}

void Config::notify_change(const std::string& key) {
    // Notify specific key watchers
    auto it = callbacks_.find(key);
    if (it != callbacks_.end()) {
        for (const auto& cb : it->second) {
            cb(key);
        }
    }
    
    // Notify wildcard watchers
    it = callbacks_.find("*");
    if (it != callbacks_.end()) {
        for (const auto& cb : it->second) {
            cb(key);
        }
    }
}

std::shared_ptr<Config> Config::section(const std::string& section) const {
    // TODO: Implement section extraction
    (void)section;
    return nullptr;
}

// ============================================================================
// Global Configuration
// ============================================================================
Config& global_config() {
    static Config instance;
    return instance;
}

}  // namespace lagari
