#include <gtest/gtest.h>

#include "lagari/core/config.hpp"
#include "lagari/core/logger.hpp"

#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace lagari;

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger for tests
        Logger::init("", LogLevel::OFF, LogLevel::OFF);
        
        // Create temp config file
        temp_dir_ = fs::temp_directory_path() / "lagari_test";
        fs::create_directories(temp_dir_);
        
        config_path_ = temp_dir_ / "test_config.yaml";
        
        std::ofstream f(config_path_);
        f << R"(
system:
  name: "Test System"
  target_fps: 30
  enabled: true
  
capture:
  width: 1280
  height: 720
  fps: 30.5
  
detection:
  confidence_threshold: 0.5
  classes:
    - person
    - car
    - drone
  weights:
    - 1.0
    - 0.8
    - 0.9
)";
    }

    void TearDown() override {
        fs::remove_all(temp_dir_);
    }

    fs::path temp_dir_;
    fs::path config_path_;
};

TEST_F(ConfigTest, LoadValid) {
    Config config;
    EXPECT_TRUE(config.load(config_path_.string()));
    EXPECT_EQ(config.file_path(), config_path_.string());
}

TEST_F(ConfigTest, LoadInvalid) {
    Config config;
    EXPECT_FALSE(config.load("/nonexistent/path/config.yaml"));
}

TEST_F(ConfigTest, GetString) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    EXPECT_EQ(config.get_string("system.name"), "Test System");
    EXPECT_EQ(config.get_string("nonexistent", "default"), "default");
}

TEST_F(ConfigTest, GetInt) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    EXPECT_EQ(config.get_int("system.target_fps"), 30);
    EXPECT_EQ(config.get_int("capture.width"), 1280);
    EXPECT_EQ(config.get_int("nonexistent", 42), 42);
}

TEST_F(ConfigTest, GetFloat) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    EXPECT_FLOAT_EQ(config.get_float("detection.confidence_threshold"), 0.5f);
    EXPECT_FLOAT_EQ(config.get_float("capture.fps"), 30.5f);
    EXPECT_FLOAT_EQ(config.get_float("nonexistent", 1.5f), 1.5f);
}

TEST_F(ConfigTest, GetBool) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    EXPECT_TRUE(config.get_bool("system.enabled"));
    EXPECT_FALSE(config.get_bool("nonexistent", false));
}

TEST_F(ConfigTest, GetStringList) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    auto classes = config.get_string_list("detection.classes");
    ASSERT_EQ(classes.size(), 3);
    EXPECT_EQ(classes[0], "person");
    EXPECT_EQ(classes[1], "car");
    EXPECT_EQ(classes[2], "drone");
}

TEST_F(ConfigTest, GetFloatList) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    auto weights = config.get_float_list("detection.weights");
    ASSERT_EQ(weights.size(), 3);
    EXPECT_FLOAT_EQ(weights[0], 1.0f);
    EXPECT_FLOAT_EQ(weights[1], 0.8f);
    EXPECT_FLOAT_EQ(weights[2], 0.9f);
}

TEST_F(ConfigTest, Has) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    EXPECT_TRUE(config.has("system.name"));
    EXPECT_TRUE(config.has("capture.width"));
    EXPECT_FALSE(config.has("nonexistent.key"));
}

TEST_F(ConfigTest, Override) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    // Original value
    EXPECT_EQ(config.get_int("capture.width"), 1280);
    
    // Override
    config.override("capture.width", "1920");
    EXPECT_EQ(config.get_int("capture.width"), 1920);
}

TEST_F(ConfigTest, ParseArgs) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    const char* argv[] = {
        "program",
        "--capture.width=640",
        "--detection.confidence_threshold", "0.8",
        "--system.enabled"
    };
    int argc = 5;
    
    config.parse_args(argc, const_cast<char**>(argv));
    
    EXPECT_EQ(config.get_int("capture.width"), 640);
    EXPECT_FLOAT_EQ(config.get_float("detection.confidence_threshold"), 0.8f);
    EXPECT_TRUE(config.get_bool("system.enabled"));
}

TEST_F(ConfigTest, ChangeCallback) {
    Config config;
    ASSERT_TRUE(config.load(config_path_.string()));
    
    bool callback_called = false;
    std::string changed_key;
    
    config.on_change("capture.width", [&](const std::string& key) {
        callback_called = true;
        changed_key = key;
    });
    
    config.override("capture.width", "1920");
    
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(changed_key, "capture.width");
}
