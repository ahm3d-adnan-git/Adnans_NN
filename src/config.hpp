#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <map>
#include <algorithm>

class Config {
public:
    std::string rel_path_train_images;
    std::string rel_path_train_labels;
    std::string rel_path_test_images;
    std::string rel_path_test_labels;
    std::string rel_path_log_file;

    int num_epochs;
    int batch_size;
    int hidden_size;
    double learning_rate;

    Config(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Could not open config file: " << filename << std::endl;
            std::exit(1);
        }

        readConfig(file);
        file.close();
    }
private:
	void readConfig(std::ifstream& file) {
    	std::map<std::string, std::string> config;
    	std::string line;

    	while (std::getline(file, line)) {
        	std::istringstream is_line(line);
        	std::string key;
        	if (std::getline(is_line, key, '=')) {
            	std::string value;
            	if (std::getline(is_line, value)) {
                    key.erase(std::remove_if(key.begin(), key.end(), ::isspace),key.end());
                    value.erase(std::remove_if(value.begin(), value.end(), ::isspace),value.end());
                	config[key] = value;
            	}
        	}
    	}
        
		rel_path_train_images = config["rel_path_train_images"];
        rel_path_train_labels = config["rel_path_train_labels"];
        rel_path_test_images = config["rel_path_test_images"];
        rel_path_test_labels = config["rel_path_test_labels"];
        rel_path_log_file = config["rel_path_log_file"];
        num_epochs = std::stoi(config["num_epochs"]);
        batch_size = std::stoi(config["batch_size"]);
        hidden_size = std::stoi(config["hidden_size"]);
        learning_rate = std::stod(config["learning_rate"]);
	}

};
#endif // READ_CONFIG_HPP