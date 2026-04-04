#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

struct SensorEntry {
    int channel;
    Eigen::Vector3d position;
    std::string label;
};

struct SensorConfig {
    std::vector<SensorEntry> sensors;
    std::vector<int> channels;         // sorted channel numbers
    Eigen::MatrixXd positions;         // (nSensors x 3) in channel order

    static SensorConfig load(const std::string& path);
};
