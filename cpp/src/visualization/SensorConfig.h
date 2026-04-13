#pragma once
#include <string>
#include <vector>
#include <map>
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

    // Forearm axis (from fascia config, or default)
    bool hasForearmAxis = false;
    Eigen::Vector3d forearmAxisOrigin = Eigen::Vector3d::Zero();
    Eigen::Vector3d forearmAxisDir    = Eigen::Vector3d(0, 0, 1);

    // Placement parameters (from sensor_config.json "placement" block)
    bool hasPlacement = false;
    double pitchMm    = 19.5;
    double rowGapMm   = 10.0;
    double staggerMm  = 9.75;
    std::map<int, double> stripAnglesDeg;   // strip_id → center angle (degrees)
    std::map<int, int>    sensorsPerRow;    // strip_id → sensors per row

    static SensorConfig load(const std::string& path);
};
