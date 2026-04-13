#include "SensorConfig.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>

SensorConfig SensorConfig::load(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open sensor config: " + path);

    nlohmann::json cfg;
    f >> cfg;

    SensorConfig sc;
    for (auto& s : cfg["sensors"]) {
        SensorEntry e;
        e.channel = s["channel"].get<int>();
        auto& pos = s["position"];
        e.position = Eigen::Vector3d(pos[0].get<double>(),
                                     pos[1].get<double>(),
                                     pos[2].get<double>());
        e.label = s.value("label", std::string("CH") + std::to_string(e.channel));
        sc.sensors.push_back(e);
    }

    // Parse forearm axis if present
    if (cfg.contains("forearm_axis")) {
        auto& ax = cfg["forearm_axis"];
        auto& o = ax["origin"];
        auto& d = ax["direction"];
        sc.forearmAxisOrigin = Eigen::Vector3d(o[0].get<double>(), o[1].get<double>(), o[2].get<double>());
        sc.forearmAxisDir    = Eigen::Vector3d(d[0].get<double>(), d[1].get<double>(), d[2].get<double>());
        sc.forearmAxisDir.normalize();
        sc.hasForearmAxis = true;
    }

    // Parse placement block if present
    if (cfg.contains("placement")) {
        auto& pl = cfg["placement"];
        sc.hasPlacement = true;
        if (pl.contains("pitch_mm"))   sc.pitchMm   = pl["pitch_mm"].get<double>();
        if (pl.contains("row_gap_mm")) sc.rowGapMm   = pl["row_gap_mm"].get<double>();
        if (pl.contains("stagger_mm")) sc.staggerMm  = pl["stagger_mm"].get<double>();
        if (pl.contains("strip_angles_deg")) {
            for (auto& [key, val] : pl["strip_angles_deg"].items())
                sc.stripAnglesDeg[std::stoi(key)] = val.get<double>();
        }
        if (pl.contains("sensors_per_row")) {
            for (auto& [key, val] : pl["sensors_per_row"].items())
                sc.sensorsPerRow[std::stoi(key)] = val.get<int>();
        }
    }

    // Sort by channel number
    std::sort(sc.sensors.begin(), sc.sensors.end(),
              [](const SensorEntry& a, const SensorEntry& b) {
                  return a.channel < b.channel;
              });

    int n = static_cast<int>(sc.sensors.size());
    sc.channels.resize(n);
    sc.positions.resize(n, 3);
    for (int i = 0; i < n; ++i) {
        sc.channels[i] = sc.sensors[i].channel;
        sc.positions.row(i) = sc.sensors[i].position.transpose();
    }

    return sc;
}
