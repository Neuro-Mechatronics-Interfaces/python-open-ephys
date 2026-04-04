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
