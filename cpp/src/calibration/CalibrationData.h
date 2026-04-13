#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>

struct CalibrationMovement {
    std::string name;
    std::string instruction;
    std::set<std::string> expectedMuscles;  // muscle group short names
};

struct MuscleGroupMapping {
    std::string groupName;
    std::vector<std::string> blenderNames;  // display names from formatMuscleName
};

struct StripInfo {
    int stripId;
    double centerAngleDeg;
    int channelStart;   // inclusive, 1-based channel number
    int channelEnd;     // exclusive
};

struct MovementObservation {
    int movementIndex;
    std::vector<double> stripRmsValues;  // per-strip average RMS
    std::vector<bool> stripActive;       // per-strip: above threshold?
};

namespace CalibrationData {
    std::vector<CalibrationMovement> getMovements();
    std::vector<MuscleGroupMapping> getMuscleGroupMappings();
    std::vector<StripInfo> getStripDefinitions(
        const std::map<int, double>& stripAnglesDeg);
}
