#include "CalibrationData.h"

std::vector<CalibrationMovement> CalibrationData::getMovements()
{
    return {
        {"Wrist Flexion",
         "Flex your wrist toward your palm and hold",
         {"FCR", "FCU", "PL"}},

        {"Wrist Extension",
         "Extend your wrist backward and hold",
         {"ECRB", "ECRL", "ECU"}},

        {"Radial Deviation",
         "Tilt your wrist toward your thumb and hold",
         {"FCR", "ECRB", "ECRL"}},

        {"Ulnar Deviation",
         "Tilt your wrist toward your pinky and hold",
         {"FCU", "ECU"}},
    };
}

std::vector<MuscleGroupMapping> CalibrationData::getMuscleGroupMappings()
{
    return {
        {"FCR",  {"Flexor Carpi Radialis"}},
        {"FCU",  {"Humeral Head of Flexor Carpi Ulnaris",
                  "Ulnar Head of Flexor Carpi Ulnaris",
                  "Common Tendon of Flexor Carpi Ulnaris"}},
        {"PL",   {"Palmaris Longus Muscle"}},
        {"FDS",  {"Flexor Digitorum Superficialis Humero-ulnar Head",
                  "Flexor Digitorum Superficialis Radial Head"}},
        {"ECRB", {"Extensor Carpi Radialis Brevis"}},
        {"ECRL", {"Extensor Carpi Radialis Longus"}},
        {"ECU",  {"Humeral Head of Extensor Carpi Ulnaris",
                  "Ulnar Head of Extensor Carpi Ulnaris",
                  "Common Tendon of Extensor Carpi Ulnaris"}},
        {"EDC",  {"Extensor Digitorum"}},
        {"EDM",  {"Extensor Digiti Minimi"}},
        {"APL",  {"Abductor Pollicis Longus"}},
        {"EPB",  {"Extensor Pollicis Brevis"}},
    };
}

std::vector<StripInfo> CalibrationData::getStripDefinitions(
    const std::map<int, double>& stripAnglesDeg)
{
    // Channel ranges per strip (1-based, matching sensor_config.json)
    // Strip 1: 26 sensors (13+13), Strip 2: 26, Strip 3: 24 (12+12),
    // Strip 4: 26, Strip 5: 26
    struct StripDef { int id; int chStart; int chEnd; };
    std::vector<StripDef> defs = {
        {1, 1,   27},   // channels 1-26
        {2, 27,  53},   // channels 27-52
        {3, 53,  77},   // channels 53-76
        {4, 77,  103},  // channels 77-102
        {5, 103, 129},  // channels 103-128
    };

    std::vector<StripInfo> result;
    for (const auto& d : defs) {
        StripInfo si;
        si.stripId = d.id;
        si.channelStart = d.chStart;
        si.channelEnd = d.chEnd;

        auto it = stripAnglesDeg.find(d.id);
        if (it != stripAnglesDeg.end()) {
            si.centerAngleDeg = it->second;
        } else {
            // Fallback: 72-degree spacing centered on strip 3 at -172.5
            si.centerAngleDeg = -172.5 + (d.id - 3) * 72.0;
        }
        result.push_back(si);
    }
    return result;
}
