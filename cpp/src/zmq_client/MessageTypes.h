#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

// Maps Open Ephys event type integers to string names
inline const char* eventTypeName(int type) {
    switch (type) {
        case 0: return "TIMESTAMP";
        case 1: return "BUFFER_SIZE";
        case 2: return "PARAMETER_CHANGE";
        case 3: return "TTL";
        case 4: return "SPIKE";
        case 5: return "MESSAGE";
        case 6: return "BINARY_MSG";
        default: return "UNKNOWN";
    }
}

struct DataPacket {
    int channel = -1;
    std::string channelName;
    double sampleRate = 0.0;
    int sampleNum = -1;   // first sample index in packet (-1 = unknown)
    int numSamples = -1;  // declared sample count (-1 = unknown)
    std::vector<float> samples;

    static DataPacket fromJson(const nlohmann::json& header,
                               const void* payload, size_t payloadLen);
};

struct Event {
    int type = 0;
    std::string typeName;
    std::string stream;
    int sourceNode = 0;
    int sampleNum = 0;
    int eventLine = 0;
    int eventState = 0;
    uint64_t eventWord = 0;
    int64_t timestamp = 0; // only valid for TIMESTAMP type

    static Event fromJson(const nlohmann::json& header,
                          const void* payload, size_t payloadLen);
};

struct Spike {
    std::string stream;
    int sourceNode = 0;
    int electrode = 0;
    int sampleNum = 0;
    int numChannels = 0;
    int numSamples = 0;
    int sortedId = 0;
    std::vector<double> threshold;
    std::vector<float> data;

    static Spike fromJson(const nlohmann::json& header,
                          const void* payload, size_t payloadLen);
};
