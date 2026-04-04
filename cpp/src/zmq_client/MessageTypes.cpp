#include "MessageTypes.h"
#include <cstring>

DataPacket DataPacket::fromJson(const nlohmann::json& header,
                                const void* payload, size_t payloadLen)
{
    DataPacket pkt;
    auto content = header.value("content", nlohmann::json::object());
    pkt.channel     = content.value("channel_num", -1);
    pkt.channelName = content.value("channel_name",
                          std::string("CH") + std::to_string(pkt.channel + 1));
    pkt.sampleRate  = content.value("sample_rate", 0.0);

    // sample_num and num_samples may be int or string
    if (content.contains("sample_num")) {
        auto& v = content["sample_num"];
        if (v.is_number()) pkt.sampleNum = v.get<int>();
        else if (v.is_string()) {
            try { pkt.sampleNum = std::stoi(v.get<std::string>()); } catch (...) {}
        }
    }
    if (content.contains("num_samples")) {
        auto& v = content["num_samples"];
        if (v.is_number()) pkt.numSamples = v.get<int>();
        else if (v.is_string()) {
            try { pkt.numSamples = std::stoi(v.get<std::string>()); } catch (...) {}
        }
    }

    // Binary payload: array of float32
    if (payload && payloadLen > 0) {
        size_t n = payloadLen / sizeof(float);
        pkt.samples.resize(n);
        std::memcpy(pkt.samples.data(), payload, n * sizeof(float));
    }
    return pkt;
}

Event Event::fromJson(const nlohmann::json& header,
                      const void* payload, size_t payloadLen)
{
    Event evt;
    auto content = header.value("content", nlohmann::json::object());
    evt.type       = content.value("type", 0);
    evt.typeName   = eventTypeName(evt.type);
    evt.stream     = content.value("stream", std::string());
    evt.sourceNode = content.value("source_node", 0);
    evt.sampleNum  = content.value("sample_num", 0);

    // Parse binary payload for TTL events:
    //   byte 0: event_line (uint8)
    //   byte 1: event_state (uint8)
    //   bytes 2-9: event_word (uint64)
    if (payload && payloadLen >= 10) {
        auto bytes = static_cast<const uint8_t*>(payload);
        evt.eventLine  = bytes[0];
        evt.eventState = bytes[1];
        std::memcpy(&evt.eventWord, bytes + 2, sizeof(uint64_t));
    }

    // TIMESTAMP type: payload is int64
    if (evt.type == 0 && payload && payloadLen >= 8) {
        std::memcpy(&evt.timestamp, payload, sizeof(int64_t));
    }

    return evt;
}

Spike Spike::fromJson(const nlohmann::json& header,
                      const void* payload, size_t payloadLen)
{
    Spike spk;
    auto spike = header.value("spike", nlohmann::json::object());
    spk.stream      = spike.value("stream", std::string());
    spk.sourceNode  = spike.value("source_node", 0);
    spk.electrode   = spike.value("electrode", 0);
    spk.sampleNum   = spike.value("sample_num", 0);
    spk.numChannels = spike.value("num_channels", 0);
    spk.numSamples  = spike.value("num_samples", 0);
    spk.sortedId    = spike.value("sorted_id", 0);

    if (spike.contains("threshold") && spike["threshold"].is_array()) {
        for (auto& v : spike["threshold"])
            spk.threshold.push_back(v.get<double>());
    }

    // Binary payload: float32 waveform data
    if (payload && payloadLen > 0) {
        size_t n = payloadLen / sizeof(float);
        spk.data.resize(n);
        std::memcpy(spk.data.data(), payload, n * sizeof(float));
    }
    return spk;
}
