#pragma once
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <Eigen/Dense>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include "MessageTypes.h"

class ZmqClient {
public:
    struct Config {
        std::string host      = "127.0.0.1";
        std::string dataPort  = "5556";
        std::string hbPort;  // empty → dataPort + 1
        double bufferSeconds  = 30.0;
        int    maxChannels    = 256;
        bool   verbose        = false;
    };

    struct Snapshot {
        Eigen::MatrixXf Y;  // (nSelectedChannels x nSamples)
        Eigen::VectorXd t;  // (nSamples) absolute timestamps in seconds
    };

    explicit ZmqClient(const Config& cfg);
    ~ZmqClient();

    void start();
    void stop();
    bool waitReady(double timeoutSec = 5.0);

    void setChannelIndex(const std::vector<int>& indices);
    std::vector<int> channelIndex() const;
    std::set<int> seenChannels() const;

    Snapshot getLatest(int n) const;

    bool   isReady()    const { return m_ready.load(); }
    double sampleRate() const;

private:
    void run();
    void setupSockets();
    void teardownSockets();
    void sendHeartbeatIfDue();
    void handleData(const nlohmann::json& header, const void* payload, size_t len);
    void handleEvent(const nlohmann::json& header, const void* payload, size_t len);
    void handleSpike(const nlohmann::json& header, const void* payload, size_t len);
    int  targetDequeLen(double fs) const;
    void rebuildDequesIfNeeded(double newFs);
    std::string makeAddr(const std::string& port) const;

    Config m_cfg;
    zmq::context_t m_ctx;
    std::unique_ptr<zmq::socket_t> m_dataSock;
    std::unique_ptr<zmq::socket_t> m_hbSock;

    mutable std::mutex m_mutex;
    std::condition_variable m_readyCv;
    std::atomic<bool> m_ready{false};
    std::atomic<bool> m_stopFlag{false};
    std::thread m_thread;

    // Stream state (protected by m_mutex)
    double m_fs = 2000.0;
    int    m_dequeLen;
    std::vector<std::deque<float>> m_buffers;
    std::set<int>                  m_seenNums;
    std::unordered_map<int, std::string> m_nameByIndex;
    std::vector<int> m_channelIndex;

    int     m_refClockCh = -1;
    int64_t m_totalSamplesWritten = 0;
    int64_t m_globalSampleIndex = 0;
    int64_t m_loopSampleIndex = 0;
    int     m_loopCycle = 0;
    int64_t m_indexOffset = 0;
    int64_t m_lastRefS0 = -1;
    int64_t m_lastRefEnd = -1;

    // Heartbeat
    double m_lastHbSend = 0.0;
    bool   m_waitingHbReply = false;

    // Event/Spike storage
    std::deque<Event> m_events;
    std::deque<Spike> m_spikes;
    static constexpr size_t kMaxEventHistory = 1000;
    static constexpr size_t kMaxSpikeHistory = 1000;
};
