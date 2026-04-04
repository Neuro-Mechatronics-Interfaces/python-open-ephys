#include "ZmqClient.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>

static double nowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ─── Construction / destruction ───

ZmqClient::ZmqClient(const Config& cfg)
    : m_cfg(cfg), m_ctx(1)
{
    if (m_cfg.hbPort.empty())
        m_cfg.hbPort = std::to_string(std::stoi(m_cfg.dataPort) + 1);

    m_dequeLen = targetDequeLen(m_fs);
    m_buffers.resize(m_cfg.maxChannels);
}

ZmqClient::~ZmqClient() { stop(); }

int ZmqClient::targetDequeLen(double fs) const {
    return std::max(1, static_cast<int>(std::round(fs * m_cfg.bufferSeconds)));
}

std::string ZmqClient::makeAddr(const std::string& port) const {
    return "tcp://" + m_cfg.host + ":" + port;
}

// ─── Socket management ───

void ZmqClient::setupSockets() {
    teardownSockets();

    m_dataSock = std::make_unique<zmq::socket_t>(m_ctx, zmq::socket_type::sub);
    m_dataSock->connect(makeAddr(m_cfg.dataPort));
    m_dataSock->set(zmq::sockopt::subscribe, "");
    m_dataSock->set(zmq::sockopt::rcvtimeo, 1000);

    m_hbSock = std::make_unique<zmq::socket_t>(m_ctx, zmq::socket_type::req);
    m_hbSock->connect(makeAddr(m_cfg.hbPort));
    m_hbSock->set(zmq::sockopt::rcvtimeo, 2000);

    m_lastHbSend = 0.0;
    m_waitingHbReply = false;
}

void ZmqClient::teardownSockets() {
    if (m_dataSock) { m_dataSock->close(); m_dataSock.reset(); }
    if (m_hbSock)   { m_hbSock->close();   m_hbSock.reset(); }
}

// ─── Start / stop ───

void ZmqClient::start() {
    if (m_thread.joinable()) return;
    setupSockets();
    m_stopFlag = false;
    m_thread = std::thread(&ZmqClient::run, this);
    if (m_cfg.verbose)
        std::cout << "[ZmqClient] started; data=" << makeAddr(m_cfg.dataPort)
                  << " hb=" << makeAddr(m_cfg.hbPort) << "\n";
}

void ZmqClient::stop() {
    m_stopFlag = true;
    if (m_thread.joinable()) m_thread.join();
    teardownSockets();
    if (m_cfg.verbose) std::cout << "[ZmqClient] stopped\n";
}

bool ZmqClient::waitReady(double timeoutSec) {
    std::unique_lock<std::mutex> lk(m_mutex);
    if (m_ready) return true;
    return m_readyCv.wait_for(lk,
        std::chrono::milliseconds(static_cast<int>(timeoutSec * 1000)),
        [this]{ return m_ready.load(); });
}

// ─── Channel selection & info ───

void ZmqClient::setChannelIndex(const std::vector<int>& indices) {
    std::lock_guard<std::mutex> lk(m_mutex);
    m_channelIndex = indices;
}

std::vector<int> ZmqClient::channelIndex() const {
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_channelIndex;
}

std::set<int> ZmqClient::seenChannels() const {
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_seenNums;
}

double ZmqClient::sampleRate() const {
    std::lock_guard<std::mutex> lk(m_mutex);
    return m_fs;
}

// ─── getLatest ───

ZmqClient::Snapshot ZmqClient::getLatest(int n) const {
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_ready)
        throw std::runtime_error("ZmqClient not ready");

    auto idx = m_channelIndex;
    if (idx.empty()) {
        idx.assign(m_seenNums.begin(), m_seenNums.end());
    }

    n = std::max(1, n);
    int nCh = static_cast<int>(idx.size());
    Snapshot snap;
    snap.Y = Eigen::MatrixXf::Zero(nCh, n);

    for (int i = 0; i < nCh; ++i) {
        int ch = idx[i];
        if (ch < 0 || ch >= m_cfg.maxChannels) continue;
        const auto& buf = m_buffers[ch];
        int have = static_cast<int>(buf.size());
        int m = std::min(have, n);
        if (m <= 0) continue;
        // Copy from tail of deque
        auto it = buf.end() - m;
        for (int j = n - m; j < n; ++j, ++it)
            snap.Y(i, j) = *it;
    }

    // Timestamps from global index
    int64_t total = m_globalSampleIndex;
    snap.t.resize(n);
    for (int j = 0; j < n; ++j)
        snap.t(j) = static_cast<double>(total - n + j) / m_fs;

    return snap;
}

// ─── Heartbeat ───

void ZmqClient::sendHeartbeatIfDue() {
    if (!m_hbSock || m_waitingHbReply) return;
    double now = nowSec();
    if (now - m_lastHbSend < 2.0) return;

    try {
        std::string msg = R"({"application":"NewZMQClient","type":"heartbeat"})";
        m_hbSock->send(zmq::buffer(msg), zmq::send_flags::none);
        m_lastHbSend = now;
        m_waitingHbReply = true;
    } catch (const zmq::error_t& e) {
        if (m_cfg.verbose)
            std::cerr << "[HB] send error: " << e.what() << "\n";
    }
}

// ─── Message handlers ───

void ZmqClient::handleData(const nlohmann::json& header,
                            const void* payload, size_t len)
{
    DataPacket pkt = DataPacket::fromJson(header, payload, len);
    int ch = pkt.channel;
    if (ch < 0 || ch >= m_cfg.maxChannels || pkt.samples.empty())
        return;

    std::lock_guard<std::mutex> lk(m_mutex);

    // Update sample rate if changed
    if (pkt.sampleRate > 0.0 && pkt.sampleRate != m_fs) {
        m_fs = pkt.sampleRate;
        rebuildDequesIfNeeded(m_fs);
    }

    // Append samples to channel buffer
    auto& buf = m_buffers[ch];
    for (float s : pkt.samples) {
        if (static_cast<int>(buf.size()) >= m_dequeLen)
            buf.pop_front();
        buf.push_back(s);
    }

    m_nameByIndex[ch] = pkt.channelName;
    m_seenNums.insert(ch);

    // Reference clock channel (first channel seen)
    if (m_refClockCh < 0) m_refClockCh = ch;
    if (ch == m_refClockCh)
        m_totalSamplesWritten += static_cast<int64_t>(pkt.samples.size());

    // Sample index tracking with loop detection
    int64_t endIdx = -1;
    if (pkt.sampleNum >= 0 && pkt.numSamples >= 0)
        endIdx = static_cast<int64_t>(pkt.sampleNum) + pkt.numSamples;

    if (ch == m_refClockCh && endIdx >= 0) {
        if (m_lastRefS0 >= 0 && pkt.sampleNum < static_cast<int>(m_lastRefS0)) {
            // Playback loop detected
            m_loopCycle++;
            if (m_lastRefEnd >= 0)
                m_indexOffset += m_lastRefEnd;
        }
        m_lastRefS0 = pkt.sampleNum;
        m_lastRefEnd = endIdx;
    }

    if (endIdx >= 0) {
        m_loopSampleIndex = endIdx;
        int64_t mono = m_indexOffset + endIdx;
        if (mono > m_globalSampleIndex)
            m_globalSampleIndex = mono;
    } else {
        m_globalSampleIndex += static_cast<int64_t>(pkt.samples.size());
    }

    // Signal ready
    if (!m_ready) {
        m_ready = true;
        m_readyCv.notify_all();
    }
}

void ZmqClient::handleEvent(const nlohmann::json& header,
                             const void* payload, size_t len)
{
    Event evt = Event::fromJson(header, payload, len);
    std::lock_guard<std::mutex> lk(m_mutex);
    m_events.push_back(std::move(evt));
    while (m_events.size() > kMaxEventHistory) m_events.pop_front();
    if (m_cfg.verbose)
        std::cout << "[Event] type=" << evt.typeName
                  << " line=" << evt.eventLine
                  << " state=" << evt.eventState << "\n";
}

void ZmqClient::handleSpike(const nlohmann::json& header,
                             const void* payload, size_t len)
{
    Spike spk = Spike::fromJson(header, payload, len);
    std::lock_guard<std::mutex> lk(m_mutex);
    m_spikes.push_back(std::move(spk));
    while (m_spikes.size() > kMaxSpikeHistory) m_spikes.pop_front();
    if (m_cfg.verbose)
        std::cout << "[Spike] electrode=" << spk.electrode
                  << " sorted_id=" << spk.sortedId << "\n";
}

void ZmqClient::rebuildDequesIfNeeded(double newFs) {
    int newLen = targetDequeLen(newFs);
    if (newLen == m_dequeLen) return;
    for (int ch = 0; ch < m_cfg.maxChannels; ++ch) {
        auto& old = m_buffers[ch];
        if (old.empty()) continue;
        int take = std::min(static_cast<int>(old.size()), newLen);
        std::deque<float> fresh;
        auto it = old.end() - take;
        for (; it != old.end(); ++it) fresh.push_back(*it);
        old = std::move(fresh);
    }
    m_dequeLen = newLen;
}

// ─── Background thread ───

void ZmqClient::run() {
    zmq::pollitem_t items[2];
    items[0] = { static_cast<void*>(*m_dataSock), 0, ZMQ_POLLIN, 0 };
    items[1] = { static_cast<void*>(*m_hbSock),   0, ZMQ_POLLIN, 0 };

    while (!m_stopFlag) {
        try {
            sendHeartbeatIfDue();
            zmq::poll(items, 2, std::chrono::milliseconds(10));

            // Heartbeat reply
            if (items[1].revents & ZMQ_POLLIN) {
                if (m_waitingHbReply) {
                    try {
                        zmq::message_t reply;
                        (void)m_hbSock->recv(reply, zmq::recv_flags::dontwait);
                        m_waitingHbReply = false;
                    } catch (...) {}
                }
            }

            // Data frames
            if (items[0].revents & ZMQ_POLLIN) {
                try {
                    std::vector<zmq::message_t> frames;
                    auto res = zmq::recv_multipart(*m_dataSock, std::back_inserter(frames),
                                                   zmq::recv_flags::dontwait);
                    if (!res || frames.size() < 2) continue;

                    // frames[1] = JSON header
                    std::string headerStr(static_cast<const char*>(frames[1].data()),
                                          frames[1].size());
                    nlohmann::json header;
                    try { header = nlohmann::json::parse(headerStr); }
                    catch (...) { continue; }

                    std::string typ = header.value("type", std::string());
                    const void* payload = (frames.size() >= 3) ? frames[2].data() : nullptr;
                    size_t payloadLen   = (frames.size() >= 3) ? frames[2].size() : 0;

                    if (typ == "data")
                        handleData(header, payload, payloadLen);
                    else if (typ == "event")
                        handleEvent(header, payload, payloadLen);
                    else if (typ == "spike")
                        handleSpike(header, payload, payloadLen);

                } catch (const zmq::error_t&) {}
            }

        } catch (const std::exception& e) {
            if (m_cfg.verbose)
                std::cerr << "[ZmqClient] loop error: " << e.what() << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
