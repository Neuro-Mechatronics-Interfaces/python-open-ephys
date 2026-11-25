"""Known-pulse fixtures for Open Ephys Binary and its ZMQ plugin."""

import json
import time
import uuid

import numpy as np
import pytest
import zmq

from pyoephys.interface import ZMQClient
from pyoephys.interface._zmq_sync import ZMQSync
from pyoephys.io import read_digital_edges
from pyoephys.sync import DigitalEdge, SampleSpan


def test_binary_preserves_lines_source_and_native_counters(tmp_path):
    channels = ["ADC/TTL", "Probe/TTL", "MessageCenter"]
    manifest = {"events": [{"folder_name": folder + "/", "sample_rate": 1000}
                          for folder in channels]}
    (tmp_path / "structure.oebin").write_text(json.dumps(manifest), encoding="utf-8")
    for name, number in zip(channels[:2], [100, 200]):
        folder = tmp_path / "events" / name
        folder.mkdir(parents=True)
        np.save(folder / "states.npy", np.array([4, -4, 256], dtype="<i2"))
        np.save(folder / "sample_numbers.npy", np.array([number, number + 8, number + 8], dtype="<i8"))
    edges = read_digital_edges(tmp_path, segment=3)
    assert len(edges) == 6
    assert edges[:3] == [
        DigitalEdge(100, 4, True, 1000, "ADC", 3, "ADC/TTL"),
        DigitalEdge(108, 4, False, 1000, "ADC", 3, "ADC/TTL"),
        DigitalEdge(108, 256, True, 1000, "ADC", 3, "ADC/TTL"),
    ]
    assert [edge.sample_number for edge in read_digital_edges(
        tmp_path / "structure.oebin", event_folder="Probe/TTL",
    )] == [200, 208, 208]
    np.save(tmp_path / "events/ADC/TTL/states.npy", np.array([0, 1, -1], dtype="<i2"))
    with pytest.raises(ValueError, match="Invalid TTL"):
        read_digital_edges(tmp_path)


def test_zmq_live_timing_notifications_and_legacy_waveforms():
    endpoint = f"inproc://sync-{uuid.uuid4().hex}"
    with zmq.Context.instance().socket(zmq.XPUB) as pub:
        pub.bind(endpoint)
        client = ZMQClient(data_port=endpoint, heartbeat_port="", max_channels=3, sync_buffer_size=24)
        try:
            client.start()
            assert pub.poll(3000, zmq.POLLIN)
            assert pub.recv() == b"\x01"

            def packet(sequence, kind, content, payload):
                pub.send_multipart([b"DATA", json.dumps({
                    "type": kind, "message_num": sequence, "content": content,
                }).encode(), payload])

            def data(sequence, first):
                content = {"stream": "ADC", "sample_rate": 1000, "channel_num": 0, "num_samples": 2}
                if first is not None:
                    content["sample_num"] = first
                packet(sequence, "data", content, np.array([1, 2], dtype=np.float32).tobytes())

            def ttl(sequence, sample, high):
                packet(sequence, "event", {"type": 3, "stream": "ADC", "source_node": 100,
                                           "sample_num": sample}, bytes([3, high]) + bytes(8))

            data(1, 100)
            ttl(2, 101, True)
            data(3, 105)
            ttl(4, 106, False)
            data(5, None)
            ttl(1, 0, True)  # GUI restart. Event arrives before the first waveform.
            data(2, 0)
            data(3, 100)
            ttl(4, 1, True)  # Playback rewind, without resetting message numbers.
            data(5, 0)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with client._lock:
                    if client.total_samples_written == 12:
                        break
                time.sleep(0.005)
            received, evicted = client.drain_sync()
            assert len(received) == 10
            assert evicted == 0

            spans = [item for item in received if isinstance(item, SampleSpan)]
            assert [(s.first_sample, s.sample_number_source, s.missing_samples, s.segment) for s in spans] == [
                (100, "device", None, 0), (105, "device", 3, 0),
                (4, "reconstructed", None, 0), (0, "device", None, 1),
                (100, "device", 98, 1), (0, "device", None, 2),
            ]
            assert [item for item in received if isinstance(item, DigitalEdge)] == [
                DigitalEdge(101, 4, True, 1000, "ADC", 0, "100"),
                DigitalEdge(106, 4, False, None, "ADC", None, "100"),
                DigitalEdge(0, 4, True, 1000, "ADC", 1, "100"),
                DigitalEdge(1, 4, True, 1000, "ADC", 2, "100"),
            ]
            np.testing.assert_array_equal(client.get_latest(2)[0], [[1, 2]])
        finally:
            client.close()


def test_zmq_buffer_reports_eviction_and_ignores_invalid_event():
    sync = ZMQSync(1)
    sync.event({"type": 3, "sample_num": 3}, bytes([0, 5]) + bytes(8))
    assert sync.drain() == ([], 0)
    sync.event({"type": 3, "sample_num": 3}, bytes([255, 1]) + bytes(8))
    sync.event({"type": 3, "sample_num": 8}, bytes([255, 0]) + bytes(8))
    items, evicted = sync.drain()
    assert items == [DigitalEdge(8, 256, False, None, "unknown", None)]
    assert evicted == 1
    assert sync.drain() == ([], 0)


def _wave(sync, sample, *, rate=1000, stream="ADC", channel=0):
    sync.waveform({"stream": stream, "sample_num": sample, "sample_rate": rate}, channel, 10, 0.0)


def _ttl(sync, sample, *, stream="ADC"):
    sync.event({"stream": stream, "type": 3, "sample_num": sample}, bytes([3, 1]) + bytes(8))


def test_event_before_rewind_waveform_uses_new_segment():
    sync = ZMQSync(20)
    sync.observe_message({"message_num": 20})
    _wave(sync, 100)
    sync.observe_message({"message_num": 21})
    _ttl(sync, 1)
    sync.observe_message({"message_num": 22})
    _wave(sync, 0)
    items, evicted = sync.drain()
    assert [item.segment for item in items] == [0, 1, 1]
    assert isinstance(items[1], DigitalEdge)
    assert items[1].sample_rate == 1000
    assert evicted == 0


def test_delayed_event_does_not_trigger_a_rewind():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 1)
    _wave(sync, 110)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, 0, 0]


def test_boundary_event_outside_new_block_is_unknown():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 105)
    _wave(sync, 0)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, None, 1]
    assert items[1].sample_rate is None


def test_event_before_rate_change_uses_new_rate_and_segment():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 111)
    _wave(sync, 110, rate=2000)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, 1, 1]
    assert items[1].sample_rate == 2000


@pytest.mark.parametrize("message_reset", [False, True])
def test_restart_does_not_relabel_preceding_pending_events(message_reset):
    sync = ZMQSync(20)
    sync.observe_message({"message_num": 20})
    _wave(sync, 100)
    _ttl(sync, 1)
    if message_reset:
        sync.observe_message({"message_num": 1})
    else:
        sync.restart()
    _ttl(sync, 2)
    _wave(sync, 0)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, None, 1, 1]


def test_drain_before_waveform_reports_unknown_without_later_mutation():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 1)
    items, _ = sync.drain()
    edge = items[1]
    assert edge.segment is None
    _wave(sync, 0)
    assert edge.segment is None
    assert sync.drain()[0][0].segment == 1


def test_pending_events_preserve_order_across_streams_and_evictions():
    sync = ZMQSync(3)
    _ttl(sync, 1)
    _ttl(sync, 2, stream="Probe")
    _wave(sync, 0)
    _wave(sync, 0, stream="Probe")
    items, evicted = sync.drain()
    assert [type(item) for item in items] == [DigitalEdge, SampleSpan, SampleSpan]
    assert [item.stream_id for item in items] == ["Probe", "ADC", "Probe"]
    assert items[0].segment == items[2].segment
    assert evicted == 1


def test_missing_indices_do_not_resolve_pending_boundary():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 1)
    _wave(sync, None)
    _wave(sync, 0)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, 1, 0, 1]


def test_rate_change_without_indices_leaves_boundary_event_unknown():
    sync = ZMQSync(20)
    _wave(sync, 100)
    _ttl(sync, 111)
    _wave(sync, None, rate=2000)
    _wave(sync, 110, rate=2000)
    items, _ = sync.drain()
    assert [item.segment for item in items] == [0, None, 1, 1]
    assert items[1].sample_rate is None


def _binary_fixture(root, states, numbers, *, name="ADC/TTL"):
    (root / "structure.oebin").write_text(json.dumps({"events": [
        {"folder_name": name, "sample_rate": 1000},
    ]}), encoding="utf-8")
    folder = root / "events" / name
    folder.mkdir(parents=True)
    np.save(folder / "states.npy", np.asarray(states, dtype="<i2"))
    np.save(folder / "sample_numbers.npy", np.asarray(numbers, dtype="<i8"))
    return root


def test_binary_empty_ttl_channel(tmp_path):
    root = _binary_fixture(tmp_path, [], [])
    assert read_digital_edges(root, event_folder="ADC/TTL") == []


@pytest.mark.parametrize("states,numbers", [([1], []), ([1, -1], [2, 1])])
def test_binary_rejects_invalid_lengths_and_counters(tmp_path, states, numbers):
    root = _binary_fixture(tmp_path, states, numbers)
    with pytest.raises(ValueError, match="Invalid TTL"):
        read_digital_edges(root)


@pytest.mark.parametrize("name", ["../outside", "..\\outside"])
def test_binary_rejects_paths_outside_event_root(tmp_path, name):
    (tmp_path / "structure.oebin").write_text(json.dumps({"events": [
        {"folder_name": name, "sample_rate": 1000},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the recording"):
        read_digital_edges(tmp_path)
