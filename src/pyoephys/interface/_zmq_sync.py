"""Optional sample-clock notifications from the Open Ephys ZMQ plugin."""

from collections import deque
from dataclasses import dataclass, field, replace
import math

from ..sync import DigitalEdge, SampleSpan


def _integer(value):
    # The wire protocol uses JSON integers. Substituting or rounding a
    # missing/bogus value could associate an event with the wrong sample.
    return value if type(value) is int else None


@dataclass
class _Stream:
    segment: int
    rate: float | None = None
    reference_channel: int | None = None
    last_start: int | None = None
    ends: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingEdge:
    edge: DigitalEdge


class ZMQSync:
    def __init__(self, capacity):
        self.notifications = deque(maxlen=capacity)
        self.evicted = 0
        self.streams = {}
        self.next_segment = 0
        self.last_message = None
        self.pending_streams = set()

    def _new_stream(self, key):
        stream = _Stream(self.next_segment)
        self.next_segment += 1
        self.streams[key] = stream
        return stream

    def restart(self):
        # Events queued before a confirmed restart must not be associated with
        # waveform metadata from the new acquisition.
        self.notifications = deque(
            (item.edge if isinstance(item, _PendingEdge) else item
             for item in self.notifications), maxlen=self.notifications.maxlen,
        )
        self.streams.clear()
        self.last_message = None
        self.pending_streams.clear()

    def observe_message(self, header):
        # The plugin increments message_num across waveform and event packets
        # and resets it at acquisition start.
        number = _integer(header.get("message_num"))
        if number is not None:
            if self.last_message is not None and number < self.last_message:
                self.restart()
            self.last_message = number

    def _append(self, item):
        if len(self.notifications) == self.notifications.maxlen:
            self.evicted += 1
        self.notifications.append(item)

    def waveform(self, content, channel, count, received):
        key = str(content.get("stream") or "unknown")
        stream = self.streams.get(key) or self._new_stream(key)
        sample = _integer(content.get("sample_num"))
        rate = content.get("sample_rate")
        rate = float(rate) if type(rate) in (int, float) and math.isfinite(rate) and rate > 0 else None
        previous = stream
        if ((rate is not None and stream.rate is not None and rate != stream.rate)
                or (channel == stream.reference_channel and sample is not None
                    and stream.last_start is not None and sample < stream.last_start)):
            stream = self._new_stream(key)
        if stream.rate is None:
            stream.rate = rate
        if stream.reference_channel is None:
            stream.reference_channel = channel
        if channel == stream.reference_channel and sample is not None:
            stream.last_start = sample

        if (sample is not None or previous is not stream) and key in self.pending_streams:
            # The plugin sends a block's events before its waveforms. Resolve
            # them in place so packet order and the single queue bound survive.
            # At a transition only events inside this block can be attributed;
            # older/delayed events may belong to either side of the boundary.
            established = previous is stream and bool(previous.ends)
            resolved = deque(maxlen=self.notifications.maxlen)
            for item in self.notifications:
                if isinstance(item, _PendingEdge) and item.edge.stream_id == key:
                    segment = (stream.segment if established or
                               (sample is not None and
                                sample <= item.edge.sample_number < sample + count) else None)
                    item = replace(
                        item.edge, segment=segment, sample_rate=stream.rate if segment is not None else None,
                    )
                resolved.append(item)
            self.notifications = resolved
            self.pending_streams.discard(key)

        # Keep reconstructed indices on their own clock. Don't report a gap
        # size across packets that lack source sample numbers.
        hardware_end, received_count = stream.ends.get(channel, (None, 0))
        missing = None
        if sample is None:
            first = received_count
            provenance = "reconstructed"
        else:
            first = sample
            provenance = "device"
            if hardware_end is not None and sample >= hardware_end:
                missing = sample - hardware_end
        stream.ends[channel] = (None if sample is None else sample + count, received_count + count)
        self._append(SampleSpan(first, count, stream.rate, key, stream.segment, channel,
                                provenance, missing, received))

    def event(self, content, payload):
        if content.get("type") != 3 or len(payload) != 10 or payload[1] not in (0, 1):
            return
        sample = _integer(content.get("sample_num"))
        if sample is None:
            return
        key = str(content.get("stream") or "unknown")
        if key not in self.streams:
            self._new_stream(key)
        node = _integer(content.get("source_node"))
        self._append(_PendingEdge(DigitalEdge(
            sample, payload[0] + 1, bool(payload[1]), None,
            key, None, None if node is None else str(node),
        )))
        self.pending_streams.add(key)

    def drain(self):
        # A drain never waits for a waveform (including event-only streams).
        items = [item.edge if isinstance(item, _PendingEdge) else item
                 for item in self.notifications]
        self.notifications.clear()
        self.pending_streams.clear()
        evicted, self.evicted = self.evicted, 0
        return items, evicted
