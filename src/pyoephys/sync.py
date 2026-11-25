"""Public sample-clock metadata for hardware synchronization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DigitalEdge:
    """A digital transition on a device's sample clock within one recording segment.

    Lines are one-based. The sample rate is unknown if a live event arrives
    before its stream metadata. Compare sample numbers within a stream and
    segment, using a shared physical marker to compare between clocks.
    A live edge has segment=None when its recording segment is ambiguous.
    """

    sample_number: int
    line: int
    rising: bool
    sample_rate: float | None
    stream_id: str
    segment: int | None
    source: str | None = None


@dataclass(frozen=True)
class SampleSpan:
    """Timing of one continuous ZMQ packet; channel indices are zero-based."""

    first_sample: int
    sample_count: int
    sample_rate: float | None
    stream_id: str
    segment: int
    channel: int
    # "device" copies the sender's sample_num. "reconstructed" counts local samples.
    sample_number_source: str
    missing_samples: int | None
    received_monotonic: float
