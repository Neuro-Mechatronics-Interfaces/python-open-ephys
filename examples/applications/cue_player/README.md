# LSL Cue Player

`cue_player.py` is a standalone participant-facing prompt application. It
publishes timestamped string markers to an LSL stream while displaying a
configured gesture or task schedule.

It does not launch, require, or depend on a recorder or another repository.
Any LSL-compatible recording client can subscribe to the marker stream.

Run it from the repository root with:

```bash
python examples/applications/cue_player/cue_player.py \
  --schedule examples/applications/cue_player/gesture_protocol.json
```

The example schedules are in this directory. Use the manual event publisher
at `examples/interface/lsl/lsl_event_sender.py` when you need ad hoc markers
instead of a timed prompt schedule.
