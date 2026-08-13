"""Small GUI for sending manual and periodic events over an LSL marker stream.

Each event is sent as a single string sample on one irregular-rate LSL marker
stream.  This is intended for task markers, button presses, state changes, and
other discrete events that need to be visible to an Open Ephys or LSL client.

Run from the repository root with::

    python examples/interface/lsl/lsl_event_sender.py

The GUI only creates the LSL outlet after ``Connect`` is pressed, so event
definitions can be edited before publishing starts.
"""

import argparse
import sys
import time

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

    HAS_QT = True
except Exception as exc:
    HAS_QT = False
    QT_IMPORT_ERROR = str(exc)

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except Exception as exc:
    StreamInfo = None
    StreamOutlet = None
    local_clock = None
    PYLSL_IMPORT_ERROR = str(exc)
else:
    PYLSL_IMPORT_ERROR = None


_DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #262b33;
    color: #c8ccd4;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3a414b;
    border-radius: 4px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #d0d4db;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { color: #c8ccd4; }
QLineEdit, QDoubleSpinBox {
    background-color: #2a3039;
    border: 1px solid #3a414b;
    border-radius: 3px;
    padding: 3px 6px;
    color: #c8ccd4;
}
QPushButton {
    background-color: #3a414b;
    border: 1px solid #4a525e;
    border-radius: 3px;
    padding: 5px 12px;
    color: #c8ccd4;
    font-weight: bold;
}
QPushButton:hover { background-color: #444d59; }
QPushButton:pressed { background-color: #505a68; }
QPushButton:disabled { color: #666; background-color: #2a3039; }
QCheckBox { color: #c8ccd4; spacing: 6px; }
QScrollArea { border: none; }
"""


def create_marker_outlet(stream_name, source_id):
    """Create the irregular-rate, one-channel string outlet used by the GUI."""
    if StreamInfo is None or StreamOutlet is None:
        raise RuntimeError(f"pylsl import failed: {PYLSL_IMPORT_ERROR}")
    info = StreamInfo(stream_name, "Markers", 1, 0.0, "string", source_id)
    return StreamOutlet(info)


class EventRow(QWidget):
    """Controls and timer for one named event."""

    def __init__(self, sender, name="event", parent=None):
        super().__init__(parent)
        self.sender = sender
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._send_periodic)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Event name / marker value")
        self.periodic = QCheckBox("Periodic")
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.01, 86400.0)
        self.interval.setDecimals(2)
        self.interval.setSingleStep(0.1)
        self.interval.setValue(1.0)
        self.interval.setSuffix(" s")
        self.send_button = QPushButton("Send")
        self.delete_button = QPushButton("Delete")

        layout.addWidget(self.name_edit, 0, 0)
        layout.addWidget(self.periodic, 0, 1)
        layout.addWidget(self.interval, 0, 2)
        layout.addWidget(self.send_button, 0, 3)
        layout.addWidget(self.delete_button, 0, 4)

        self.send_button.clicked.connect(self.send_now)
        self.delete_button.clicked.connect(self._delete)
        self.periodic.toggled.connect(self._set_timer_state)
        self.interval.valueChanged.connect(self._update_interval)
        self._set_timer_state(False)

    def send_now(self):
        name = self.name_edit.text().strip()
        if not name:
            self.sender.set_status("Enter an event name before sending", error=True)
            return
        self.sender.send_event(name)

    def _send_periodic(self):
        # A row may be configured before the outlet is connected.  In that
        # case the timer can run quietly until publishing starts.
        if self.sender.outlet is None:
            return
        self.send_now()

    def _set_timer_state(self, enabled):
        self.interval.setEnabled(enabled)
        self.sender._update_row_timer(self, enabled)

    def _update_interval(self, _value):
        if self.periodic.isChecked():
            self.sender._update_row_timer(self, True)

    def start_periodic_timer(self):
        self.timer.start(max(1, round(self.interval.value() * 1000)))

    def stop_periodic_timer(self):
        self.timer.stop()

    def _delete(self):
        self.timer.stop()
        self.setParent(None)
        self.deleteLater()
        self.sender.remove_event(self)


class LSLEventSenderWindow(QMainWindow):
    def __init__(self, stream_name="OpenEphys_Events", source_id="open_ephys_events"):
        super().__init__()
        self.outlet = None
        self.rows = []
        self._init_ui(stream_name, source_id)
        self.setStyleSheet(_DARK_STYLE)

    def _init_ui(self, stream_name, source_id):
        self.setWindowTitle("Events \u2192 LSL Marker Stream")
        self.setMinimumSize(680, 360)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(6)

        settings = QGroupBox("LSL Settings")
        grid = QGridLayout(settings)
        grid.addWidget(QLabel("Stream name"), 0, 0)
        self.stream_name_edit = QLineEdit(stream_name)
        grid.addWidget(self.stream_name_edit, 0, 1)
        grid.addWidget(QLabel("Source ID"), 1, 0)
        self.source_id_edit = QLineEdit(source_id)
        grid.addWidget(self.source_id_edit, 1, 1)
        self.sync_periodic = QCheckBox("Sync periodic timers to Connect")
        self.sync_periodic.setChecked(True)
        self.sync_periodic.setToolTip(
            "When enabled, all periodic event timers begin from the same publishing start time."
        )
        self.sync_periodic.toggled.connect(self._sync_mode_changed)
        grid.addWidget(self.sync_periodic, 2, 0, 1, 2)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._toggle_connection)
        grid.addWidget(self.connect_button, 0, 2, 3, 1)
        layout.addWidget(settings)

        events = QGroupBox("Events")
        events_layout = QVBoxLayout(events)
        header = QGridLayout()
        header.addWidget(QLabel("Name / marker value"), 0, 0)
        header.addWidget(QLabel("Mode"), 0, 1)
        header.addWidget(QLabel("Interval"), 0, 2)
        events_layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.rows_widget = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        self.rows_layout.addStretch()
        scroll.setWidget(self.rows_widget)
        events_layout.addWidget(scroll)

        add_button = QPushButton("+ Add event")
        add_button.clicked.connect(lambda: self.add_event())
        events_layout.addWidget(add_button)
        layout.addWidget(events)

        bottom = QHBoxLayout()
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #ff6666; font-weight: bold;")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        self.count_label = QLabel("Events sent: 0")
        bottom.addWidget(self.count_label)
        layout.addLayout(bottom)

        self.sent_count = 0
        self.add_event("start")
        self.add_event("stop")

    def add_event(self, name="event"):
        row = EventRow(self, name)
        self.rows.append(row)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        return row

    def remove_event(self, row):
        if row in self.rows:
            self.rows.remove(row)

    def _update_row_timer(self, row, enabled):
        """Apply the global synchronization setting to one event row."""
        if not enabled:
            row.stop_periodic_timer()
        elif not self.sync_periodic.isChecked() or self.outlet is not None:
            row.start_periodic_timer()
        else:
            # With synchronization enabled, rows wait for Connect so they all
            # receive a common timer origin.
            row.stop_periodic_timer()

    def _start_periodic_timers(self):
        for row in self.rows:
            if row.periodic.isChecked():
                row.start_periodic_timer()

    def _stop_periodic_timers(self):
        for row in self.rows:
            row.stop_periodic_timer()

    def _sync_mode_changed(self, enabled):
        if enabled:
            self._stop_periodic_timers()
            if self.outlet is not None:
                self._start_periodic_timers()
        else:
            for row in self.rows:
                if row.periodic.isChecked():
                    row.start_periodic_timer()

    def _toggle_connection(self):
        if self.outlet is not None:
            if self.sync_periodic.isChecked():
                self._stop_periodic_timers()
            self.outlet = None
            self.connect_button.setText("Connect")
            self.stream_name_edit.setEnabled(True)
            self.source_id_edit.setEnabled(True)
            self.set_status("Disconnected", error=True)
            return

        stream_name = self.stream_name_edit.text().strip()
        source_id = self.source_id_edit.text().strip()
        if not stream_name or not source_id:
            self.set_status("Stream name and source ID are required", error=True)
            return
        try:
            self.outlet = create_marker_outlet(stream_name, source_id)
        except Exception as exc:
            self.set_status(f"Connection error: {exc}", error=True)
            return
        self.connect_button.setText("Disconnect")
        self.stream_name_edit.setEnabled(False)
        self.source_id_edit.setEnabled(False)
        if self.sync_periodic.isChecked():
            self._start_periodic_timers()
        self.set_status(f"Connected: {stream_name}")

    def send_event(self, name):
        if self.outlet is None:
            self.set_status("Connect to LSL before sending events", error=True)
            return
        timestamp = local_clock() if local_clock is not None else time.time()
        try:
            self.outlet.push_sample([name], timestamp=timestamp)
        except TypeError:
            # Older pylsl versions accept the timestamp positionally.
            self.outlet.push_sample([name], timestamp)
        self.sent_count += 1
        self.count_label.setText(f"Events sent: {self.sent_count}")
        self.set_status(f"Sent: {name}")

    def set_status(self, text, error=False):
        color = "#ff6666" if error else "#44ff44"
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.status_label.setText(text)

    def closeEvent(self, event):
        for row in self.rows:
            row.timer.stop()
        self.outlet = None
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Send manual and periodic LSL marker events")
    parser.add_argument("--stream-name", default="OpenEphys_Events")
    parser.add_argument("--source-id", default="open_ephys_events")
    args = parser.parse_args(argv)

    if not HAS_QT:
        raise SystemExit(f"PyQt5 import failed: {QT_IMPORT_ERROR}")
    app = QApplication(sys.argv)
    window = LSLEventSenderWindow(args.stream_name, args.source_id)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    main()
