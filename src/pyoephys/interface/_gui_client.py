from open_ephys.control import OpenEphysHTTPServer


class GUIClient:
    """
    Wrapper around open_ephys.control.OpenEphysHTTPServer
    """
    def __init__(self, host="127.0.0.1"):
        self.server = OpenEphysHTTPServer(host)

    def start_acquisition(self, duration_sec=0):
        return self.server.acquire(duration_sec)

    def stop_acquisition(self):
        return self.server.idle()

    def idle(self):
        return self.server.idle()

    def status(self):
        return self.server.status()

    def start_record(self, duration_sec=0):
        return self.server.record(duration_sec)

    def set_recording_params(self, base_text=None, append_text=None, parent_dir=None):
        if base_text:
            self.server.set_base_text(base_text)
        if append_text:
            self.server.set_append_text(append_text)
        if parent_dir:
            self.server.set_parent_dir(parent_dir)

    def get_recording_info(self, key=""):
        return self.server.get_recording_info(key)

    def quit(self):
        return self.server.quit()

    def load_config(self, path):
        return self.server.load(path)

    def clear_signal_chain(self):
        return self.server.clear_signal_chain()

    def message(self, msg):
        return self.server.message(msg)

    def close_gui(self):
        return self.server.quit()
