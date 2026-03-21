"""
Real-time 3D surface heatmap visualization of EMG sensor data from Open Ephys ZMQ.

Loads a 3D model (.stl/.obj/.glb/.ply), interpolates sensor values across the
mesh surface using Gaussian RBF weighting, and colors the entire mesh as a
smooth, continuous heatmap.

Usage:
    cd python-open-ephys/examples/three_dim_viz
    python viz_3d_heatmap.py
    OR
    python viz_3d_heatmap.py --model models/forearm.stl --config sensor_config.json
"""

import json
import argparse
import numpy as np
from scipy.spatial.distance import cdist
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from pyvistaqt import BackgroundPlotter
from pyoephys.interface import ZMQClient, NotReadyError
from PyQt5 import QtWidgets
from PyQt5 import QtCore


def load_sensor_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    positions = {}
    for s in cfg["sensors"]:
        positions[int(s["channel"])] = tuple(s["position"])
    return positions


def compute_rbf_weights(mesh_points, sensor_points, sigma, radius):
    """Euclidean Gaussian RBF weights with hard radius cutoff.

    Returns: (weights, in_range_mask)
    """
    dists = cdist(mesh_points, sensor_points) # calculate distances between mesh points and sensors, yields n_mesh by n_sensors sized array
    in_range = dists.min(axis=1) <= radius # array of booleans of size n_mesh corresponding to which mesh verticies are within range of at least one sensor

    weights = np.exp(-0.5 * (dists / sigma) ** 2) # pass distance through gaussian function to yield weights for interpolation
    weights[dists > radius] = 0.0 # zero out wights for out of range verticies using 2d bool mask
    
    #sum of weights for each vertex so we can get wieghted average that isn't affected by how close the point is to many sensors
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0 # prevent division by zero for out of range values
    weights /= row_sums

    print(f"RBF weights ready. {in_range.sum()} / {len(mesh_points)} vertices in range.")
    return weights, in_range


def build_scene(plotter, model_path, sensor_positions, sigma, radius, clim):
    if model_path:
        try:
            mesh = pv.read(model_path)
        except Exception as e:
            raise ValueError(f"Error reading model file: {e}")
 
    mesh = mesh.extract_surface(algorithm=None).triangulate()

    channels = sorted(sensor_positions.keys())
    sensor_pts = np.array([sensor_positions[ch] for ch in channels], dtype=np.float64)

    mesh_pts = np.array(mesh.points, dtype=np.float64)
    weights, in_range = compute_rbf_weights(mesh_pts, sensor_pts, sigma, radius)

    # Out-of-range vertices get clim_min (white in our colormap)
    mesh["amplitude"] = np.full(mesh.n_points, clim[0], dtype=np.float64)

    plotter.add_mesh(
        mesh,
        scalars="amplitude",
        cmap=LinearSegmentedColormap.from_list(
            "heat", ["white", "yellow", "orange", "red"]
        ),
        clim=clim,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "RMS Amplitude"},
        name="heatmap_surface",
    )

    sensor_cloud = pv.PolyData(sensor_pts)
    plotter.add_mesh(sensor_cloud, color="black", point_size=8,
                     render_points_as_spheres=True, name="sensor_dots")

    for i, ch in enumerate(channels):
        plotter.add_point_labels(
            pv.PolyData(sensor_pts[i:i+1]),
            [f"CH{ch}"],
            font_size=10,
            text_color="black",
            point_size=1,
            name=f"label_{ch}",
        )

    return mesh, channels, sensor_pts, weights, in_range

def main():
    parser = argparse.ArgumentParser(description="3D EMG Surface Heatmap Viewer")
    parser.add_argument("--model", type=str, default="./models/forearm.stl",
                        help="Path to 3D model file (.stl, .obj, .glb, .ply)")
    parser.add_argument("--config", type=str, default="./sensor_config.json",
                        help="Path to sensor_config.json with sensor positions")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys host")
    parser.add_argument("--port", default="5556", help="ZMQ data port")
    parser.add_argument("--window-ms", type=int, default=200,
                        help="RMS window size in milliseconds")
    parser.add_argument("--update-ms", type=int, default=50,
                        help="Visualization update interval in milliseconds")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Gaussian spread (meters). Controls how far each sensor's "
                             "influence bleeds across the surface.")
    parser.add_argument("--radius", type=float, default=0.01,
                        help="Max distance (meters) from a sensor for a vertex to be painted.")
    parser.add_argument("--clim-min", type=float, default=0.0,
                        help="RMS value that maps to the cold (white) end of the colormap")
    parser.add_argument("--clim-max", type=float, default=50.0,
                        help="RMS value that maps to the hot (red) end of the colormap")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # --- Sensor positions ---
    if args.config:
        try:
            sensor_positions = load_sensor_config(args.config)
            print(f"Loaded {len(sensor_positions)} sensors from {args.config}")
        except Exception as e:
            raise ValueError(f"Error loading sensor config: {e}")
    # --- ZMQ client ---
    client = ZMQClient(
        host_ip=args.host,
        data_port=args.port,
        auto_start=False,
        verbose=args.verbose,
    )
    client.start()
    print(f"ZMQ client started, connecting to {args.host}:{args.port} ...")

    # --- PyVista scene ---
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plotter = BackgroundPlotter(title="EMG 3D Heatmap", window_size=(1200, 800))
    plotter.set_background("white")
    plotter.add_axes()

    clim = [args.clim_min, args.clim_max]
    mesh, channels, sensor_pts, rbf_weights, in_range = build_scene(
        plotter, args.model, sensor_positions, args.sigma, args.radius, clim
    )
    n_sensors = len(channels)

    # Camera
    center = sensor_pts.mean(axis=0)
    plotter.camera.focal_point = center
    plotter.camera.position = center + np.array([0.1, 0.05, 0.05])

    # --- Update callback ---
    last_sample_idx = [0]  # track last seen sample index to skip stale data

    def update():
        if not client.ready_event.is_set():
            return

        if not client.channel_index:
            client.set_channel_index(sorted(client.seen_nums))
            print(f"Channels detected: {client.channel_index}")
            print(f"Sample rate: {client.fs} Hz")

        try:
            Y, t = client.get_latest(int(client.fs * args.window_ms / 1000))
        except NotReadyError:
            return

        if Y is None or Y.size == 0:
            return

        # Skip if we already rendered this data
        current_idx = int(t[-1] * client.fs)
        if current_idx == last_sample_idx[0]:
            return
        last_sample_idx[0] = current_idx

        rms = np.sqrt(np.mean(Y ** 2, axis=1))

        sensor_values = np.zeros(n_sensors, dtype=np.float64)
        for i, ch in enumerate(channels):
            if ch < len(rms):
                sensor_values[i] = rms[ch]

        # Interpolate in-range vertices, out-of-range stays at clim_min (white)
        vertex_scalars = np.full(mesh.n_points, clim[0], dtype=np.float64)
        vertex_scalars[in_range] = (rbf_weights[in_range] @ sensor_values) # use boolean mask to avoid calculating amplitude values for mesh verticies that are out of range.

        mesh["amplitude"] = vertex_scalars
        mesh.Modified()
        plotter.render()
        print(f"[update] rms: {sensor_values.round(1)}")

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(args.update_ms)
    
    print("\n3D viewer is running. Close the window to stop.\n")

    ################### UI Controls ############################
    
    # 1. Create a container widget for your controls
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()

 
    btn = QtWidgets.QPushButton("Toggle Edges")
    def toggle():
        print("Toggling edges...")

    btn.clicked.connect(toggle)
    layout.addWidget(btn)
    container.setLayout(layout)

    # 3. Add the container to the plotter as a Dock
    dock = QtWidgets.QDockWidget("Display Settings")
    dock.setWidget(container)
    plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

     ###########################################################

    plotter.app.exec_()

    client.stop()
    print("Done.")

if __name__ == "__main__":
    main()
