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
import os
from pathlib import Path

# Create colormap once for performance
HEATMAP_COLORMAP = LinearSegmentedColormap.from_list(
    "heat", ["white", "yellow", "orange", "red"]
)


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


def build_scene(viz_params, plotter, model_path, sensor_positions, sigma, radius):
    meshes = []
    mesh_names = []
    mesh_intervals = []
    vertex_index_start = 0
    vertex_index_end = 0
    
    channels = sorted(sensor_positions.keys())
    sensor_pts = np.array([sensor_positions[ch] for ch in channels], dtype=np.float64)

    if model_path:
        try:
            directory_path = Path(model_path)  # Path to the current directory
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    mesh = pv.read(model_path + file_path.name)
                    mesh = mesh.extract_surface(algorithm=None).triangulate()
                    meshes.append(mesh)
                    vertex_index_end += mesh.n_points
                    mesh_names.append(" ".join([word.capitalize() if word != "of" else word for word in file_path.name[:-6].split(" ")]))
                    mesh_intervals.append((vertex_index_start, vertex_index_end))
                    vertex_index_start = vertex_index_end
        except Exception as e:
            raise ValueError(f"Error reading model file: {e}")

        # Out-of-range vertices get clim_min (white in our colormap)
        totalMesh = pv.merge(meshes)
        totalMesh["amplitude"] = np.full(totalMesh.n_points, viz_params["viz_range"][0], dtype=np.float64)
        
        mesh_pts = np.array(totalMesh.points, dtype=np.float64)
        totalWeights, totalRange = compute_rbf_weights(mesh_pts, sensor_pts, sigma, radius)
        
        plotter.add_mesh(
            totalMesh,
            scalars="amplitude",
            cmap=HEATMAP_COLORMAP,
            clim=viz_params["viz_range"],
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={"title": "RMS Amplitude"},
            name="heatmap_surface",
            nan_opacity=0.0,
            opacity=1.0,
        )

        sensor_cloud = pv.PolyData(sensor_pts)
        plotter.add_mesh(sensor_cloud, color="black", point_size=8,
                        render_points_as_spheres=True, name="sensor_dots",
                        pickable=False)

    for i, ch in enumerate(channels):
        plotter.add_point_labels(
            pv.PolyData(sensor_pts[i:i+1]),
            [f"CH{ch}"],
            font_size=10,
            text_color="black",
            point_size=1,
            name=f"label_{ch}",
            pickable=False,
        )
    muscles = {
        "names": mesh_names,
        "intervals": mesh_intervals,
        "meshes": meshes,
        "visible": [True] * len(mesh_names),
        "buttons": [],
    }

    return muscles, totalMesh, totalWeights, totalRange, channels, sensor_pts

def main():
    parser = argparse.ArgumentParser(description="3D EMG Surface Heatmap Viewer")
    parser.add_argument("--model", type=str, default="./models/forearm_muscles/",
                        help="Path to 3D model file (.stl, .obj, .glb, .ply)")
    parser.add_argument("--config", type=str, default="./sensor_config.json",
                        help="Path to sensor_config.json with sensor positions")
    parser.add_argument("--host", default="127.0.0.1", help="Open Ephys host")
    parser.add_argument("--port", default="5556", help="ZMQ data port")
    parser.add_argument("--window-ms", type=int, default=200,
                        help="RMS window size in milliseconds")
    parser.add_argument("--frame-rate", type=int, default=20,
                        help="Visualization update frequency in Hz")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Gaussian spread (meters). Controls how far each sensor's "
                             "influence bleeds across the surface.")
    parser.add_argument("--radius", type=float, default=0.01,
                        help="Max distance (meters) from a sensor for a vertex to be painted.")
    parser.add_argument("--clim-min", type=float, default=0.0,
                        help="RMS value that maps to the cold (white) end of the colormap")
    parser.add_argument("--clim-max", type=float, default=50.0,
                        help="RMS value that maps to the hot (red) end of the colormap")
    args = parser.parse_args()

    viz_params = {
        "frame_rate": args.frame_rate,
        "sigma": args.sigma,
        "radius_cutoff": args.radius,
        "viz_range": [args.clim_min, args.clim_max],
        "visualize_function": {"name": "RMS", "func": lambda Y: np.sqrt(np.mean(Y ** 2, axis=1))},
        "visualize_window_ms": args.window_ms,
        "mesh": None,
        "weights": None,
        "in_range": None,
        "viz_intervals": None
    }

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
        verbose=True,
        # align_to_header_index=True,
    )
    client.start()
    print(f"ZMQ client started, connecting to {args.host}:{args.port} ...")

    # --- PyVista scene ---
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plotter = BackgroundPlotter(title="EMG 3D Heatmap", window_size=(1200, 800))
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    plotter.add_axes()

    muscles, totalMesh, totalWeights, totalRange, channels, sensor_pts = build_scene(
        viz_params, plotter, args.model, sensor_positions, args.sigma, args.radius
    )

    # Initialize mesh parameters and pre-allocate vertex_scalars for performance
    viz_params["totalMesh"] = totalMesh
    viz_params["mesh"] = totalMesh
    viz_params["weights"] = totalWeights
    viz_params["in_range"] = totalRange
    viz_params["vertex_scalars"] = np.full(totalMesh.n_points, viz_params["viz_range"][0], dtype=np.float64)

    n_sensors = len(channels)

    # Camera
    center = sensor_pts.mean(axis=0)
    plotter.camera.focal_point = center
    plotter.camera.position = center + np.array([0.1, 0.05, 0.05])

    def toggleMuscleViz(idx, isChecked):
        muscles["visible"][idx] = isChecked
        start, end = muscles["intervals"][idx]
        if not isChecked:
            viz_params["vertex_scalars"][start:end] = np.nan
        else:
            viz_params["vertex_scalars"][start:end] = viz_params["viz_range"][0]
        viz_params["mesh"]["amplitude"] = viz_params["vertex_scalars"]
        viz_params["mesh"].Modified()
        plotter.render()

    def onMeshClick(picked_point, picker):
        """Handle mesh click to toggle muscle visibility."""
        dataset = picker.GetDataSet()
        if dataset is None:
            return

        mesh = pv.wrap(dataset)
        cell_id = mesh.find_containing_cell(picked_point)
        if cell_id < 0:
            return

        vertex_id = mesh.get_cell(cell_id).point_ids[0]

        for idx, (start, end) in enumerate(muscles["intervals"]):
            if start <= vertex_id < end:
                new_state = not muscles["visible"][idx]
                muscles["buttons"][idx].setChecked(new_state)
                toggleMuscleViz(idx, new_state)
                return

    # Enable click-to-toggle muscle visibility
    plotter.enable_surface_point_picking(
        callback=onMeshClick,
        show_point=False,
        show_message=False,
        left_clicking=True,
        picker='cell',
        use_picker=True,
    )

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

        data = viz_params["visualize_function"]["func"](Y)

        sensor_values = np.zeros(n_sensors, dtype=np.float64)
        for i, ch in enumerate(channels):
            if ch < len(data):
                sensor_values[i] = data[ch]

        # Interpolate in-range vertices, out-of-range stays at clim_min (white)
        # Reuse pre-allocated array for performance
        vertex_scalars = viz_params["vertex_scalars"]
        vertex_scalars.fill(viz_params["viz_range"][0])
        in_range_mask = viz_params["in_range"]
        vertex_scalars[in_range_mask] = (viz_params["weights"][in_range_mask] @ sensor_values)

        # Mask hidden muscles to NaN (rendered transparent via nan_opacity=0)
        for idx, (start, end) in enumerate(muscles["intervals"]):
            if not muscles["visible"][idx]:
                vertex_scalars[start:end] = np.nan

        viz_params["mesh"]["amplitude"] = vertex_scalars 
        viz_params["mesh"].Modified()
        plotter.render()
        print(f"[update] {viz_params["visualize_function"]["name"]}: {sensor_values.round(1)}")

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000 // viz_params["frame_rate"])

    print("\n3D viewer is running. Close the window to stop.\n")

    ################### UI Controls ############################

    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    layout.setSpacing(4)

    vizFuncBtn = QtWidgets.QPushButton("Set Visualization Function")
    vizFuncMenu = QtWidgets.QMenu()

    def setVizFunc(function):
        viz_params["visualize_function"] = function
        plotter.remove_actor("heatmap_surface")
        plotter.add_mesh(
            viz_params["mesh"],
            scalars="amplitude",
            cmap=HEATMAP_COLORMAP,
            clim=viz_params["viz_range"],
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={"title": viz_params["visualize_function"]["name"]},
            name="heatmap_surface",
            nan_opacity=0.0,
            opacity=1.0,
        )

    vizFuncMenu.addAction("RMS Amplitude", lambda: setVizFunc({"name": "RMS Amplitude", "func": lambda Y: np.sqrt(np.mean(Y ** 2, axis=1))}))
    vizFuncMenu.addAction("Absolute Spike Amplitude", lambda: setVizFunc({"name": "Abs Spike Amplitude", "func": lambda Y: np.max(np.abs(Y), axis=1)}))
    vizFuncBtn.setMenu(vizFuncMenu)
    layout.addWidget(vizFuncBtn)

    # --- Selected Muscles ---
    musclesLabel = QtWidgets.QLabel("Selected Muscles")
    musclesLabel.setStyleSheet("font-weight: bold; font-size: 12px;")
    layout.addWidget(musclesLabel)

    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollWidget = QtWidgets.QWidget()
    scrollLayout = QtWidgets.QVBoxLayout()
    scrollLayout.setContentsMargins(0, 0, 0, 0)
    scrollLayout.setAlignment(QtCore.Qt.AlignTop)
    scrollWidget.setLayout(scrollLayout)
    scrollArea.setWidget(scrollWidget)
    layout.addWidget(scrollArea, stretch=1)

    muscBtns = []

    for idx, muscleName in enumerate(muscles["names"]):
        btn = QtWidgets.QPushButton(muscleName)
        btn.setCheckable(True)
        btn.setChecked(True)
        btn.clicked.connect(lambda checked, i=idx: toggleMuscleViz(i, checked))
        scrollLayout.addWidget(btn)
        muscBtns.append(btn)

    muscles["buttons"] = muscBtns

    container.setLayout(layout)

    dock = QtWidgets.QDockWidget("Display Settings")
    dock.setWidget(container)
    plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    ###########################################################

    plotter.app.exec_()

    client.stop()
    print("Done.")

if __name__ == "__main__":
    main()
