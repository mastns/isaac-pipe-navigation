# Installation

The simulator needs three things: NVIDIA Isaac Sim 5.1 with Isaac Lab 2.3.2 (the scene), the ROS 2 Python bindings
(only to record bags) and two small Python packages. The simulation itself runs without ROS 2; recording is then
disabled with a warning.

Tested configuration: Ubuntu 22.04, NVIDIA RTX 3080 Ti (12 GB; the simulation uses about 4 GB), driver 560.35, Python
3.11, Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.2, PyTorch 2.7.0 (CUDA 12.8), ROS 2 Humble built for Python 3.11.

## 1. Isaac Sim and Isaac Lab

Use Python 3.11 (the version of the Isaac Sim 5.1 pip packages). The commands below are the ones used for the tested
configuration; they follow the official pip installation guides of
[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab).

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev git cmake build-essential   # Ubuntu 22.04 ships Python 3.10 only

python3.11 -m venv ~/venvs/isaac-pipe && source ~/venvs/isaac-pipe/bin/activate
pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com      # about 20 GB
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
pip install "setuptools<81" wheel                  # flatdict (an Isaac Lab dependency) needs pkg_resources to build
pip install --no-build-isolation flatdict==4.0.1
~/IsaacLab/isaaclab.sh --install none              # core packages only, no RL frameworks
pip install "packaging==23.0"                      # pinned by isaacsim-core; the steps above can upgrade it

git clone https://github.com/mastns/isaac-pipe-navigation.git && cd isaac-pipe-navigation
pip install -r requirements.txt
```

The first start asks you to accept the NVIDIA Omniverse EULA. Read it, then either accept it interactively or set
`OMNI_KIT_ACCEPT_EULA=YES` for non-interactive use. The first start also downloads and compiles extension and shader
caches and can take several minutes. The Jetbot model (32 MB) and the ground plane are fetched from NVIDIA's asset
server at run time, so the machine needs internet access on every start.

## 2. ROS 2 Python bindings (for recording)

The recorder imports `rosbag2_py`, `rclpy`, `sensor_msgs` and `geometry_msgs` in-process. Isaac Sim 5.1 runs Python 3.11
and does not bundle `rosbag2_py`, while the ROS 2 Humble binary packages are built for Python 3.10, so ROS 2 Humble has to
be built from source against Python 3.11 and sourced before the simulator starts. Follow the ROS 2 instructions for
[building Humble from source](https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Development-Setup.html) (it
lists the build dependencies) with the Isaac environment's Python active. Two CMake arguments matter: the Python
interpreter, and a pip-installed `pybind11`, because Ubuntu 22.04's `pybind11-dev` (2.9.1) fails against the Python 3.11
headers with an "incomplete type PyFrameObject" error in `rclpy`:

```bash
pip install pybind11
mkdir -p ~/ros2_humble_py311/src && cd ~/ros2_humble_py311
vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src
colcon build --packages-up-to ros2bag ros2cli rosbag2_transport rosbag2_py \
  --cmake-args -DPython3_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
```

The authors built a full Humble workspace with exactly these two CMake arguments; the reduced `--packages-up-to` list above
is the one used for their working install, but this command has not been re-run since. If `sensor_msgs` or
`geometry_msgs` are reported missing, add them to the list.

Then, in every shell that starts the simulator:

```bash
source ~/venvs/isaac-pipe/bin/activate
source ~/ros2_humble_py311/install/local_setup.bash
python -c "import rosbag2_py, rclpy; from sensor_msgs.msg import Image, Imu; from geometry_msgs.msg import PoseStamped; print('ROS 2 bindings OK')"
```

## 3. Check the installation

```bash
python scripts/headless_demo.py --duration 8
```

This drives the rover for 8 simulated seconds without a window, records a bag under `rosbags/headless_demo/` and
checks it. You should see thirteen `PASS` lines (all topics present, IMU and ground truth at 200 Hz, images at 28.57 Hz,
monotonic time stamps, the rover moved, spray dots created) and the exit status 0. The unit tests (no Isaac Sim needed) are described in
the [README](../README.md#development).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` while installing Isaac Lab | `flatdict` is built with a current setuptools that no longer ships `pkg_resources`; use the `setuptools<81` and `--no-build-isolation` lines above |
| `[WARN] ROS2 bag recording disabled (import failed)` | the ROS 2 install is not sourced, or it was built for another Python version (see section 2) |
| pip prints `requires packaging==23.0` conflicts | harmless; `pip install "packaging==23.0"` silences them |
| `ModuleNotFoundError: No module named 'torch'` (or `isaaclab`) | the Isaac Sim environment is not active; `source ~/venvs/isaac-pipe/bin/activate` first |
| `PxgCudaDeviceMemoryAllocator failed to allocate memory` | the GPU is out of memory (another Isaac Sim process still running?); the simulation needs about 4 GB |
| the window opens but the rover does not move | the Kit window needs keyboard focus; arrow keys Up/Down drive, Z/X turn |
| the textures take a while on the first start | `assets/textures/` is generated once (about half a minute) |
