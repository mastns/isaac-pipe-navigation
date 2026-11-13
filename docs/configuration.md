# Configuration reference

Every tunable default lives in the `DEFAULTS / CONFIG` section at the top of `pipe_rover_sim.py`. Most of them can be
overridden on the command line (`python3 pipe_rover_sim.py --help`); the rest are constants of the file. Isaac Lab's own
options (`--headless`, `--device`, `--rendering_mode`, `--verbose`, ...) are accepted as well. `--verbose` also enables the
per-burst spray messages of this project.

Units: metres, seconds, radians (degrees where the name says `_DEG` or the option says deg), pixels.

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--control_mode {teleop,auto}` | `teleop` | `teleop`: keyboard drive (GUI window needs focus). `auto`: no controller; the rover stays still and recording cannot be started |
| `--num_envs` | 1 | must be 1 |
| `--seed` | unseeded | seeds NumPy and PyTorch (IMU noise and spray patterns) |
| `--follow_camera` / `--no_follow_camera` | on | the GUI viewport follows the rover (ignored with `--headless`) |
| `--follow_distance`, `--follow_height` | 0.35, 0.22 | follow camera offset behind / above the rover |
| `--hud` / `--no_hud` | on | "Stereo View" window with the four camera images (GUI only) |
| `--teleop_linear_speed`, `--teleop_angular_speed` | 0.25 m/s, 2.0 rad/s | speeds at full key press |
| `--wheel_radius`, `--wheel_base`, `--max_wheel_speed` | 0.0325, 0.112, 20 | (v, w) to wheel-speed conversion and clamp (rad/s) |
| `--stereo_width`, `--stereo_height` | 320, 240 | all four cameras (px) |
| `--stereo_baseline` | 0.06 | both stereo pairs |
| `--stereo_mount_x`, `--stereo_mount_z` | 0.12, 0.10 | front pair, chassis frame |
| `--rear_stereo_mount_x`, `--rear_stereo_mount_z` | -0.10, 0.10 | rear pair, chassis frame |
| `--pipe_usd` | `assets/pipe/Real_sim.usd` | pipe model |
| `--pipe_scale` | 1.5 | uniform scale of the pipe model |
| `--pipe_x`, `--pipe_y`, `--pipe_z` | 0.1, 0.0, 0.375 | pipe position in the world |
| `--pipe_texture` / `--no_pipe_texture` | off | procedural corrosion texture on the pipe wall (otherwise a plain grey wall) |
| `--pipe_texture_seed`, `--pipe_texture_size` | 42, 512 | corrosion texture seed and side length (px) |
| `--headlight_intensity` | 5000 | 0 switches the headlights off |
| `--headlight_cone_angle` | 35 | cone half-angle (deg) |
| `--headlight_color_r`, `--headlight_color_g`, `--headlight_color_b` | 1.0, 0.97, 0.92 | light colour |
| `--spray_dot_radius` | 0.006 | mean dot radius (m), +-30 % jitter |
| `--spray_cluster_size` | 120 | rays per burst |
| `--spray_cone_half_angle` | 70 | deg, around the rover's forward axis |
| `--spray_color_r`, `--spray_color_g`, `--spray_color_b` | 1.0, 0.35, 0.0 | paint colour (also its emission) |
| `--auto_spray_interval` | 0.5 | s of simulation time between automatic bursts (0 = manual only) |
| `--ros2_record` / `--no_ros2_record` | on | enable the bag recorder (recording is started and stopped with `B`) |
| `--ros2_bag_dir` | `<repo>/rosbags` | directory that receives the bags |
| `--ros2_bag_prefix` | `stereo_imus` | bag folder name prefix |

## Constants without a command-line option

| Constant | Value | Meaning |
|---|---|---|
| `PHYSICS_DT` | 0.005 s | physics step (200 Hz); the IMUs and the ground truth are recorded on every step |
| `IMAGE_RECORD_PERIOD` | 0.033 s | minimum period between recorded image sets (effective: every 7th step, 35 ms) |
| `HUD_UPDATE_PERIOD` | 0.1 s | refresh period of the stereo window |
| `ROBOT_INIT_POS` | (0.3, 0.0, 0.15) | initial chassis position |
| `IMU_RADIUS`, `IMU_ANGLES_DEG` | 0.05 m, (0, 120, 240) | IMU ring |
| `IMU_*_NOISE_STD`, `IMU_*_BIAS_STD` | see [sensors](sensors_and_data.md#imus) | per-sample IMU noise parameters |
| `IMU_MARKER_RADIUS`, `IMU_MARKER_Z_ABOVE` | 0.008, 0.12 | the black marker spheres (visual only) |
| `PIPE_TEX_SCALE` | 5.0 | tiling of the corrosion texture |
| `DOME_LIGHT_INTENSITY` | 250 | overcast sky dome |
| `SUN_INTENSITY`, `SUN_COLOR` | 200, (0.9, 0.88, 0.82) | distant light |
| `SUN_ELEVATION_DEG`, `SUN_AZIMUTH_DEG` | -35, 20 | rotateX / rotateY of the sun prim (the sun ends up about 50 deg above the horizon) |
| `GROUND_TEX_SCALE`, `GROUND_TEXTURE_SEED/SIZE` | 12, 99 / 1024 | mud ground texture |
| `SKY_TEXTURE_SEED`, `SKY_TEXTURE_SIZE` | 88, 2048 x 1024 | overcast sky texture |

Values that live in the helper modules: headlight geometry (`pipe_rover/headlights.py`), the spray ray range, skip margin
and dot offsets (`pipe_rover/spray_marker.py`, `pipe_rover/spray.py`), the topic names (`pipe_rover/recorder.py`).

## Generated files

Textures are generated on the first run into `assets/textures/` (about half a minute) and reused afterwards; the file name
contains the size and seed, so changing them creates a new file. Bags go to `rosbags/` (ignored by git).
