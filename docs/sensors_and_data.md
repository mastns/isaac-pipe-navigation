# Sensors, frames and recorded data

Everything on this page was checked against the code and, where marked *(measured)*, against bags recorded with
Isaac Sim 5.1.0 / Isaac Lab 2.3.2 (`scripts/headless_demo.py`).

## Frames

* **Chassis frame**: the articulation root link of the Jetbot (`/World/envs/env_0/Robot/chassis`), x forward, y left,
  z up. Its origin is at axle height (about 3.4 cm above the floor when the rover rests on its wheels).
* **World frame**: the Isaac Sim stage (z up, metres). The rover starts at (0.3, 0, 0.15) m with identity orientation
  and settles at z of about 0.116 m. The pipe axis runs from the mouth at x = 0.1 m along +x.

## Sensor placement (chassis frame, metres)

| Sensor | Position (x, y, z) | Orientation | Topic |
|---|---|---|---|
| front left camera | (0.12, +0.03, 0.10) | looks along +x | `/stereo/front/left/image_raw` |
| front right camera | (0.12, -0.03, 0.10) | looks along +x | `/stereo/front/right/image_raw` |
| rear left camera | (-0.10, +0.03, 0.10) | yawed 180 deg, looks along -x | `/stereo/rear/left/image_raw` |
| rear right camera | (-0.10, -0.03, 0.10) | yawed 180 deg, looks along -x | `/stereo/rear/right/image_raw` |
| IMU 0 | (0.050, 0.000, 0) | chassis axes | `/imu0` |
| IMU 1 | (-0.025, +0.0433, 0) | chassis axes | `/imu1` |
| IMU 2 | (-0.025, -0.0433, 0) | chassis axes | `/imu2` |

The IMUs sit on a ring of radius 5 cm at 0, 120 and 240 degrees (`IMU_RADIUS`, `IMU_ANGLES_DEG`). The three black
spheres drawn above the chassis mark the ring positions but are 12 cm higher than the sensors (visual only).

**Rear pair naming.** The rear cameras are named by the side of the robot they sit on, not by stereo convention:
`rear_left` (y = +0.03) is physically the *right-hand* camera of the rear-looking rig: in the `rear_left` camera frame
(x forward, y left, z up) the baseline to `rear_right` is (0, +0.06, 0), whereas for the front pair it is (0, -0.06, 0).
Swap the two rear topics (or negate the baseline in the extrinsics) before feeding them to a stereo pipeline that assumes
a left/right order.

## Cameras

All four cameras use Isaac Lab's default `PinholeCameraCfg` at 320 x 240 px:

| | |
|---|---|
| focal length / horizontal aperture | 24.0 / 20.955 (Isaac Lab units) |
| intrinsics (Isaac Lab convention) | fx = fy = 366.50 px, cx = 160, cy = 120 |
| field of view | 47.2 deg horizontal, 36.3 deg vertical |
| distortion | none (square pixels, no rolling shutter, no sensor noise) |
| baseline | 0.06 m (both pairs) |
| disparity | `fx * b / Z` = 22.0 px m / Z (73 px at 0.3 m) |

The recorded images are **grayscale** (`mono8`): the 8-bit tone-mapped RGB output of the renderer is converted with the
ITU-R BT.601 weights (0.299, 0.587, 0.114) and truncated (not rounded) to `uint8`. The "Stereo View" window of the GUI
shows the colour images.

**Image formation.** Rendering is not configured by this project; it uses Isaac Lab's defaults: the "balanced" RTX preset
(shadows and direct lighting on, but no global illumination, reflections or ambient occlusion) with DLSS anti-aliasing,
which reconstructs each frame from a lower-resolution render using motion data of the previous frames, and the
deep-learning denoiser. Isaac Sim logs a DLSS warning about the small render resolution at start-up. Because of this the
images of two identical runs differ slightly *(measured: mean absolute difference of about 0.1 to 0.5 grey levels)*.
Changing the rendering mode (`--rendering_mode`) changes the recorded images.

### Camera frames and timing

The cameras use Isaac Lab's `world` convention: x forward, y left, z up. A pipeline that expects the optical frame
(x right, y down, z forward) needs the rotation `R_chassis<-optical` below (the columns are the optical axes written in
chassis coordinates); the translations are the positions in the table above.

| | `R_chassis<-optical` |
|---|---|
| front pair | `[[0, 0, 1], [-1, 0, 0], [0, -1, 0]]` |
| rear pair | `[[0, 0, -1], [1, 0, 0], [0, -1, 0]]` |

No TF tree and no `CameraInfo` are recorded; the intrinsics above and `sim_config.json` are the calibration. Images, IMU
samples and poses written in the same physics step carry the same stamp. No camera-IMU time offset is simulated; how
closely the rendered image follows the physics state of that step was not measured.

## IMUs

The IMU data come from Isaac Lab's `Imu` sensor: the acceleration is the finite difference of the link velocity over
one physics step with the lever-arm term of the ring position, **gravity included** (a level, resting rover reads about
+9.81 m/s^2 on z), and the angular velocity is the gyro. The simulator then adds noise and a random-walk bias
(`pipe_rover/imu_noise.py`) on every physics step, i.e. at 200 Hz:

| | white noise std (per sample) | bias random-walk step std (per sample) |
|---|---|---|
| accelerometer | 0.05 m/s^2 | 0.005 m/s^2 |
| gyroscope | 0.005 rad/s | 0.001 rad/s |

In continuous-time units (`sigma_c = sigma_d * sqrt(dt)` for noise, `sigma_c = sigma_d / sqrt(dt)` for the random walk,
`dt` = 5 ms):

| | noise density | random walk |
|---|---|---|
| accelerometer | 3.54e-3 m/s^2/sqrt(Hz) | 7.07e-2 m/s^3/sqrt(Hz) |
| gyroscope | 3.54e-4 rad/s/sqrt(Hz) | 1.41e-2 rad/s^2/sqrt(Hz) |

The bias starts at zero when the simulator starts, evolves **whether or not a bag is being recorded**, and is never
re-initialised, so its standard deviation after 10 s / 60 s / 600 s of simulation time is 0.22 / 0.55 / 1.7 m/s^2
(accelerometer) and 0.045 / 0.11 / 0.35 rad/s (gyroscope). The three IMUs have independent noise and bias. The `Imu`
messages carry no orientation (`orientation_covariance[0] = -1`) and zero (unknown) covariances; the noise parameters
are written to `sim_config.json` next to each bag.

## Spray marking

Every `--auto_spray_interval` seconds of simulation time (0.5 s, first burst at t = 0.5 s) and on key `P`, the
rover casts `--spray_cluster_size` (120) PhysX rays from the chassis origin into a cone around its forward axis
(half-angle 70 degrees, range 3 m). The polar angle of each ray is drawn uniformly (not uniformly in solid angle), so the
rays, and therefore the dots, concentrate near the axis. Every hit that is not on the rover or the ground plane
leaves an emissive orange sphere (radius 6 mm +-30 %) on the wall, offset along the surface normal. Rays that start
inside the rover's collider are re-cast from just behind that hit. Each dot is one USD prim, so a long run creates many
prims (about 240 per simulated second when every ray hits).

PhysX scene queries are enabled explicitly (`enable_scene_query_support=True`); Isaac Lab only enables them
automatically when the GUI is running, so without this setting a `--headless` run places no dots.

## Headlights

Two `UsdLux.SphereLight` lights (radius 2 cm, intensity 5000, colour (1, 0.97, 0.92)) at (0.12, +-0.025, 0.08) in the
chassis frame, shining along the chassis +x with a cone half-angle of 35 degrees, softness 0.25 and focus 3. They are
moved with the chassis on every step. The rear cameras are not lit by the headlights (they shine forward only and the
renderer has no indirect lighting); their images show the ambient and dome lighting and the self-luminous dots.

## Recorded ROS 2 bag

`sqlite3` storage, CDR serialisation, no compression. One bag directory per recording,
`<ros2_bag_dir>/stereo_imus_<YYYYmmdd_HHMMSS>/`, containing `metadata.yaml`, the `.db3` file and `sim_config.json`
(script version, package versions, timing, sensor parameters and all command-line arguments).

| Topic | Type | Frame id | Rate |
|---|---|---|---|
| `/stereo/front/left/image_raw`, `/stereo/front/right/image_raw` | `sensor_msgs/msg/Image`, `mono8`, 320 x 240 | `stereo_front_left`, `stereo_front_right` | 28.57 Hz |
| `/stereo/rear/left/image_raw`, `/stereo/rear/right/image_raw` | `sensor_msgs/msg/Image`, `mono8`, 320 x 240 | `stereo_rear_left`, `stereo_rear_right` | 28.57 Hz |
| `/imu0`, `/imu1`, `/imu2` | `sensor_msgs/msg/Imu` | `imu_0`, `imu_1`, `imu_2` | 200 Hz |
| `/ground_truth/pose` | `geometry_msgs/msg/PoseStamped` | `world` | 200 Hz |

* **Rates** *(measured)*: the IMUs and the ground truth are written on every 5 ms physics step. The images are written
  when an accumulator of physics steps reaches `IMAGE_RECORD_PERIOD` = 0.033 s; with a 5 ms step that happens every
  7th step, i.e. every 35 ms = **28.57 Hz** (not 30 Hz).
* **Time stamps** are simulation time in integer nanoseconds, exact multiples of 5 ms, starting at 0 when the main
  loop starts. Header stamps and bag times are identical. The first message of a bag carries the simulation time at
  which `B` was pressed. No `/clock` is recorded.
* **Ground truth** is the pose of the chassis root link (not the centre of mass, an IMU or a camera) in the world frame;
  the quaternion is converted from Isaac's (w, x, y, z) to the ROS fields.
* **Size** *(measured)*: about 9.1 MB (8.7 MiB) per simulated second, about 0.55 GB per minute.
* **Inspect / convert**: `ros2 bag info <bag>`, `python3 scripts/check_bag.py <bag>` (no ROS needed), and
  `rosbags-convert --src <bag> --dst out.bag` (package `rosbags`) turns it into a ROS 1 bag with all 8 topics. A bag whose
  recording was cut off (the process was killed) has no `metadata.yaml`; `ros2 bag reindex <bag> sqlite3` rebuilds it, and
  `check_bag.py` reads the bare `.db3` anyway.

## Behaviours worth knowing

* The rover is driven by key presses that add up (`Se2Keyboard`): commands are binary (no ramp), a missed key release
  leaves the rover driving, and Left/Right do nothing. Key `L` zeroes the command; if a drive key is still held, releasing
  it afterwards drives the rover the other way.
* The drive commands are nominal (the wheel radius and base used to convert (v, w) to wheel speeds, 0.0325 m and 0.112 m,
  are not those of the Jetbot asset). *Measured*: on the straight part of the pipe the forward speed matched the
  commanded 0.25 m/s; turning in place on the curved pipe floor achieved less yaw rate than commanded (5 to 17 % less at
  2 rad/s depending on the direction, about 75 % less at 1 rad/s). Use `/ground_truth/pose` for the motion, not the commands.
* The rover is dropped from z = 0.15 m onto the pipe floor when the simulation starts, which produces an impact in the IMU
  data (accelerations of about 200 m/s^2 within the first 0.1 s) and it comes to rest slightly nose-up (pitch about 5 degrees).
  Start recording after the first half second (`scripts/headless_demo.py` does).
* `R` puts the rover back at its start pose. An open bag keeps recording, so the ground-truth pose jumps; stop the
  recording first if that matters. The IMU bias and the simulation time are not reset.
* Spray dots, the pipe and the headlights are authored on the USD stage at run time; the pipe collider becomes active at
  the next physics step after it is spawned.
* The Jetbot asset (32 MB) and the default ground plane are fetched from NVIDIA's asset server on every start.
* Runs are not bit-reproducible in general (hand-driven, RTX rendering). With the same scripted drive and the same
  `--seed` on the same machine the IMU and ground-truth streams were identical *(measured: maximum absolute difference 0)*;
  the images were not (rendering noise).
* `sim_config.json` stores paths below the repository as `<repo>/...`.
