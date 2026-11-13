# Provenance and verification

`pipe_rover_sim.py` is a cleaned-up, tested version of the Isaac Sim script `stereo_imus_V9.py` (its name during
development) that this code derives from. This page lists what changed and how it was checked.

## What is (not) in this repository

The simulation environment, the sensors, the spray marking and the recorder are here. Recorded bags, visual-inertial
estimators and analysis or plotting code are **not** part of this repository.

## Differences to the original script

Intended behaviour changes (none of them alters a recording made with the original defaults, except where noted):

| Change | Why |
|---|---|
| PhysX scene queries are enabled explicitly | In Isaac Lab 2.3.2 they are only switched on when the GUI runs, so the spray ray casts found nothing in `--headless` runs. With the GUI nothing changes |
| Key `R` really resets the rover | it only cleared actuator and sensor buffers before (the rover stayed where it was) |
| Time stamps are exact multiples of 5 ms | the original truncated a float, so most stamps were 1 ns low and header and bag time could differ by 1 ns |
| `array.array` payload for the images | about 600 times faster than assigning `bytes`; the message content is identical |
| `sim_config.json` is written next to every bag | provenance: script version, package versions, sensor parameters, all arguments |
| `--seed` | optional; seeds NumPy and PyTorch (IMU noise and spray patterns) |
| asset paths and the default bag directory (`<repo>/rosbags`) are relative to the repository, a missing pipe model is an error, `--pipe_stl` is gone | the original resolved assets against the current directory and silently ran without a pipe |
| every error in the start-up or the main loop is reported and ends the program with status 1 | an error such as a missing pipe model could end the program with status 0 and without a message |
| numeric options are validated, `--num_envs` must be 1, `--control_mode auto` warns that it has no controller | these options were accepted but did nothing useful |
| the follow camera is ignored in `--headless` runs | there is no viewport to move |
| the bag directory is created inside the error handling and bag names do not collide | a second recording started within the same second failed to open its bag |
| the simulation context is cleared before the app closes | otherwise `simulation_app.close()` can hang (seen in headless runs): Isaac Lab's keep-rendering-after-Stop handler calls `render()` while the app shuts down |
| the wheel joints are looked up with `preserve_order=True` and materials are bound with `MaterialBindingAPI.Apply` | robustness, no change of behaviour |
| debug output, dead code and the Pillow-less texture fallback were removed; helpers moved to `pipe_rover/` | readability and tests |
| `IMAGE_RECORD_RATE` and `HUD_UPDATE_RATE` were renamed to `..._PERIOD` | they are periods in seconds |

Deliberately **unchanged** (so that results stay comparable), documented in [sensors_and_data.md](sensors_and_data.md):
the 28.57 Hz image rate, the truncating grayscale conversion, the rear-pair naming, the IMU noise and random-walk bias
model, the angular distribution of the spray rays, the un-ramped keyboard commands and the nominal wheel geometry.

## Verification

* **Unit tests** (`pytest`, about 100 tests): the procedural textures are pixel-identical to the images of the original
  script (golden hashes), the IMU noise model and the spray sampling reproduce the original inline code bit for bit under
  the same seed, the simulator defaults and command-line checks are pinned, and the ROS 2 recorder is tested twice: against
  strict fakes of the ROS 2 modules (runs anywhere, also in CI) and with a write/read round trip through the real
  `rosbag2_py` (needs the ROS 2 bindings; run in the Isaac Sim Python 3.11 environment).
* **Simulator regression** (Isaac Sim 5.1.0, Isaac Lab 2.3.2, RTX 3080 Ti): the original script and this version were run
  headless with the same seed and the same scripted drive (8 s). All message counts and rates are equal; the
  `/imu0..2` and `/ground_truth/pose` messages have a maximum absolute field difference of 0 (the stamps differ by the 1 ns
  mentioned above); the images differ from run to run by the same amount between two runs of the *same* version
  (renderer noise, mean absolute difference 0.1 to 0.5 grey levels). Both versions placed the same number of spray dots (1914).
  The original needed scene queries forced on for this comparison, see the first row of the table.
* **Other real runs**: a 15 s scripted drive along the pipe centre line (follows both 45 degree bends to the far mouth,
  3046 dots), the `R` reset, `--seed` reproducibility (identical IMU streams for equal seeds, different for different
  seeds), `--pipe_texture`, `--headlight_intensity 0` and `--help`. All of them ran headless. The GUI-only branches of the
  main loop (the `Stereo View` HUD window, updated from the camera images, and the follow camera, moved every step) were
  run for 5 s in a headless process with the GUI branches switched on: nothing is displayed there, but the `omni.ui` and
  viewport calls run without error. An interactive session with a visible window and a real keyboard was not run (the
  keyboard code was reviewed against the Isaac Lab sources and exercised with stand-ins).
* **Code review**: the original script went through an automated, AI-assisted multi-agent audit (eight review passes:
  logic, Isaac Lab API usage, recorder, sensor modelling, robustness, code quality, documentation, release hygiene; 63 merged
  findings, each re-checked by separate verification passes), and this repository through a second automated review (six
  passes) before it was published. The fixes above and the notes in this documentation come from those reviews.
