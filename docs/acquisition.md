# Acquisition (Raspberry Pi)

Eye cameras are driven from a Raspberry Pi 4. The capture script is:

[`src/eye_tracking_system_tools/raspberry_pi/RPI4_vid_cap_stable.py`](../src/eye_tracking_system_tools/raspberry_pi/RPI4_vid_cap_stable.py)

It records H.264, a per-frame timestamp CSV, and an info JSON, with a TTL strobe on GPIO BOARD pin 11. Run it **on the Pi** (it needs `picamera2` and `RPi.GPIO`, which are not in the analysis conda env).

```bash
python3 src/eye_tracking_system_tools/raspberry_pi/RPI4_vid_cap_stable.py \
  --framerate 60.0 --name session_name --quality 23 -t 10 --preview
```

Optional: `--auto-exposure` / `-ae`, `--exposure-time` / `-et`, `--iso`. Output goes under `./RPiCameraVideos/<name>/`.

Printable headmount files and the Meshroom template live in [`src/eye_tracking_system_tools/utils/`](../src/eye_tracking_system_tools/utils/). See [`3D_printing_files/README.md`](../src/eye_tracking_system_tools/utils/3D_printing_files/README.md).

Before running [`meshroom_pipeline_template.mg`](../src/eye_tracking_system_tools/utils/meshroom_pipeline_template.mg), set the graph’s `sensorDatabase` and `tree` inputs to that Meshroom install’s AliceVision share folder (`cameraSensors.db` and the vlfeat vocabulary tree). Those two fields are left empty in the template.

After recording, arrange each session as a `block_xxx/` folder and open it in the [Preprocessing GUI](preprocessing.md).
