#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPi Camera Recorder with per-frame TTL strobe, live status, optional fixed exposure/ISO,
and HDMI preview window.

Usage:
  python3 capture_with_prev.py \
      --framerate 30.0 --name session_name --quality 23 -t 10 --preview \
      [--auto-exposure|-ae] [--exposure-time|-et 10000] [--iso 200]
"""
import argparse
import signal
import time
import json
import csv
import os
import threading
from pathlib import Path

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
import RPi.GPIO as GPIO

# GPIO / TTL Setup
TTL_PIN = 11  # BOARD numbering
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TTL_PIN, GPIO.OUT, initial=GPIO.LOW)

# Globals
frame_counter = 0
timestamps = []
stop_event = threading.Event()

# Signal Handler
def signal_handler(sig, frame):
    print("\nReceived Ctrl+C -> stopping...")
    stop_event.set()
signal.signal(signal.SIGINT, signal_handler)

# Preview Mode Helper
def determine_preview_mode(preview_requested: bool):
    """
    If preview_requested, configure an HDMI QtGL preview window,
    maximizing to full screen via wmctrl hack.
    """
    if not preview_requested:
        return None
    print("Enabling fullscreen HDMI preview window")
    os.environ["DISPLAY"] = ":0"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    # QtGL preview ignores QT_FULLSCREEN, we'll maximize via wmctrl
    return Preview.QTGL

# Status Reporter Thread
def status_reporter(start_time: float):
    while not stop_event.wait(5.0):
        elapsed = time.monotonic() - start_time
        fps = frame_counter / elapsed if elapsed > 0 else 0.0
        print(f"[STATUS] Elapsed: {elapsed:.1f}s | Frames: {frame_counter} | FPS: {fps:.2f}")

# Main Recording Routine
def run_recording(
    framerate: float,
    name: str,
    quality: int,
    duration: float,
    preview_requested: bool,
    auto_exposure: bool,
    exposure_time: int | None,
    iso: int | None,
):
    print(f"run_recording: fps={framerate}, name={name}, quality={quality}, duration={duration}, preview={preview_requested}")
    # Prepare output dirs
    session_dir = Path("./RPiCameraVideos") / name
    session_dir.mkdir(parents=True, exist_ok=True)
    h264_path = str(session_dir / f"{name}.h264")
    info_path = str(session_dir / f"{name}_info.json")
    timestamps_path = str(session_dir / f"{name}_timestamps.csv")

    # Fixed frame duration
    frame_dur_us = int(1_000_000 / framerate)
    controls = {"FrameDurationLimits": (frame_dur_us, frame_dur_us)}
    if not auto_exposure:
        if exposure_time is not None:
            controls["ExposureTime"] = exposure_time
        if iso is not None:
            controls["AnalogueGain"] = iso / 100.0
    print(f"Controls: {controls}")

    # Initialize camera
    picam2 = Picamera2()
    # Configure recording + display main stream
    config = picam2.create_video_configuration(
        main={ 'size': (640, 480), 'format': 'YUV420' },
        display='main',
        controls=controls
    )
    print(f"Configuring camera: {config}")
    picam2.configure(config)
    print("Camera configured")

    # TTL callback per frame
    def ttl_strobe(request):
        global frame_counter, timestamps
        frame_counter += 1
        meta = request.get_metadata()
        ts = meta.get("SensorTimestamp", int(time.time() * 1e9)) / 1e9
        timestamps.append(ts)
        GPIO.output(TTL_PIN, GPIO.HIGH)
        time.sleep(0.0003)
        GPIO.output(TTL_PIN, GPIO.LOW)
    picam2.post_callback = ttl_strobe

    # Start preview
    preview_mode = determine_preview_mode(preview_requested)
    if preview_mode:
        print(f"Starting preview ({preview_mode})")
        picam2.start_preview(preview_mode)
        # Helper: maximize the preview window via wmctrl
        def maximize_preview():
            time.sleep(1.0)  # wait for window to appear
            os.system(
                'wmctrl -r "Picamera2 Preview" '
                '-b add,maximized_vert,maximized_horz'
            )
        threading.Thread(target=maximize_preview, daemon=True).start()

    # Launch status thread
    print(f"Recording to {h264_path} @ {framerate} fps")
    start_time = time.monotonic()
    threading.Thread(target=status_reporter, args=(start_time,), daemon=True).start()

    # Encoder
    try:
        encoder = H264Encoder(quality=quality)
    except TypeError:
        encoder = H264Encoder()
    print(f"Encoder: {encoder}")

    # Start recording
    picam2.start_recording(encoder, h264_path)
    print("Recording started")

    # Run until done or interrupted
    try:
        if duration > 0:
            stop_event.wait(duration)
        else:
            stop_event.wait()
    finally:
        stop_event.set()
        print("Stopping recording...")
        picam2.stop_recording()
        if preview_mode:
            picam2.stop_preview()
        GPIO.cleanup()

        # Save timestamps
        with open(timestamps_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec"] )
            for t in timestamps:
                writer.writerow([t])
        # Save metadata
        info = {
            "name": name,
            "framerate": framerate,
            "quality": quality,
            "duration": duration,
            "frame_duration_us": frame_dur_us,
            "start_time": start_time,
            "preview": preview_requested,
            "auto_exposure": auto_exposure,
            "frames_recorded": frame_counter
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Saved video: {h264_path}")
        print(f"Saved timestamps: {timestamps_path}")
        print(f"Saved metadata: {info_path}")

# CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RPi Camera Recorder w/ TTL strobe + live status + optional exposure/ISO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--framerate", type=float, default=30.0,
                        help="Target recording frame rate (fps)")
    parser.add_argument("--name", type=str, default="video",
                        help="Session name (folder & file prefix)")
    parser.add_argument("--quality", type=int, default=23,
                        help="Encoder quality parameter, if supported")
    parser.add_argument("-t", "--time", type=float, default=10.0,
                        help="Duration (s); -1 for infinite")
    parser.add_argument("--preview", action="store_true",
                        help="Enable HDMI fullscreen preview")
    parser.add_argument("-ae", "--auto-exposure", action="store_true",
                        help="Use automatic exposure/gain")
    parser.add_argument("-et", "--exposure-time", type=int, default=None,
                        help="Manual exposure time (us)")
    parser.add_argument("--iso", type=int, default=None,
                        help="Manual ISO (100->1x gain)")
    args = parser.parse_args()
    run_recording(
        framerate         = args.framerate,
        name              = args.name,
        quality           = args.quality,
        duration          = args.time,
        preview_requested = args.preview,
        auto_exposure     = args.auto_exposure,
        exposure_time     = args.exposure_time,
        iso               = args.iso,
    )
