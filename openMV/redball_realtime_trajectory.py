# Real-time red ball trajectory recording, no video saving
# This work is licensed under the MIT license.
# Copyright (c) 2013-2023 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# Red ball trajectory capture (only display trajectory: start drawing when ball appears, clear after 3 seconds of disappearance)

import sensor
import time
import math

# ================== Core Configuration ==================
# 1. Red threshold (replace with your calibrated values!)
RED_THRESHOLD = (10, 80, 15, 127, -128, 127)
# 2. Detection thresholds
PIXELS_THRESHOLD = 50
AREA_THRESHOLD = 50
# 3. Image flip settings
FLIP_VERTICAL = True
FLIP_HORIZONTAL = False
# 4. Trajectory configuration
TRAJ_MAX_POINTS = 50    # Maximum number of trajectory points
TRAJ_LINE_COLOR = (0, 255, 0)  # Trajectory line color (green)
TRAJ_CROSS_COLOR = (0, 255, 0) # Center point cross color (green)
TRAJ_CROSS_SIZE = 10     # Cross size
# 5. Visual trigger configuration
BALL_LOST_TIMEOUT = 3.0  # Clear trajectory after ball disappears for 3 seconds
ball_lost_start_time = 0 # Start time of ball disappearance

# ================== Global State Variables ==================
trajectory = []         # Trajectory point cache [(cx1, cy1), (cx2, cy2), ...]
is_tracking = False     # Whether tracking red ball
clock = time.clock()

# ================== Initialization ==================
def init_camera():
    """Initialize camera"""
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)       # 320x240 resolution
    sensor.set_windowing((320, 160))        # Wide window (your preferred style)
    sensor.skip_frames(time=3000)
    # Fix color recognition parameters
    sensor.set_auto_gain(False)
    sensor.set_auto_whitebal(False)
    # Image flip
    sensor.set_vflip(FLIP_VERTICAL)
    sensor.set_hmirror(FLIP_HORIZONTAL)
    print("Initialization complete! Point the red ball at the camera to start tracking...")

def draw_full_trajectory(img):
    """Draw complete trajectory: connecting lines + last center point cross"""
    # 1. Draw trajectory connecting lines
    if len(trajectory) >= 2:
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i-1]
            x2, y2 = trajectory[i]
            img.draw_line((x1, y1, x2, y2), color=TRAJ_LINE_COLOR, thickness=2)
    # 2. Draw last center point cross
    if len(trajectory) > 0:
        last_x, last_y = trajectory[-1]
        img.draw_cross(last_x, last_y, color=TRAJ_CROSS_COLOR, size=TRAJ_CROSS_SIZE)

def clear_trajectory():
    """Clear trajectory (called after red ball disappears for 3 seconds)"""
    global trajectory, is_tracking, ball_lost_start_time
    trajectory = []
    is_tracking = False
    ball_lost_start_time = 0
    print("🛑 Red ball disappeared for 3 seconds, clearing trajectory!")

# ================== Main Loop (Core Visual Trigger Logic) ==================
init_camera()

while True:
    clock.tick()
    img = sensor.snapshot()

    # 1. Detect red ball
    blobs = img.find_blobs(
        [RED_THRESHOLD],
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=True
    )

    # 2. Red ball detected → start/continue tracking, draw trajectory
    if blobs:
        largest_blob = max(blobs, key=lambda b: b.area())
        cx, cy = largest_blob.cx(), largest_blob.cy()

        # Mark current center point
        img.draw_cross(cx, cy, color=TRAJ_CROSS_COLOR, size=TRAJ_CROSS_SIZE)

        # Start tracking (prompt when first detecting red ball)
        if not is_tracking:
            is_tracking = True
            print("✅ Red ball detected, starting trajectory drawing!")

        # Add trajectory point, control maximum points
        trajectory.append((cx, cy))
        if len(trajectory) > TRAJ_MAX_POINTS:
            trajectory.pop(0)

        # Draw existing trajectory
        draw_full_trajectory(img)

        # Reset disappearance timer
        ball_lost_start_time = 0
    else:
        # 3. No red ball detected → check timer to clear trajectory
        if is_tracking:
            # Still draw existing trajectory (briefly retain after ball disappears)
            draw_full_trajectory(img)

            # Start timing
            if ball_lost_start_time == 0:
                ball_lost_start_time = time.time()
            # No ball detected for more than 3 seconds → clear trajectory
            elif (time.time() - ball_lost_start_time) >= BALL_LOST_TIMEOUT:
                clear_trajectory()

    # Output debug information
    status = "Tracking" if is_tracking else "Waiting for red ball"
    print(f"FPS: {clock.fps():.1f} fps | Status: {status} | Trajectory points: {len(trajectory)}")