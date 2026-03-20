import cv2
import numpy as np
import matplotlib.pyplot as plt
from orange_detector import OrangeDetector
from kalmanfilter import KalmanFilter

# ========== 1. Initialize trajectory storage and plotting environment ==========
detect_pts = []  # List of red detection points [cx, cy]
predict_pts = []  # List of blue prediction points [px, py]

# Matplotlib interactive plot settings
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('X Pixel')
ax.set_ylabel('Y Pixel')
ax.set_title('Trajectory of Detection (Red) and Prediction (Blue)')
ax.grid(True)
ax.invert_yaxis()  # Adapt to image coordinate system

# ========== 2. Video reading + saving initialization ==========
# Input video path
input_video_path = "ball05.mp4"
# Output video path (save video with predictions)
output_video_path = "ball05_with_prediction.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError(f"Cannot open video file: {input_video_path}")

# Get basic video parameters (for saving video)
fps = cap.get(cv2.CAP_PROP_FPS)  # Original video frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (MP4)

# Initialize video writer (save processed video)
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,  # Saved video frame rate matches original
    (width, height)
)

# ========== 3. Detector + filter initialization ==========
od = OrangeDetector()  # Source: https://pysource.com/free-sample-source-codes/ (9. Kalman filter, predict the trajectory of an Object | Source Code)
kf = KalmanFilter()    # Source: https://pysource.com/free-sample-source-codes/ (9. Kalman filter, predict the trajectory of an Object | Source Code)

# Video playback speed adjustment (optional)
speed_factor = 0.5  # 0.5 = 50% of original speed, smaller = slower
base_delay = 100 if fps == 0 else int(1000/fps)
delay = int(base_delay / speed_factor)

# ========== 4. Main loop: video processing + trajectory drawing + saving ==========
while True:
    ret, frame = cap.read()
    if ret is False:
        break

    # Object detection
    orange_bbox = od.detect(frame)
    x, y, x2, y2 = orange_bbox
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    # Kalman filter prediction
    predicted = kf.predict(cx, cy)
    px, py = predicted[0], predicted[1]

    # Store valid trajectory points
    if cx > 0 and cy > 0:
        detect_pts.append([cx, cy])
        predict_pts.append([px, py])

    # Draw detection/prediction points on video frame
    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)    # Red detection point
    cv2.circle(frame, (px, py), 20, (255, 0, 0), 4)    # Blue prediction point

    # ========== Core: Save current frame to output video ==========
    video_writer.write(frame)

    # Real-time video frame display
    cv2.imshow("Video Frame (Press ESC to quit)", frame)

    # Update Matplotlib trajectory plot
    if len(detect_pts) > 1:
        ax.clear()
        ax.set_xlabel('X Pixel')
        ax.set_ylabel('Y Pixel')
        ax.set_title('Trajectory of Detection (Red) and Prediction (Blue)')
        ax.grid(True)
        ax.invert_yaxis()

        detect_arr = np.array(detect_pts)
        predict_arr = np.array(predict_pts)

        ax.plot(detect_arr[:, 0], detect_arr[:, 1], 'r-',
                linewidth=2, label='Detection (Red)')
        ax.plot(predict_arr[:, 0], predict_arr[:, 1], 'b-',
                linewidth=2, label='Prediction (Blue)')
        ax.scatter(cx, cy, color='red', s=50, zorder=5)
        ax.scatter(px, py, color='blue', s=50, zorder=5)

        ax.legend()
        plt.draw()
        plt.pause(0.001)

    # Control playback speed and exit
    key = cv2.waitKey(delay)
    if key == 27:  # ESC key to exit
        break

# ========== 5. Resource release + final processing ==========
# Must release video writer, otherwise the saved video will be corrupted
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Close interactive plotting, keep trajectory plot
plt.ioff()
plt.show()

print(f"Video with predicted trajectory saved to: {output_video_path}")