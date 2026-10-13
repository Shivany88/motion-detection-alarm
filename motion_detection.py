import cv2
import pygame
import numpy as np
import time
import datetime
import os

# Folder where this script is saved (works on any PC)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the pygame mixer
pygame.mixer.init()

# Initialize the video capture object
cam = cv2.VideoCapture(0)

# Warm up the camera: auto-exposure/brightness needs time to settle,
# otherwise the first frame looks very different and triggers a false alarm
for _ in range(30):
    cam.read()
time.sleep(2)

# Capture the first frame (reference frame, keep the scene still now)
ret, first_frame = cam.read()
if not ret:
    print("Error: Could not read from camera.")
    cam.release()
    exit()

# Convert the first frame to grayscale and apply Gaussian blur
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

alarm_triggered = False
MOTION_SOUND_PATH = os.path.join(BASE_DIR, "motion_alarm.wav")

RECORDING = False
out = None
VIDEO_SAVE_PATH = os.path.join(BASE_DIR, "videos")
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)  # create videos folder if missing

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the current frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Calculate the difference between the current frame and the first frame
    delta_frame = cv2.absdiff(first_frame, gray_frame)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Count white pixels in the threshold frame
    white_pixel_count = np.sum(thresh_frame == 255)

    if white_pixel_count > 1000:
        if not alarm_triggered:
            pygame.mixer.music.load(MOTION_SOUND_PATH)
            pygame.mixer.music.play(-1)  # Loop the sound continuously while motion is detected
            alarm_triggered = True

        if not RECORDING:
            RECORDING = True
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(VIDEO_SAVE_PATH, f"motion_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG'
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))  # 20 fps
            print(f"Recording started: {filename}")

        if RECORDING and out is not None:
            out.write(frame)
    else:
        # No motion: stop the alarm so it can ring again on next motion
        if alarm_triggered:
            pygame.mixer.music.stop()
            alarm_triggered = False

        if RECORDING:
            RECORDING = False
            if out is not None:
                out.release()
                out = None
            print("Recording stopped.")

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)

    # Break the loop on 'Esc' key press and stop the alarm
    if cv2.waitKey(1) == 27:
        pygame.mixer.music.stop()
        if RECORDING and out is not None:
            out.release()
        break

    # Introduce a small delay to reduce CPU usage
    time.sleep(0.1)

# Release the capture object and close windows
cam.release()
cv2.destroyAllWindows()