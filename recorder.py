import pyautogui
import cv2
import numpy as np
import os
import datetime
import time

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)
    face_img_resized = None
    for (x, y, w, h) in faces:
        x = max(0, x - 20)
        y = max(0, y - 20)
        w = min(w + 40, img.shape[1] - x)
        h = min(h + 40, img.shape[0] - y)
        face_img = img[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img, face_img_resized, face_count

# Specify folders for saving the video and frames
video_save_dir = "C://Users//DELL//OneDrive//Documents//video"
frames_save_dir = "C://Users//DELL//OneDrive//Documents//frames"
os.makedirs(video_save_dir, exist_ok=True)
os.makedirs(frames_save_dir, exist_ok=True)

# Video parameters
resolution = (1920, 1080)
codec = cv2.VideoWriter_fourcc(*"XVID")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(video_save_dir, f"Recording_{timestamp}.avi")
fps = 60.0
out = cv2.VideoWriter(video_filename, codec, fps, resolution)

# Create an empty window and start recording
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 480, 270)
print("Recording screen... Press 'q' to stop or recording will stop after 3 minutes.")
start_time = time.time()
face_detected = False

while (time.time() - start_time) < 180:  # 3 minutes limit
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_faces, face_img_resized, face_count = detect_faces(frame)
    out.write(frame_with_faces)
    cv2.imshow('Live', frame_with_faces)

    if face_count > 0 and not face_detected:
        cv2.imwrite(os.path.join(frames_save_dir, f'frame_{timestamp}_{int(time.time())}.jpg'), face_img_resized)
        face_detected = True

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
print("Screen recording saved successfully.")
