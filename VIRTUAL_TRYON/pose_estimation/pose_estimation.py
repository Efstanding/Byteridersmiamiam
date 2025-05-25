import cv2
import numpy as np
import mediapipe as mp

# Load image
image = cv2.imread("pose.jpeg")
if image is None:
    raise ValueError("Image not found. Check path.")

h, w, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(image_rgb)

# Create black canvas
pose_image = np.zeros((h, w, 3), dtype=np.uint8)

# Draw skeleton
if results.pose_landmarks:
    for connection in mp_pose.POSE_CONNECTIONS:
        start = results.pose_landmarks.landmark[connection[0]]
        end = results.pose_landmarks.landmark[connection[1]]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(pose_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for landmark in results.pose_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(pose_image, (x, y), 4, (255, 255, 255), -1)

# Save pose-conditioning image
cv2.imwrite("pose_conditioning.png", pose_image)
print("Saved as pose_conditioning.png")
