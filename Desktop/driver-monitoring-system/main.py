import cv2
import dlib
import numpy as np
import time
import winsound
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import imutils

# 🔥 NEW: Flask integration
from app import app, update_status
import threading

print("Program started...")

# ================= FLASK THREAD =================
def run_flask():
    app.run(debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# 🔊 Alarm
def trigger_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)

# 😮 Mouth Aspect Ratio
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# ================= CNN MODEL =================
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load model
device = torch.device("cpu")
eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load("models/eye_cnn.pth", map_location=device))
eye_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((24, 24)),
    transforms.ToTensor()
])

# ================= SETTINGS =================
MAR_THRESHOLD = 0.7
CONSEC_FRAMES = 15
COUNTER = 0

# Load dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(mStart, mEnd) = (48, 68)

# Camera start
vs = VideoStream(src=0).start()
time.sleep(1.0)

# ================= LOOP =================
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # ================= HEAD POSE =================
        image_points = np.array([
            shape[30], shape[8], shape[36],
            shape[45], shape[48], shape[54]
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1]

        # ================= CNN EYE =================
        leftEye = shape[lStart:lEnd]
        lx, ly = leftEye[:, 0], leftEye[:, 1]

        x1, y1 = int(min(lx)), int(min(ly))
        x2, y2 = int(max(lx)), int(max(ly))

        pred = 1

        if x2 > x1 and y2 > y1:
            eye_img = gray[y1:y2, x1:x2]

            if eye_img.size != 0:
                eye_img = cv2.resize(eye_img, (24, 24))
                eye_tensor = transform(eye_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = eye_model(eye_tensor)
                    pred = torch.argmax(output, dim=1).item()

        # ================= MOUTH =================
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # ================= DECISION =================
        risk = 0

        if pred == 0:
            COUNTER += 1
        else:
            COUNTER = 0

        if COUNTER >= CONSEC_FRAMES:
            risk += 2

        if mar > MAR_THRESHOLD:
            risk += 1

        if abs(yaw) > 20:
            cv2.putText(frame, "LOOKING AWAY!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            risk += 1

        # ================= ALERT =================
        if risk >= 2:
            update_status("DROWSY")   # 🔥 NEW
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            trigger_alarm()
        else:
            update_status("SAFE")     # 🔥 NEW

        # ================= DEBUG =================
        cv2.putText(frame, f"Eye: {'Closed' if pred == 0 else 'Open'}",
                    (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"MAR: {mar:.2f}",
                    (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()