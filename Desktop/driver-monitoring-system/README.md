 # Driver Monitoring System 🚗

## 📌 Overview
A real-time driver drowsiness detection system using Computer Vision and Deep Learning.  
It detects eye closure, yawning, and head movement to determine driver alertness and help prevent accidents.

---

## 🚀 Features
- 👁 CNN-based Eye Detection (Open / Closed)
- 😮 Yawn Detection using Mouth Aspect Ratio (MAR)
- 🧠 Head Pose Detection (Looking Away)
- 🔊 Real-time Alarm System
- 🌐 Flask API for live monitoring (`/status`)

---

## 🛠 Tech Stack
- Python
- OpenCV
- Dlib
- PyTorch
- Flask

---


## 📁 Project Structure
## 📥 Download Required Model

Download the dlib model from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place inside:
models/