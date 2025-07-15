# AI-Powered Theft Detection System 🛡️

This project is a real-time intelligent video surveillance tool that detects and flags suspicious activities (like theft or unauthorized bag handling) using YOLOv5 object detection.

## 📹 Project Objective

Detect suspicious activity — like a person interacting with a backpack or handbag — in real-time security footage and alert via saved frames and logs.

---

## 🔍 Features

- Real-time object detection using **YOLOv5**
- Flags suspicious person-bag interactions using **IoU-based proximity**
- Saves suspicious frames with **timestamps + bounding boxes**
- Adjustable thresholds for **confidence and IoU**
- Easily extendable with face detection or gesture tracking

---

## ⚙️ Tech Stack

- Python
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- OpenCV
- Google Colab / Local

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/ai-theft-detection-yolov5.git
cd ai-theft-detection-yolov5
pip install -r requirements.txt
