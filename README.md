Intelligent Video Surveillance System
Real-time Suspicious Activity Detection with YOLOv5
This project implements an intelligent video surveillance system capable of detecting and flagging suspicious activities, such as theft or unauthorized bag handling, in real-time video streams. Leveraging the power of deep learning with YOLOv5, this system is designed to enhance early crime detection in sensitive environments like retail stores, banks, and warehouses.

✨ Key Features
Deep Learning with YOLOv5: Utilizes the robust YOLOv5s model (pretrained on the COCO dataset) for accurate and efficient object detection.

Custom Filtering Logic: Implements sophisticated logic to identify and flag suspicious objects (e.g., handbags, backpacks) when they are in close proximity to persons, indicating potential unauthorized handling or abandonment.

Real-Time Frame Analysis: Processes input video streams (from webcam or uploaded files) frame by frame, providing live detection results.

Auto Flagging System: Automatically saves timestamps and screenshots of frames where suspicious activities are detected, creating a detailed log for security review.

Adjustable Sensitivity: Offers configurable confidence and IoU (Intersection over Union) thresholds through an intuitive Streamlit interface, allowing users to fine-tune detection accuracy and sensitivity to specific environments.

🛠️ Tools & Tech Stack
Language: Python

Framework: Ultralytics YOLOv5

Libraries: OpenCV, Streamlit, PyTorch, NumPy

Platform: Designed for local execution with Streamlit, easily adaptable for cloud platforms like Google Colab.

Model: Pretrained YOLOv5s (yolov5s.pt)

🚀 Impact & Use Cases
Crime Prevention: Acts as a proactive security measure, preventing theft in retail stores, banks, and warehouses by alerting security teams in real-time to suspicious events.

Enhanced Security Monitoring: Provides an intelligent layer to traditional surveillance, reducing the need for constant manual monitoring.

Extensibility: Can be further extended with advanced features like facial recognition, re-identification, and complex gesture tracking for broader security analysis.

Smart Surveillance: A valuable component in modern smart surveillance systems and AI-based monitoring solutions, contributing to safer public and private spaces.
