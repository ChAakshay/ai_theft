import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image # Pillow library for image handling, often used with OpenCV/Streamlit
import time
import os

# --- Configuration ---
MODEL_PATH = 'yolov5s.pt'  # Path to your YOLOv5s weights file. Will be downloaded if not present.
CONFIDENCE_THRESHOLD = 0.25  # Initial confidence threshold for object detection
IOU_THRESHOLD = 0.45       # Initial IoU threshold for Non-Maximum Suppression (NMS)

# COCO Class Names (for object detection)
# This list maps class IDs from the YOLOv5 model (trained on COCO) to human-readable names.
# Ensure this list matches the classes your YOLOv5 model was trained on.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# IDs for suspicious objects (backpack, handbag) based on COCO_CLASSES list
# These are the objects the system will specifically look for in proximity to persons.
SUSPICIOUS_OBJECT_IDS = [
    COCO_CLASSES.index('backpack'),
    COCO_CLASSES.index('handbag')
]

# --- Helper Functions ---

@st.cache_resource # Caches the model loading to prevent reloading on every Streamlit rerun
def load_model():
    """
    Loads the YOLOv5 model from Ultralytics hub.
    This function is decorated with `@st.cache_resource` to ensure the model is loaded
    only once, even when Streamlit reruns the script due to user interactions.
    """
    try:
        # Attempts to load yolov5s.pt. If not found locally, it will download it.
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        st.stop() # Stops the Streamlit app execution if model loading fails

def check_proximity(person_bbox, suspicious_object_bbox, threshold_pixels=50):
    """
    Determines if a suspicious object is "near" a person based on bounding box centers.
    
    Args:
        person_bbox (tuple): Bounding box coordinates of the person (x_min, y_min, x_max, y_max).
        suspicious_object_bbox (tuple): Bounding box coordinates of the suspicious object.
        threshold_pixels (int): The maximum distance in pixels for objects to be considered "near".

    Returns:
        bool: True if the objects are within the threshold distance, False otherwise.
    """
    px_min, py_min, px_max, py_max = person_bbox
    sx_min, sy_min, sx_max, sy_max = suspicious_object_bbox

    # Calculate center points of the bounding boxes
    p_center_x = (px_min + px_max) / 2
    p_center_y = (py_min + py_max) / 2
    s_center_x = (sx_min + sx_max) / 2
    s_center_y = (sy_min + sy_max) / 2

    # Calculate Euclidean distance between the centers
    distance = np.sqrt((p_center_x - s_center_x)**2 + (p_center_y - s_center_y)**2)

    # Return True if the distance is less than the specified threshold
    return distance < threshold_pixels

# --- Streamlit App Layout and Logic ---

# Set Streamlit page configuration
st.set_page_config(
    page_title="Intelligent Video Surveillance",
    layout="wide",  # Use wide layout to maximize content area
    initial_sidebar_state="expanded" # Sidebar is expanded by default
)

st.title("Intelligent Video Surveillance System")
st.subheader("Real-time Suspicious Activity Detection with YOLOv5")

st.markdown("""
This system detects and flags suspicious activities (like theft or unauthorized bag handling)
in real-time video streams using YOLOv5 object detection.
""")

# Sidebar for user controls and configuration
st.sidebar.header("Configuration")
source_option = st.sidebar.radio(
    "Select Video Source:",
    ("Webcam", "Upload Video File"),
    help="Choose whether to use your computer's webcam or upload a video file."
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.0,
    value=CONFIDENCE_THRESHOLD,
    step=0.01,
    help="Adjust the minimum confidence score for an object detection to be considered valid. Higher values reduce false positives."
)

iou = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.01,
    max_value=1.0,
    value=IOU_THRESHOLD,
    step=0.01,
    help="Intersection over Union threshold for Non-Maximum Suppression. Higher values allow more overlapping bounding boxes."
)

suspicious_proximity_threshold = st.sidebar.slider(
    "Suspicious Object Proximity Threshold (pixels)",
    min_value=10,
    max_value=200,
    value=50,
    step=5,
    help="The maximum distance (in pixels) between the center of a person's bounding box and a suspicious object's bounding box to be flagged."
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Your Name/Team Name for enhanced security monitoring.")

# Load the YOLOv5 model using the cached function
model = load_model()
# Apply the user-defined confidence and IoU thresholds to the model
model.conf = confidence
model.iou = iou

# Main content area for video feed and activity logs
st.subheader("Live Feed / Video Analysis")
st.write("Processing frames in real-time...")

# Placeholder for the video feed. This allows the video to update in place.
video_placeholder = st.empty()

# Placeholder for suspicious activity logs. This allows the logs to update dynamically.
st.subheader("Suspicious Activity Log")
log_container = st.empty()
suspicious_activities = [] # List to store details of detected suspicious events
frame_counter = 0 # Counter to keep track of processed frames

# Create a directory to save screenshots of suspicious frames
SUSPICIOUS_FRAMES_DIR = "suspicious_frames"
os.makedirs(SUSPICIOUS_FRAMES_DIR, exist_ok=True) # Creates the directory if it doesn't exist

cap = None # Initialize video capture object to None

# Handle video source selection
if source_option == "Webcam":
    st.info("Starting webcam feed... Please allow camera access if prompted by your browser.")
    cap = cv2.VideoCapture(0) # 0 typically refers to the default webcam

elif source_option == "Upload Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary path so OpenCV can read it
        temp_video_path = "temp_uploaded_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture(temp_video_path)
        st.info(f"Uploaded video '{uploaded_file.name}' is being processed.")
    else:
        st.warning("Please upload a video file to begin analysis.")
        # If no file is uploaded, 'cap' remains None, preventing the processing loop from starting.

# Start video processing if a video source is successfully opened
if cap is not None:
    if not cap.isOpened():
        st.error("Error: Could not open video source. Please check camera connection or ensure the uploaded file is valid.")
    else:
        st.write("Monitoring live feed... (To stop, close this browser tab or stop the Streamlit application in your terminal.)")
        
        # Main loop for real-time frame processing
        while cap.isOpened():
            ret, frame = cap.read() # Read a single frame from the video stream
            if not ret:
                st.warning("End of video stream or failed to read frame. If using a file, it will loop.")
                # If it's an uploaded video file, loop back to the beginning for continuous demo
                if source_option == "Upload Video File":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to the first frame
                    continue # Continue to the next iteration of the loop
                else:
                    break # If webcam fails, break the loop

            frame_counter += 1 # Increment frame counter

            # Convert the frame from BGR (OpenCV's default) to RGB (YOLOv5's expected format)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection inference on the current frame
            # The 'results' object contains all detected objects with their bounding boxes, confidence, and class.
            results = model(img_rgb)

            # Lists to hold bounding box information for persons and suspicious objects
            detected_persons = []
            suspicious_objects_in_frame = []
            
            # Iterate through each detected object in the results
            # results.xyxy[0] provides detections as [x1, y1, x2, y2, confidence, class_id]
            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                label = COCO_CLASSES[int(cls)] # Get the class name using the class ID

                # Draw a green bounding box and label for all detected objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # BGR color (Green)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Categorize detections: store persons and potential suspicious objects
                if label == 'person':
                    detected_persons.append((x1, y1, x2, y2))
                elif int(cls) in SUSPICIOUS_OBJECT_IDS:
                    suspicious_objects_in_frame.append((x1, y1, x2, y2, int(cls))) # Store class ID too

            # --- Custom Filtering Logic: Check for suspicious activity ---
            is_suspicious_frame = False
            # Only proceed if both persons AND suspicious objects are present in the frame
            if detected_persons and suspicious_objects_in_frame:
                for person_bbox in detected_persons:
                    for obj_bbox_with_cls in suspicious_objects_in_frame:
                        obj_bbox = obj_bbox_with_cls[:4] # Extract bbox from (x1,y1,x2,y2,cls_id)
                        obj_cls_id = obj_bbox_with_cls[4]

                        # Check if the suspicious object is in close proximity to the person
                        if check_proximity(person_bbox, obj_bbox, suspicious_proximity_threshold):
                            is_suspicious_frame = True
                            
                            # --- Auto Flagging System: Highlight and Save ---
                            # Draw red bounding boxes for the person and suspicious object involved
                            cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 0, 255), 3) # Red for person
                            cv2.rectangle(frame, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), (0, 0, 255), 3) # Red for suspicious object
                            # Add "SUSPICIOUS!" text overlay
                            cv2.putText(frame, "SUSPICIOUS ACTIVITY DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            
                            # Generate a unique filename using timestamp and frame number
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            screenshot_filename = f"{SUSPICIOUS_FRAMES_DIR}/suspicious_frame_{timestamp}_frame_{frame_counter}.jpg"
                            cv2.imwrite(screenshot_filename, frame) # Save the flagged frame as an image

                            # Create a log entry for this suspicious event
                            log_entry = {
                                "timestamp": timestamp,
                                "frame": frame_counter,
                                "description": f"Suspicious object ({COCO_CLASSES[obj_cls_id]}) detected near a person.",
                                "screenshot": screenshot_filename # Path to the saved screenshot
                            }
                            # Add the log entry to the list if it's not a duplicate for the current frame
                            # This prevents multiple identical logs if multiple suspicious pairs are found in one frame
                            if not any(entry['frame'] == log_entry['frame'] for entry in suspicious_activities):
                                suspicious_activities.append(log_entry)
                            
                            break # Break from inner loop (obj_bbox) once one suspicious pair is found for this person
                    if is_suspicious_frame:
                        break # Break from outer loop (person_bbox) once suspicious activity is confirmed for this frame

            # Display the current processed frame in the Streamlit UI
            # 'channels="BGR"' tells Streamlit that the image is in BGR format (OpenCV's default)
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Update the suspicious activity log display in the Streamlit UI
            with log_container.container():
                st.write("---") # Separator
                if not suspicious_activities:
                    st.info("No suspicious activities detected yet. Monitoring...")
                else:
                    st.write(f"Detected {len(suspicious_activities)} suspicious activities:")
                    # Display logs in reverse chronological order (most recent first)
                    for i, activity in enumerate(reversed(suspicious_activities)):
                        st.markdown(f"**Activity {len(suspicious_activities) - i}:**") # Numbering from 1
                        st.write(f"  - **Timestamp:** {activity['timestamp']}")
                        st.write(f"  - **Frame:** {activity['frame']}")
                        st.write(f"  - **Description:** {activity['description']}")
                        # Display the saved screenshot for the log entry
                        st.image(activity['screenshot'], caption=f"Suspicious Frame {activity['frame']}", width=200)

            # Introduce a small delay to control the frame rate and reduce CPU usage
            # Adjust this value (in seconds) based on desired performance and system capabilities
            time.sleep(0.01)

        # After the loop finishes (e.g., video ends, webcam disconnected), release resources
        cap.release() # Release the video capture object
        cv2.destroyAllWindows() # Close any OpenCV windows (though Streamlit handles display)
        st.success("Video analysis finished. Restart the app to analyze again.")

else:
    # Message displayed if no video source is selected or opened successfully at the start
    st.info("Please select a video source (Webcam or Upload Video File) to begin the analysis.")

