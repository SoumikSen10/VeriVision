import os
import cv2
import shutil
from lstmScript import process_frame_with_lstm  # Import LSTM model processing function
from resnextScript import process_frame_with_resnext  # Import ResNeXt model processing function
from capsnetScript import process_frame_with_capsule_net  # Import CapsuleNet model processing function

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to create the directory next to the script file
def create_frames_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir_path = os.path.join(script_dir, 'temp_frames')
    
    if not os.path.exists(frames_dir_path):
        os.makedirs(frames_dir_path)
        print(f"Frames directory created at: {frames_dir_path}")
    
    return frames_dir_path

# Function to remove the directory after processing
def remove_frames_dir(frames_dir_path):
    if os.path.exists(frames_dir_path):
        shutil.rmtree(frames_dir_path)
        print(f"Frames directory removed: {frames_dir_path}")

# Function to crop faces from frames with larger bounding box expansion
def detect_and_crop_face(frame, scale_factor=1.4):  # Reduced scale factor to 1.4 for 40% expansion
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        print("No face detected in the current frame.")
        return None
    
    # We will assume the largest face is the main face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])  # Find the largest face by area
    
    # Expand the bounding box by 40%
    x = max(0, x - int(w * (scale_factor - 1) / 2))
    y = max(0, y - int(h * (scale_factor - 1) / 2))
    w = min(frame.shape[1] - x, int(w * scale_factor))
    h = min(frame.shape[0] - y, int(h * scale_factor))
    
    cropped_face = frame[y:y+h, x:x+w]
    
    return cropped_face

# Function to capture frames from video and feed them for feature extraction
# frame_skip is dynamically set based on the live video requirements
def capture_and_process_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frame_count = 0
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    print(f"Starting to capture frames from {video_path}...")

    # Create directory to store frames temporarily and get its path
    frames_dir_path = create_frames_dir()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Capture every 'frame_skip' frames for real-time processing
        if frame_count % frame_skip == 0:
            cropped_face = detect_and_crop_face(frame)
            
            if cropped_face is not None:
                frame_filename = os.path.join(frames_dir_path, f"frame_{extracted_frame_count}.png")
                cv2.imwrite(frame_filename, cropped_face)
                print(f"Captured and cropped face: {frame_filename}")

                # Redirect the cropped face to all three models' processing functions
                lstm_result = process_frame_with_lstm(cropped_face)
                resnext_result = process_frame_with_resnext(cropped_face)
                capsule_result = process_frame_with_capsule_net(cropped_face)

                # Print or log the results
                print(f"LSTM Result: {lstm_result}")
                print(f"ResNeXt Result: {resnext_result}")
                print(f"CapsuleNet Result: {capsule_result}")

                extracted_frame_count += 1
        
        frame_count += 1
    
    cap.release()

    if extracted_frame_count == 0:
        print("No frames were captured.")
        return False

    print(f"Total frames captured and processed: {extracted_frame_count}")
    
    return True, frames_dir_path

# Main function to handle the video processing
def process_video(video_path, frame_skip=30):
    print("Processing video...")
    
    # Step 1: Capture and process frames
    success, frames_dir_path = capture_and_process_frames(video_path, frame_skip)
    if not success:
        print("Frame capture failed. Exiting.")
        return
    
    print("Video processing completed successfully.")
    
    # Step 2: Remove the frames directory after processing
    #print("Removing frames storage directory")
    #remove_frames_dir(frames_dir_path)

# Specify the video file and the frame skip interval (1 frame per second)
video_file = './ML/testvideo.mp4'  # Replace with your actual video path
frame_skip = 30  # Set frame skip interval based on live stream requirements (e.g., 30 for 1 frame per second in a 30 fps video)

process_video(video_file, frame_skip)
