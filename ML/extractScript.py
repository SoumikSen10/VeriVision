import os
import cv2
import shutil
from lstmScript import process_frame_with_lstm  # Import LSTM model processing function
from resnextScript import process_frame_with_resnext  # Import ResNeXt model processing function
from capsnetScript import process_frame_with_capsule_net  # Import CapsuleNet model processing function

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
            frame_filename = os.path.join(frames_dir_path, f"frame_{extracted_frame_count}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Captured frame: {frame_filename}")
            
            # Redirect the frame to all three models' processing functions
            lstm_result = process_frame_with_lstm(frame)
            resnext_result = process_frame_with_resnext(frame)
            capsule_result = process_frame_with_capsule_net(frame)
            
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
    print("Removing frames storage directory")
    remove_frames_dir(frames_dir_path)

# Specify the video file and the frame skip interval (1 frame per second)
video_file = './ML/testvideo.mp4'# Replace with your actual video path
frame_skip = 30  # Set frame skip interval based on live stream requirements (e.g., 30 for 1 frame per second in a 30 fps video)

process_video(video_file, frame_skip)
