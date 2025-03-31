import cv2
import os

def extract_frames_from_videos(input_folder='2025-03-19', output_folder='frames', interval=300):
    """
    Extracts frames from all videos in the input_folder every `interval` seconds.
    Saves frames in separate folders named after each video file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue  # Skip non-video files
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_file}")
            continue
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_interval = fps * interval  # Number of frames to skip
        
        video_name = os.path.splitext(video_file)[0]  # Get video name without extension
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        frame_count = 0
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:  # Capture every `interval` seconds
                frame_filename = os.path.join(video_output_folder, f"frame_{frame_index:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_index += 1
                print(f"Saved: {frame_filename}")
            
            frame_count += 1
        
        cap.release()
    print("Frame extraction complete.")

if __name__ == "__main__":
    extract_frames_from_videos()
