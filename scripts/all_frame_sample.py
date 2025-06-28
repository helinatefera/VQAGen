import cv2
import argparse
import os


def analyze_video(video_path, output_folder="../../ video_frames"):
    # Create output folder if it doesn't exist
    output_dir = os.path.join(os.path.dirname(video_path), output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration_seconds = total_frames / fps if fps > 0 else 0  # Duration in seconds
    
    print(f"Video FPS: {fps} (frames per second)")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    
    # Loop through all frames and save them
    frame_count = 0
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:  # Break if no more frames
            break
        frame_count += 1
        # Save frame as JPEG in the output folder
        frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count} to {frame_filename}")
    
    print(f"Processed Frames: {frame_count}")
    print(f"Frames saved to: {output_dir}")
    
    # Release the video capture object
    cap.release()
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze video frames and save them to a folder named after the video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    
    # Verify video path exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist.")
    else:
        analyze_video(args.video_path)