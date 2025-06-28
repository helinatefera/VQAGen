import os
import cv2
import numpy as np
import torch
import clip
from multilingual_clip import pt_multilingual_clip
from PIL import Image
from sklearn.cluster import KMeans
import argparse
import transformers

device = "cpu"
# Load openai/CLIP for image embeddings
clip_model, preprocess = clip.load("ViT-B/32", device=device)
# Load multilingual-clip for text embeddings
text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained("M-CLIP/XLM-Roberta-Large-Vit-B-32")

def load_annotations(annotation_file):
    """
    Load an annotations file where each line contains a video_id and a caption, separated by a comma.
    """
    annotations = {}
    try:
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Replace tabs with spaces and collapse multiple spaces
                line = ' '.join(line.split())
                # Split on comma instead of space
                parts = line.split(',', 1)
                if len(parts) < 2:
                    print(f"Warning: Skipping invalid line {line_number} in {annotation_file}: '{line}'")
                    continue
                video_id, caption = parts
                video_id = video_id.strip()
                caption = caption.strip()
                if not caption:
                    print(f"Warning: Empty caption for video_id {video_id} at line {line_number}")
                    continue
                annotations[video_id] = annotations.get(video_id, []) + [caption]
        print(f"Loaded {len(annotations)} video IDs from {annotation_file}")
        if "JM4913Fe-ic_4_15" not in annotations:
            print(f"Debug: Video ID 'JM4913Fe-ic_4_15' not found in annotations")
            print(f"Available video IDs: {list(annotations.keys())[:5]}...")
    except Exception as e:
        print(f"Error reading annotations file {annotation_file}: {e}")
        return {}
    return annotations

def select_caption(captions, max_words=50):
    """
    Select a caption suitable for M-CLIP, truncating if necessary.
    """
    for caption in captions:
        try:
            _ = tokenizer([caption], padding=True, return_tensors="pt")
            return caption
        except Exception:
            continue
    shortest = min(captions, key=lambda x: len(x.split()))
    truncated = " ".join(shortest.split()[:max_words])
    print(f"Warning: All captions too long, truncated to: {truncated}")
    return truncated

def load_frames_from_folder(folder_path):
    """
    Load all image frames from the specified folder.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} does not exist.")
        return []
    if not os.path.isdir(folder_path):
        print(f"Error: Path {folder_path} is not a directory.")
        return []
    frame_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Found {len(frame_files)} frame files in {folder_path}")
    frames = []
    for file in frame_files:
        try:
            frame = cv2.imread(file)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Failed to load frame {file}")
        except Exception as e:
            print(f"Error loading frame {file}: {e}")
    return frames

def compute_frame_features(frames):
    """
    Compute CLIP features for each frame using openai/CLIP.
    """
    features = []
    for frame in frames:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = clip_model.encode_image(image_input)
            features.append(feature.cpu().numpy().squeeze())
        except Exception as e:
            print(f"Error computing features for frame: {e}")
    return np.array(features)

def compute_caption_embedding(caption):
    """
    Compute M-CLIP embedding for the caption.
    """
    try:
        with torch.no_grad():
            text_features = text_model.forward([caption], tokenizer)
        return text_features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error computing caption embedding: {e}")
        return None

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    try:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return -1

def select_representative_frames(frames, caption, num_selected=16):
    """
    Select representative frames using KMeans clustering and caption similarity.
    """
    if len(frames) == 0:
        print("No frames available to process.")
        return []
    frame_features = compute_frame_features(frames)
    if len(frame_features) == 0:
        print("No valid frame features computed.")
        return []
    caption_embedding = compute_caption_embedding(caption)
    if caption_embedding is None:
        print("Failed to compute caption embedding.")
        return []
    
    if len(frames) < num_selected:
        print(f"Fewer frames ({len(frames)}) than requested ({num_selected}). Returning all frames.")
        return frames
    
    try:
        kmeans = KMeans(n_clusters=num_selected, random_state=42, n_init=10)
        labels = kmeans.fit_predict(frame_features)
        selected_frames = []
        for i in range(num_selected):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                continue
            sims = [cosine_similarity(frame_features[j], caption_embedding) for j in idxs]
            best_idx = idxs[np.argmax(sims)]
            selected_frames.append(frames[best_idx])
        return selected_frames
    except Exception as e:
        print(f"Error in KMeans clustering: {e}")
        return []

def process_video(video_id, video_frames_folder, annotation_file, output_base_folder):
    """
    Process a single video's frames: load frames, select representative frames, and save them.
    """
    # Construct and validate the video's frame folder path
    video_folder_path = os.path.join(video_frames_folder, video_id)
    print(f"Checking video folder: {video_folder_path}")
    if not os.path.exists(video_folder_path):
        print(f"Error: Video folder {video_folder_path} does not exist.")
        return
    if not os.path.isdir(video_folder_path):
        print(f"Error: {video_folder_path} is not a directory.")
        return
    
    # Load annotations and get captions for the video
    annotations = load_annotations(annotation_file)
    captions = annotations.get(video_id, None)
    if captions is None:
        print(f"Error: Caption not found for video id {video_id} in {annotation_file}. Skipping.")
        return
    
    # Select one caption
    caption = select_caption(captions, max_words=50)
    print(f"Selected caption: {caption}")
    
    # Load frames
    frames = load_frames_from_folder(video_folder_path)
    if len(frames) == 0:
        print(f"No frames found for video id {video_id} in {video_folder_path}")
        return
    
    # Select representative frames
    selected_frames = select_representative_frames(frames, caption, num_selected=16)
    if len(selected_frames) == 0:
        print(f"Processed {video_id}: saved 0 selected frames.")
        return
    
    # Create output folder named after video_id
    video_output_folder = os.path.join(output_base_folder, video_id)
    try:
        os.makedirs(video_output_folder, exist_ok=True)
        print(f"Created output folder: {video_output_folder}")
    except Exception as e:
        print(f"Error creating output folder {video_output_folder}: {e}")
        return
    
    # Save selected frames
    for idx, frame in enumerate(selected_frames):
        frame_filename = os.path.join(video_output_folder, f"selected_frame_{idx:03d}.jpg")
        try:
            success = cv2.imwrite(frame_filename, frame)
            if success:
                print(f"Saved frame {idx} to {frame_filename}")
            else:
                print(f"Failed to save frame {idx} to {frame_filename}")
        except Exception as e:
            print(f"Error saving frame {idx} to {frame_filename}: {e}")
    
    print(f"Processed {video_id}: saved {len(selected_frames)} selected frames in {video_output_folder}.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a single video's frames to select representative frames.")
    parser.add_argument("--video_id", type=str, required=True, help="Video ID (name of the folder containing frames)")
    parser.add_argument("--video_frames_folder", type=str, default="/home/amerti/Documents/video_frames",
                        help="Base folder containing video frame folders")
    parser.add_argument("--annotation_file", type=str, default="/home/amerti/Documents/MSC/amahric_dataset/MSDV_amharic_caption.txt",
                        help="Path to the annotations file")
    parser.add_argument("--output_base_folder", type=str, default="/home/amerti/Documents/16_best",
                        help="Base folder to save selected frames")
    args = parser.parse_args()
    
    # Verify video frames folder exists
    if not os.path.exists(args.video_frames_folder):
        print(f"Error: Video frames base folder {args.video_frames_folder} does not exist.")
        exit(1)
    
    # Verify annotations file exists
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotations file {args.annotation_file} does not exist.")
        exit(1)
    
    # Create output base folder if it doesn't exist
    if not os.path.exists(args.output_base_folder):
        try:
            os.makedirs(args.output_base_folder)
            print(f"Created base output folder: {args.output_base_folder}")
        except Exception as e:
            print(f"Error creating base output folder {args.output_base_folder}: {e}")
            exit(1)
    
    # Process the specified video
    process_video(args.video_id, args.video_frames_folder, args.annotation_file, args.output_base_folder)