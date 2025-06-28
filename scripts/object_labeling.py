import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import pickle
import argparse
from multilingual_clip import pt_multilingual_clip
import transformers
import torchvision
from torchvision import transforms


parser = argparse.ArgumentParser(description="Extract object-level CLIP features for one video.")
parser.add_argument("--video_frames_root", required=True, help="Path to folder containing video frame folders")
parser.add_argument("--video_id", required=True, help="Video ID (folder name) to process")
args = parser.parse_args()

video_frames_root = args.video_frames_root
target_video_id = args.video_id


csv_path = "msvd_dataset_amharic_caption.csv"
output_path = "train_object_clip_q_label_am.pkl"

device = "cpu"
print(f"Using device: {device}")

clip_model, preprocess = clip.load("ViT-B/32", device=device)

fastrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
fastrcnn_model.eval()
det_transform = transforms.Compose([transforms.ToTensor()])

model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
model_mul = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv(csv_path)
qa_dict = {row['video_id'].split(',')[0]: row['caption'] for _, row in df.iterrows()}

if os.path.exists(output_path):
    with open(output_path, "rb") as f:
        results = pickle.load(f)
else:
    results = {}

video_folders = os.listdir(video_frames_root)
if target_video_id not in video_folders:
    print(f"Video '{target_video_id}' not found in {video_frames_root}")
    exit()

vid = target_video_id.strip()
if vid not in qa_dict or vid in results:
    print(f"Skipping '{vid}' (already processed or missing caption)")
    exit()

words = [w.strip() for w in qa_dict[vid].split() if w.strip()]
with torch.no_grad():
    word_feats = model_mul.forward(words, tokenizer).to(device)
    word_feats /= word_feats.norm(dim=-1, keepdim=True)

folder_path = os.path.join(video_frames_root, vid)
if not os.path.exists(folder_path):
    print(f"Frame folder not found: {folder_path}")
    exit()

vdata = []
for idx, frame in enumerate(sorted(os.listdir(folder_path))):
    img_path = os.path.join(folder_path, frame)
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    inp = det_transform(img).to(device)
    with torch.no_grad():
        pred = fastrcnn_model([inp])[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    boxes = boxes[scores >= 0.5]

    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        crop = img.crop((x1, y1, x2, y2))
        clip_inp = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(clip_inp)
            feat /= feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * feat @ word_feats.T).softmax(dim=-1)
            best = sim.argmax().item()

        vdata.append({
            "frame_id": idx,
            "box": [x1, y1, x2, y2],
            "word": words[best],
            "clip_feature": feat.cpu().numpy(),
            "confidence": sim[0, best].item()
        })

if vdata:
    results[vid] = vdata
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

print(f"Finished processing video: {target_video_id}")
