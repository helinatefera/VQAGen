import os
import pickle
import torch
import gc
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TimesformerModel
import torch.nn as nn
from tqdm.auto import tqdm
import argparse

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Extract TimeSformer features for one video.")
parser.add_argument("--video_frames_root", required=True, help="Path to folder containing video frame folders")
parser.add_argument("--video_id", required=True, help="Video ID (folder name) to process")
args = parser.parse_args()

video_frames_root = args.video_frames_root
target_video_id = args.video_id

# -------------------------------
# Model and Config
# -------------------------------
class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
    def forward(self, x):
        w = torch.softmax(self.proj(x).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1)

device = "cpu"
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device).eval()
attn = SimpleAttention(model.config.hidden_size).to(device)

# -------------------------------
# Paths and Output Setup
# -------------------------------
out_path = "train_test_video_timesformer_features.pkl"
MAX_FRAMES, CHUNK = 16, 4

features = {}
if os.path.exists(out_path):
    with open(out_path, "rb") as f:
        features = pickle.load(f)

# -------------------------------
# Process One Video
# -------------------------------
vid = target_video_id.strip()
if vid in features:
    print(f"Video '{vid}' already processed.")
    exit()

fldr = os.path.join(video_frames_root, vid)
if not os.path.isdir(fldr):
    print(f"Frame folder '{fldr}' not found.")
    exit()

all_frames = sorted(
    p for p in os.listdir(fldr) if p.lower().endswith((".jpg", ".png"))
)
if len(all_frames) == 0:
    print(f"No frames found in '{fldr}'.")
    exit()

idx = np.linspace(0, len(all_frames) - 1, min(MAX_FRAMES, len(all_frames)), dtype=int)
frames = [os.path.join(fldr, all_frames[i]) for i in idx]

cls_list, frame_feats = [], []
for i in range(0, len(frames), CHUNK):
    imgs = [Image.open(p).convert("RGB") for p in frames[i:i + CHUNK]]
    inp = processor(imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inp).last_hidden_state  # [1, 1 + T*P, D]

    cls_list.append(out[:, 0, :])  # [1, D]
    B, TP1, D = out.shape
    T = len(imgs)
    tkn = out[:, 1:, :].reshape(B, T, -1, D).mean(2)  # [1, T, D]
    frame_feats.append(tkn)

    del imgs, inp, out, tkn
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

cls_vec = torch.cat(cls_list, dim=0).mean(0, keepdim=True)
frames_cat = torch.cat(frame_feats, dim=1)
temporal = attn(frames_cat)

features[vid] = {
    "cls": cls_vec.cpu().numpy(),
    "temporal": temporal.detach().cpu().numpy()
}

with open(out_path, "wb") as f:
    pickle.dump(features, f)

print(f"Finished extracting features for video: {vid}")
