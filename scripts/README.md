## 🧪 One-Video Feature Extraction Pipeline

This guide helps you run all scripts for extracting visual features from a single video for Amharic VideoQA tasks.

### 📥 1. Download Amharic Captions

Download the caption file from Hugging Face:

👉 [Download MSDV Amharic Captions](https://huggingface.co/datasets/hinaltt/video_caption/tree/main)

Save it as:

```
video_captions/MSDV_amharic_caption.txt
```

---

### 🎞️ 2. Extract All Frames

Run the following command to extract all frames from a video:

```bash
python scripts/all_frame_sample.py path/to/sample_video.mp4
```

All frames will be saved in `video_frames/<video_id>`.

---

### 🧠 3. Select 16 Representative Frames

This step uses CLIP and M-CLIP with KMeans and caption-guided selection:

```bash
python scripts/m-clipSampling.py \
  --video_id <video_id> \
  --video_frames_folder video_frames \
  --annotation_file video_captions/MSDV_amharic_caption.txt \
  --output_base_folder 16_best
```

Resulting folder: `16_best/<video_id>/selected_frame_000.jpg` to `015.jpg`.

---

### 🔍 4. Extract Object-Level Features

This step uses Faster R-CNN and CLIP to extract object features and align them with caption words:

```bash
python scripts/object_labeling.py \
  --video_frames_root 16_best \
  --video_id <video_id>
```

Output: `object_clip_q_label_am.pkl` containing boxes, words, and features.

---

### ⏱️ 5. Extract TimeSformer Features

Use TimeSformer for extracting CLS and temporal features:

```bash
python scripts/timeSformer.py \
  --video_frames_root 16_best \
  --video_id <video_id>
```

Output: `timesformer_features.pkl` with `cls` and `temporal` features.
