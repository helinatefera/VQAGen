# Multimodal Understanding for Amharic Video Question Answering using Bidirectional Cross Modal Attention

[![Demo](https://img.shields.io/badge/Demo-github.com%2Fhelinatefera%2FVQAGen--Demo-blue?style=for-the-badge)](https://github.com/helinatefera/VQAGen-Demo)

![Model Architecture](img/arch.png)

This project proposes the first Video Question Answering (VideoQA) system tailored for the Amharic language. It integrates multiple modalities—visual frames, object-level features, and text—through a bidirectional cross-modal attention mechanism.

In the model:

- **Text features**: Amharic BERT encodes the question.
- **Visual features**: Extracted from TimeSformer (CLS), CLIP, and FastRCNN, projected to the same space.
- **Cross-modal attention**:
  - **Text → Visual**: Tokens attend to visual regions.
  - **Visual → Text**: Visual regions attend to token embeddings.
- **Fusion**: Attention outputs are concatenated and passed to a classifier.

---

## 🎯 Objectives

- Build a multimodal QA pipeline for Amharic videos.
- Apply novel frame selection using MCLIP.
- Extract video, object, and text features.
- Train a cross-modal attention model.
- Evaluate performance using classification metrics.

---

## 📁 Dataset Structure

- `videos/`: Video frames organized by `video_id/`
- CSV files with: `video_id`, `question`, `answer` (Amharic)
- Feature folders:
  - `TimeSformer` CLS features
  - `FastRCNN` object features
  - `CLIP` frame embeddings
- Splits: `train`, `val`, `test`

---

## 🧠 Model Summary

- **Text Encoder**: Amharic BERT
- **Visual Encoders**:
  - TimeSformer (temporal)
  - CLIP (frame-level)
  - FastRCNN (object-level)
- **Fusion**: 2-layer bidirectional cross-modal attention (8 heads)
- **Classifier**: Linear layer over concatenated fusion output

---

## 🛠 Training Configuration

- **Loss**: CrossEntropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 50 (early stop at 5)
- **Metrics**: Accuracy, Precision, Recall, F1-score

---

## 📦 Clone the Repository

```bash
git clone https://github.com/helinatefera/VQAGen
cd VQAGen
````

---

## 📥 Download Dataset

Download the dataset from Hugging Face:

👉 [HuggingFace Dataset Link](https://huggingface.co/datasets/hinaltt/vide0_qa_dataset/tree/main)

After downloading, extract everything and place it into the `datasets` folder:

```bash
mkdir -p datasets
# Move all downloaded contents into datasets/
```

Expected structure:

```
datasets/
├── qa/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── obj_feat/
│   ├── *.pkl
├── clip-rcnn-attn/
│   ├── *.pkl
```

---

## 🚀 Train the Model

```bash
python -m vqagen
```

---

## 🧼 License

MIT License © helinatefera


## 📞 Contact

👤 **Helina Tefera**  
✉️ [E-Mail](mailto:helinatefera1212@gmail.com)  
📱 [Phone](tel:+251929453545)

