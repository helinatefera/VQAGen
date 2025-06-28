# Multimodal Understanding for Amharic Video Question Answering using Bidirectional Cross Modal Attention

[![Demo](https://img.shields.io/badge/Live%20Demo-github.com%2Fhelinatefera%2FVQAGen--Demo-blue?style=for-the-badge)](https://github.com/helinatefera/VQAGen-Demo)

![Model Architecture](img/arch.png)


This project proposes a first Video Question Answering (VideoQA) system designed specifically for the Amharic language. It integrates multiple modalities visual frames, object features, and textual data using a bidirectional cross-modal attention mechanism to answer natural language questions based on video content. 

In the bidirectional cross-modal attention used in my model, features are processed as follows:

Amharic BERT encodes the question into a sequence of token embeddings. Visual features using timeSformer, object-level features from CLIP and FastRCN, linearly projected into the same dimensional space as the text.

The bidirectional attention operates in two directions:

* **Text-to-Visual Attention**: Each token in the question attends to relevant visual features, helping the model identify visual cues that align with the question semantics.
* **Visual-to-Text Attention**: Each visual feature attends to the question tokens, allowing the model to refine visual understanding based on the linguistic context.

Both attention outputs are combined or fused via concatenation and passed through additional layers for classification.

## 🎯 Objectives

- Build a functional multimodal QA pipeline for Amharic videos
- Novel Frame Selection Method Based on MCLIP
- Extract visual, object, and textual features for each video
- Train a cross-modal attention model to align vision and language
- Evaluate performance using standard classification metrics

## 📁 Dataset

- **Videos**: Each video is segmented into frames and stored by `video_id`
- **CSV Files**: Include `video_id`, `question`, `answer` (in Amharic)
- **Features**:
  - TimeSformer (CLS token) features
  - Object-level features using FastRCNN
- **Splits**: Training, validation, and test sets are separated

## 🧠 Model

- **Text Encoder**: Amharic BERT
- **Visual Encoders**:
  - CLIP for frame-level features
  - Temporal Video repesentation with TimeSformer
  - FastRCNN for object features
- **Fusion Mechanism**: Bidirectional cross-modal attention (8 heads, 2 layers)
- **Output**: Simple classifier over answer candidates

## 🛠 Training Configuration

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 50 with early epoch 5
- **Metrics**: Accuracy, Precision, Recall, F1-score

## 🧪 Evaluation Tasks

- Predict the correct answer for a given Amharic question and video
- Evaluate performance across various feature types and combinations
- Analyze model attention weights to interpret modality contribution




