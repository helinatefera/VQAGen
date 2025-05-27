from transformers import CLIPTokenizer, CLIPTextModel
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cpu")  

class ClipBasedVQAGenerator:
    def __init__(
        self,
        train_feat,
        val_feat,
        test_feat,
        train_qa_am,
        val_qa_am,
        test_qa_am,
        batch_size,
        train_obj_feat_am=None,
        val_obj_feat_am=None,
        test_obj_feat_am=None,
        lang=None
    ):
        # Initialize CLIP components
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_model.to(device)
        self.clip_text_model.eval()
        self.batch_size = batch_size
        # Process answer labels
        all_ans = pd.concat(
            [train_qa_am["answer"], val_qa_am["answer"], test_qa_am["answer"]],
            ignore_index=True,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_ans)
        
        # Add label columns to DataFrames before creating datasets
        train_qa_am['label'] = self.label_encoder.transform(train_qa_am['answer'])
        val_qa_am['label'] = self.label_encoder.transform(val_qa_am['answer'])
        test_qa_am['label'] = self.label_encoder.transform(test_qa_am['answer'])
        
        # Create datasets
        self.train_ds = ClipBasedVQADataset(
            train_qa_am, train_feat, train_obj_feat_am, 
            self.clip_tokenizer, self.clip_text_model
        )
        self.val_ds = ClipBasedVQADataset(
            val_qa_am, val_feat, val_obj_feat_am,
            self.clip_tokenizer, self.clip_text_model
        )
        self.test_ds = ClipBasedVQADataset(
            test_qa_am, test_feat, test_obj_feat_am,
            self.clip_tokenizer, self.clip_text_model
        )
        
        # Get the correct input_dim from the dataset
        input_dim = self.train_ds.total_dim
        
        # Initialize model with the correct input_dim
        self.model = AttentionClassifier(
            input_dim, 
            512, 
            len(self.label_encoder.classes_), 
            attn_dim=2048
        )
        self.model.to(device)
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def collate_fn(self, batch):
        filtered = [(x, y) for x, y in batch if y.item() >= 0]
        if not filtered:
            # Use the total_dim from train_ds as fallback
            return torch.zeros(1, 1, self.train_ds.total_dim, device=device), torch.tensor([-1], device=device)
        
        x_seqs, labels = zip(*filtered)
        x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0)
        y = torch.stack(labels)
        return x_padded, y

class ClipBasedVQADataset(Dataset):
    def __init__(self, qa_df, feat_dict, obj_feat, clip_tokenizer, clip_text_model):
        self.qa_df = qa_df
        self.feat = feat_dict
        self.obj_feat = obj_feat
        self.clip_tokenizer = clip_tokenizer
        self.clip_text_model = clip_text_model
        
        # Get sample dimensions
        sample_vid = next(iter(self.feat))
        f_feats = self.feat[sample_vid]["frame_features"]
        self.frame_dim = f_feats[0].shape[-1] if f_feats else 512
        self.temp_dim = 514
        self.spat_dim = 515
        
        # Calculate expected dimensions
        self.video_dim = 4
        self.attn_dim = 512
        self.text_dim = 512
        self.total_dim = self.video_dim + self.attn_dim + self.frame_dim + self.text_dim + self.temp_dim + self.spat_dim

    def __len__(self):
        return len(self.qa_df)

    def __getitem__(self, idx):
        row = self.qa_df.iloc[idx]
        vid = row["video_id"]
        question = row["question"]
        label = torch.tensor(row["label"], dtype=torch.long, device=device)

        video_feature = torch.zeros(self.video_dim, device=device)
        attn_feature = torch.zeros(self.attn_dim, device=device)
        frame_features = torch.zeros(self.frame_dim, device=device)
        temporal_feature = torch.zeros(self.temp_dim, device=device)
        spatial_feature = torch.zeros(self.spat_dim, device=device)
        
        if vid in self.feat:
            video_feature = torch.tensor(self.feat[vid]['video_feature'], dtype=torch.float, device=device)
            attn_feature = torch.tensor(self.feat[vid]['attn_feature'], dtype=torch.float, device=device)
            raw_features = self.feat[vid]['frame_features']
            if raw_features:
                frame_features = torch.stack([
                    torch.tensor(f, dtype=torch.float, device=device) 
                    if not isinstance(f, torch.Tensor) else f.to(device) 
                    for f in raw_features
                ]).mean(dim=0)
        
        if vid in self.obj_feat:
            temporal_data = self.obj_feat[vid].get('temporal_features', [])
            spatial_data = self.obj_feat[vid].get('spatial_features', [])
            
            if temporal_data:
                displacements = torch.tensor([t['displacement'] for t in temporal_data], dtype=torch.float, device=device)
                emb_diffs = torch.tensor([t['emb_diff'] for t in temporal_data], dtype=torch.float, device=device)
                emb_froms = torch.stack([torch.tensor(t['emb_from'], dtype=torch.float, device=device) for t in temporal_data])
                temporal_feature = torch.cat([
                    displacements.mean().unsqueeze(0),
                    emb_diffs.mean().unsqueeze(0),
                    emb_froms.mean(dim=0)
                ])
            
            if spatial_data:
                distances = torch.tensor([s['distance'] for s in spatial_data], dtype=torch.float, device=device)
                offsets = torch.tensor([s['horizontal_offset'] for s in spatial_data], dtype=torch.float, device=device)
                emb_sims = torch.tensor([s['emb_similarity'] for s in spatial_data], dtype=torch.float, device=device)
                obj1_embs = torch.stack([torch.tensor(s['obj1_emb'], dtype=torch.float, device=device) for s in spatial_data])
                spatial_feature = torch.cat([
                    distances.mean().unsqueeze(0),
                    offsets.mean().unsqueeze(0),
                    emb_sims.mean().unsqueeze(0),
                    obj1_embs.mean(dim=0)
                ])

        inputs = self.clip_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            q_feat = self.clip_text_model(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        
        fusion = torch.cat([
            video_feature,
            attn_feature,
            frame_features,
            q_feat,
            temporal_feature,
            spatial_feature
        ], dim=0)

        assert fusion.shape[0] == self.total_dim, f"Expected dim {self.total_dim}, got {fusion.shape[0]}"
        
        return fusion, label    

class AttentionClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, attn_dim=2048):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_dim, attn_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(attn_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Ensure x has the correct shape for MultiheadAttention (batch_size, seq_len, embed_dim)
        if x.dim() == 2:  # (batch_size, input_dim)
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.input_projection(x)
        attn_output, _ = self.attn(x, x, x)
        pooled = attn_output.mean(dim=1)
        return self.mlp(pooled)