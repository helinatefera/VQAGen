from pathlib import Path
import pandas as pd
import pickle
from .models_am import clip_based
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import random
import numpy as np

device = torch.device("cpu")  
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lang = cfg.training.lang   
        self.data_root_dir = Path(cfg.data.root_dir)
        self.epochs = cfg.training.epochs
        self.warmup_epochs = cfg.training.WARMUP_EPOCHS
        self.patience = cfg.training.PATIENCE
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.lr = cfg.training.lr
        self.start_epoch = cfg.training.start_epoch
        self.checkpoint_path = cfg.training.best_checkpoint_path
        self.batch_size = cfg.training.batch_size
        self.seed = cfg.training.seed
        # Initialize components
        self.load_data()

    def load_data(self):
        if self.lang == 'eng':
            self.train_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.train)
            self.val_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.val)
            self.test_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.test)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.train, 'rb') as f:
                self.train_obj_feat = pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.val, 'rb') as f:
                self.val_obj_feat = pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.test, 'rb') as f:
                self.test_obj_feat = pickle.load(f)
        else:
            self.train_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.train)
            self.val_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.val)
            self.test_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.test)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.train, 'rb') as f:
                self.train_obj_feat = pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.val, 'rb') as f:
                self.val_obj_feat = pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.test, 'rb') as f:
                self.test_obj_feat = pickle.load(f)
        
        with open(self.data_root_dir / self.cfg.data.feat_path.train, 'rb') as f:
            self.train_feat = pickle.load(f)
        with open(self.data_root_dir / self.cfg.data.feat_path.val, 'rb') as f:
            self.val_feat = pickle.load(f)
        with open(self.data_root_dir / self.cfg.data.feat_path.test, 'rb') as f:
            self.test_feat = pickle.load(f)

    def evaluate_metrics(self, model, loader, label_encoder):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in loader:
                if y[0] < 0:  # Skip invalid samples
                    continue
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        total = len(all_labels)
        if total == 0:
            return 0, 0, 0, 0, None

        loss = total_loss / total
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        return loss, accuracy, f1, precision, cm

    def train(self):
        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.clip = clip_based.ClipBasedVQAGenerator(
            lang="am",
            train_qa_am=self.train_df,
            val_qa_am=self.val_df,
            test_qa_am=self.test_df,
            train_obj_feat_am=self.train_obj_feat,
            val_obj_feat_am=self.val_obj_feat,
            test_obj_feat_am=self.test_obj_feat,
            train_feat=self.train_feat,
            val_feat=self.val_feat,
            test_feat=self.test_feat,
            batch_size=self.batch_size
        )  
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.clip.model.parameters(), 
            lr=self.lr
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=3
        )
        
        # Load checkpoint if exists
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            try:
                self.clip.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                train_losses = checkpoint.get('loss_history', [])
                train_accs = checkpoint.get('train_acc_history', [])
                val_accs = checkpoint.get('val_acc_history', [])
                best_val_acc = max(val_accs) if val_accs else 0.0
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
        else:
            print("No checkpoint found. Starting from scratch.")
            
        train_losses = []
        train_accs = []
        val_accs = []
        val_losses = []  # Added to track validation loss for plotting

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Training phase
            self.clip.model.train()
            epoch_loss, epoch_acc = self.run_epoch(self.clip.train_loader, training=True)
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Validation phase
            val_loss, val_acc = self.run_epoch(self.clip.val_loader, training=False)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Checkpointing
            if val_acc > self.best_val_acc:
                self.save_checkpoint(epoch, train_losses, train_accs, val_accs)
            
            # Early stopping
            if self.check_early_stopping(val_acc):
                break
                
            print(f"Epoch {epoch}/{self.epochs}: "
                  f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_f1, val_precision, val_cm = self.evaluate_metrics(
            self.clip.model, self.clip.val_loader, self.clip.label_encoder
        )
        print("\nValidation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        print(f"Precision: {val_precision:.4f}")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_f1, test_precision, test_cm = self.evaluate_metrics(
            self.clip.model, self.clip.test_loader, self.clip.label_encoder
        )
        print("\nTest Metrics:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1 Score: {test_f1:.4f}")
        print(f"Precision: {test_precision:.4f}")
        
        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accs) + 1)
        plt.plot(epochs, [acc * 100 for acc in train_accs], 'b-o', label='Training Accuracy')
        plt.plot(epochs, [acc * 100 for acc in val_accs], 'r-o', label='Validation Accuracy')
        plt.plot([epochs[-1]], [test_accuracy * 100], 'g*', markersize=10, label=f'Test Accuracy ({test_accuracy * 100:.2f}%)')
        plt.title('Training, Validation, and Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('accuracy_plot.png')  # Adjusted path for local execution
        plt.close()

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_plot.png')  # Adjusted path for local execution
        plt.close()

    def run_epoch(self, loader, training=True):
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(loader, desc="Training" if training else "Validation")
        for x, y in loop:
            if y[0] < 0:  # Skip invalid samples
                continue
                
            x, y = x.to(device), y.to(device)
            
            if training:
                self.optimizer.zero_grad()
                out = self.clip.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    out = self.clip.model(x)
                    loss = self.criterion(out, y)
            
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
            
            loop.set_postfix(loss=total_loss/total, acc=100.*correct/total)
        
        return total_loss/total, 100.*correct/total

    def save_checkpoint(self, epoch, train_losses, train_accs, val_accs):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.clip.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': train_losses,
            'train_acc_history': train_accs,
            'val_acc_history': val_accs
        }, self.checkpoint_path)
        self.best_val_acc = max(val_accs)

    def check_early_stopping(self, val_acc):
        if val_acc <= self.best_val_acc:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping triggered.")
                return True
        else:
            self.patience_counter = 0
        return False