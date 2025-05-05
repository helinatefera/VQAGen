from pathlib import Path
import pandas as pd
import pickle

class Trainer:
    def __init__(self,cfg):
        self.cfg = cfg
        self.lang = cfg.training.lang   
        self.data_root_dir = Path(cfg.data.root_dir)

        self.load_data()
    

    def load_data(self):
        # Load your dataset here
        
        if self.lang == 'eng':
            self.train_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.train)
            self.val_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.val)
            self.test_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_eng.test)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.train, 'rb') as f:
                    self.train_obj_feat =  pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.val, 'rb') as f:
                    self.val_obj_feat =  pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_eng.test, 'rb') as f:
                    self.test_obj_feat =  pickle.load(f)
        else:
            self.train_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.train)
            self.val_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.val)
            self.test_df = pd.read_csv(self.data_root_dir / self.cfg.data.qa_path_am.test)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.train, 'rb') as f:
                    self.train_obj_feat =  pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.val, 'rb') as f:
                    self.val_obj_feat =  pickle.load(f)
            with open(self.data_root_dir / self.cfg.data.obj_feat_am.test, 'rb') as f:
                    self.test_obj_feat =  pickle.load(f)
        
        with open(self.data_root_dir / self.cfg.data.feat_path.train, 'rb') as f:
          self.train_feat =  pickle.load(f)
        with open(self.data_root_dir / self.cfg.data.feat_path.val, 'rb') as f:
            self.val_feat =  pickle.load(f)
        with open(self.data_root_dir / self.cfg.data.feat_path.test, 'rb') as f:
            self.test_feat =  pickle.load(f)
    

    def train(self):
        # Implement your training loop here
        print("Training started...")