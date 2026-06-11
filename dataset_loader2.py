# dataset_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pose_extractor import extract_gesture_matrix_from_image
import config

class RobotGestureDataset(Dataset):
    def __init__(self, base_folder_path, master_class_to_idx=None):
        self.samples = []
        if master_class_to_idx is None:
            self.class_names = sorted([d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        else:
            self.class_to_idx = master_class_to_idx
            self.class_names = list(master_class_to_idx.keys())
        
        for cls_name in os.listdir(base_folder_path):
            cls_folder = os.path.join(base_folder_path, cls_name)
            if os.path.isdir(cls_folder) and cls_name in self.class_to_idx:
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_folder, img_name), self.class_to_idx[cls_name]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        matrix = extract_gesture_matrix_from_image(img_path)
        return torch.tensor(matrix).unsqueeze(0), torch.tensor(label, dtype=torch.long)

def get_splits_and_loaders(folder_path, master_class_to_idx=None, split_data=True):
    """
    Creates data loaders. Splits into Train/Val only if split_data=True (for Mother dataset).
    Keeps dataset 100% whole if split_data=False (for Patient dataset).
    """
    dataset = RobotGestureDataset(folder_path, master_class_to_idx)
    if len(dataset) == 0: return None, None
    
    if split_data:
        # Enforce an 80/20 train/test split for the Mother dataset verification
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader
    else:
        # Pass the whole 100% user dataset for personal fine-tuning
        whole_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        return whole_loader, None