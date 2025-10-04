# -*- coding: utf-8 -*-

"""
PHASE 1 (LSTM Variant): Lip-Sync Detection Pipeline with Overfitting Fixes

This script implements Phase 1 improvements to address overfitting:
1. Dropout added to Mamba blocks and increased in fusion head
2. ReduceLROnPlateau scheduler instead of CosineAnnealingLR
3. Enhanced data augmentation (ColorJitter, RandomRotation, RandomAffine)
4. Label smoothing for BCE loss
5. Unified training approach (no two-stage training)
"""

# --- 1. IMPORTS ---

import os
import cv2
import time
import torch
import librosa
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings

# --- Environment Setup ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
print("✅ Libraries imported successfully.")

# --- 2. CONFIGURATION ---

class Config:
    def __init__(self):
        # --- Paths and Device ---
        self.data_dir = "C:\Thesis\Thesis Team Deepfake Detection\Datasets\AVLips v1.0\AVLips"
        self.model_save_dir = "C:\Thesis\Thesis Team Deepfake Detection\Test Runs\LSTM based v2\models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # --- Data Sampling (for quick debugging) ---
        self.use_sampling = True
        self.num_samples_per_class = 2000  # Use a subset for faster runs
        
        # --- Visual Stream ---
        self.vis_image_size = (128, 128)
        self.vis_num_frames = 16
        self.vis_cnn_feature_dim = 1280  # EfficientNet-B0 output
        self.vis_mamba_d_model = 128
        
        # --- Audio Stream ---
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 5
        self.aud_chunk_duration = 1.0
        self.aud_n_mels = 128
        self.aud_cnn_feature_dim = 1280  # MobileNetV2 output
        self.aud_mamba_d_model = 128
        
        
        # --- Recurrent Temporal Modeling (LSTM replacement for Mamba) ---
        self.lstm_num_layers = 2
        self.lstm_bidirectional = True
        self.lstm_dropout = 0.2
# --- Training Parameters (PHASE 1 CHANGES) ---
        self.batch_size = 32
        self.accumulation_steps = 4  # Effective batch size 128
        self.epochs = 25  # Single unified training
        self.learning_rate = 5e-4  # Conservative starting point
        self.weight_decay = 0.05  # Increased for regularization
        self.patience = 6

config = Config()
print(f"✅ Configuration loaded. Using device: {config.device}")
print(f"🔥 Effective Batch Size: {config.batch_size * config.accumulation_steps}")

# --- 3. LABEL SMOOTHING LOSS ---

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)

# --- 4. DATA PROCESSING & DATASET CLASSES ---

# --- Visual Processing ---
def process_visual_stream(video_path: str, config: Config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < config.vis_num_frames: 
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, config.vis_num_frames, dtype=int)
    frames = []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: 
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # Crop the lower half of the face for the mouth region
            mouth_crop = frame[y + int(h * 0.6):y + h, x + int(w * 0.25):x + int(w * 0.75)]
            if mouth_crop.size > 0:
                resized_crop = cv2.resize(mouth_crop, config.vis_image_size)
                # Convert BGR to RGB
                resized_crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                frames.append(resized_crop_rgb)
    
    cap.release()
    return np.stack(frames) if len(frames) == config.vis_num_frames else None

# --- Audio Processing ---
def process_audio_stream(video_path: str, config: Config):
    try:
        parts = Path(video_path).parts
        audio_filename = Path(video_path).stem + ".wav"
        label_folder = parts[-2]
        base_data_dir = str(Path(video_path).parent.parent)
        audio_path = os.path.join(base_data_dir, "wav", label_folder, audio_filename)
        
        y, sr = librosa.load(audio_path, sr=config.aud_sample_rate)
        total_samples = int(config.aud_chunk_duration * config.aud_num_chunks * sr)
        
        if len(y) < total_samples:
            y = np.pad(y, (0, total_samples - len(y)), mode='constant')
        else:
            y = y[:total_samples]
        
        samples_per_chunk = int(config.aud_chunk_duration * sr)
        mel_list = []
        
        for i in range(config.aud_num_chunks):
            chunk = y[i*samples_per_chunk : (i+1)*samples_per_chunk]
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=config.aud_n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            mel_list.append(torch.tensor(mel_db, dtype=torch.float32))
        
        return torch.stack(mel_list, axis=0)
    except Exception:
        return None

# --- Dataset for Initial Disk I/O ---
class DualStreamDataset(Dataset):
    """Processes files from disk one time for caching."""
    def __init__(self, file_paths, labels, config):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # First, process visual stream. If it fails, we skip audio.
            visual_frames_hwc = process_visual_stream(video_path, self.config)
            if visual_frames_hwc is None: 
                return None
            
            # Transpose from (T, H, W, C) to (T, C, H, W) immediately.
            visual_frames_tchw = visual_frames_hwc.transpose(0, 3, 1, 2)
            
            # Now process audio stream
            audio_mels = process_audio_stream(video_path, self.config)
            if audio_mels is None: 
                return None
            
            # Add a channel dimension for the CNN
            audio_tensor = audio_mels.unsqueeze(1)
            
            return (visual_frames_tchw, audio_tensor), torch.tensor(label, dtype=torch.float32)
        except Exception:
            return None

# --- Dataset for Serving from RAM ---
class RAMCachedDataset(Dataset):
    """Serves pre-processed data from a list in RAM and applies transforms on-the-fly."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Data is already pre-processed and in (visual_np, audio_tensor) format
        visual_frames_np, audio_tensor = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            augmented_frames = []
            # visual_frames_np is already (T, C, H, W)
            for frame_np in visual_frames_np:
                # frame_np is (C, H, W), transpose to (H, W, C) for ToPILImage
                frame_hwc = frame_np.transpose(1, 2, 0)
                augmented_frames.append(self.transform(frame_hwc))
            visual_tensor = torch.stack(augmented_frames)
        else:
            visual_tensor = torch.from_numpy(visual_frames_np).float()
        
        return (visual_tensor, audio_tensor), label

# --- 5. IMPROVED MODEL ARCHITECTURE WITH DROPOUT ---

class VisualStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.proj = nn.Linear(config.vis_cnn_feature_dim, config.vis_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)  # PHASE 1: Added dropout
        
        # LSTM replaces Mamba for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.vis_mamba_d_model,
            hidden_size=config.vis_mamba_d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.temporal_dropout = nn.Dropout(0.2)  # keep similar regularization

        # Determine output feature dimension after LSTM
        self.out_dim = config.vis_mamba_d_model * (2 if config.lstm_bidirectional else 1)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.avgpool(self.cnn_features(x)).view(b, t, -1)
        projected_features = self.proj_dropout(self.proj(features))
        temporal_out, _ = self.lstm(projected_features)
        temporal_out = self.temporal_dropout(temporal_out)
        # Take the last timestep representation (concat of directions if bidirectional)
        return temporal_out[:, -1, :]
    
class AudioStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn_features = mobilenet.features
        self.final_proj = nn.Linear(mobilenet.last_channel, config.aud_cnn_feature_dim)
        self.proj = nn.Linear(config.aud_cnn_feature_dim, config.aud_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)  # PHASE 1: Added dropout
        
        # LSTM replaces Mamba for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.aud_mamba_d_model,
            hidden_size=config.aud_mamba_d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.temporal_dropout = nn.Dropout(0.2)  # keep similar regularization
        
        # Determine output feature dimension after LSTM
        self.out_dim = config.aud_mamba_d_model * (2 if config.lstm_bidirectional else 1)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w).repeat(1, 3, 1, 1)
        features = F.adaptive_avg_pool2d(self.cnn_features(x), (1, 1)).view(b, t, -1)
        features = self.final_proj(features)
        projected_features = self.proj_dropout(self.proj(features))  # Apply dropout
        temporal_out, _ = self.lstm(projected_features)
        temporal_out = self.temporal_dropout(temporal_out)  # Apply dropout
        # Take the last timestep representation (concat of directions if bidirectional)
        return temporal_out[:, -1, :]

class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_stream = VisualStream(config)
        self.audio_stream = AudioStream(config)
        
        visual_out_dim = self.visual_stream.out_dim
        audio_out_dim = self.audio_stream.out_dim
        fusion_input_dim = visual_out_dim + audio_out_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # PHASE 1: Increased from 0.4 to 0.6
            nn.Linear(256, 1)
        )
    
    def forward(self, visual_input, audio_input):
        visual_features = self.visual_stream(visual_input)
        audio_features = self.audio_stream(audio_input)
        fused_features = torch.cat((visual_features, audio_features), dim=1)
        return self.fusion_head(fused_features)

# --- 6. TRAINING AND EVALUATION LOGIC ---

def train_one_epoch(model, loader, optimizer, criterion, scaler, config):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    
    for i, ((visual_data, audio_data), labels) in enumerate(pbar):
        visual_data = visual_data.to(config.device, non_blocking=True)
        audio_data = audio_data.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()
        
        with autocast():
            outputs = model(visual_data, audio_data)
            loss = criterion(outputs, labels)
            loss = loss / config.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * config.accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * config.accumulation_steps:.4f}"})
    
    return total_loss / len(loader)

def validate_one_epoch(model, loader, criterion, config):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for (visual_data, audio_data), labels in tqdm(loader, desc="Validating"):
            visual_data = visual_data.to(config.device, non_blocking=True)
            audio_data = audio_data.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()
            
            with autocast():
                outputs = model(visual_data, audio_data)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

# --- 7. MAIN EXECUTION SCRIPT ---

# Helper function for DataLoader - must be at module level for pickling
def collate_fn_skip_errors(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

def main():
    # --- Step 1: Prepare File Lists ---
    print("\n" + "="*80 + "\nSTEP 1: PREPARING FILE LISTS\n" + "="*80)
    
    real_dir = os.path.join(config.data_dir, "0_real")
    fake_dir = os.path.join(config.data_dir, "1_fake")
    
    all_real = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    all_fake = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    
    if config.use_sampling:
        print(f"🔥 Sampling {config.num_samples_per_class} videos per class...")
        real_files = np.random.choice(all_real, config.num_samples_per_class, replace=False).tolist()
        fake_files = np.random.choice(all_fake, config.num_samples_per_class, replace=False).tolist()
    else:
        print("🎬 Using the full dataset.")
        real_files, fake_files = all_real, all_fake
    
    all_files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    # Split file paths before caching
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.3, random_state=42, stratify=labels)
    val_files, test_files, val_labels, test_labels = train_test_split(
        test_files, test_labels, test_size=0.5, random_state=42, stratify=test_labels)
    
    print(f"Total Videos: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    # --- Step 2: Pre-load and Cache Data ---
    print("\n" + "="*80 + "\nSTEP 2: PRE-LOADING & CACHING DATA INTO RAM\n" + "="*80)
    
    def cache_data(files, labels, desc):
        dataset = DualStreamDataset(files, labels, config)
        loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=2, collate_fn=collate_fn_skip_errors)
        cached_data, cached_labels = [], []
        
        for data, batch_labels in tqdm(loader, desc=f"Caching {desc}"):
            if data is not None:
                visual_batch, audio_batch = data
                for i in range(visual_batch.shape[0]):
                    cached_data.append((visual_batch[i].numpy(), audio_batch[i]))
                    cached_labels.append(batch_labels[i])
        
        return cached_data, torch.tensor(cached_labels)
    
    cached_train_data, cached_train_labels = cache_data(train_files, train_labels, "Train Set")
    cached_val_data, cached_val_labels = cache_data(val_files, val_labels, "Validation Set")
    cached_test_data, cached_test_labels = cache_data(test_files, test_labels, "Test Set")
    
    print(f"✅ Caching complete!")
    print(f" - Train samples: {len(cached_train_data)}")
    print(f" - Val samples: {len(cached_val_data)}")
    print(f" - Test samples: {len(cached_test_data)}")
    
    # --- Step 3: Define ENHANCED Augmentations and Create Final DataLoaders ---
    print("\n" + "="*80 + "\nSTEP 3: CREATING FINAL DATALOADERS WITH ENHANCED AUGMENTATION\n" + "="*80)
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # PHASE 1: Enhanced training transform with more augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # ADDED
        transforms.RandomRotation(10),  # ADDED
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ADDED
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RAMCachedDataset(cached_train_data, cached_train_labels, transform=train_transform)
    val_dataset = RAMCachedDataset(cached_val_data, cached_val_labels, transform=val_test_transform)
    test_dataset = RAMCachedDataset(cached_test_data, cached_test_labels, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print("✅ DataLoaders created successfully from RAM cache with enhanced augmentation.")
    
    # --- Step 4: UNIFIED Training (No Two-Stage) ---
    print("\n" + "="*80 + "\nSTEP 4: UNIFIED TRAINING WITH REDUCED LEARNING RATE PLATEAU\n" + "="*80)
    
    model = FusionModel(config).to(config.device)
    
    # PHASE 1: Use AdamW with weight decay and conservative learning rate
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # PHASE 1: Use label smoothing loss instead of regular BCE
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    
    scaler = GradScaler()
    
    # PHASE 1: Use ReduceLROnPlateau instead of CosineAnnealingLR
    scheduler = lr_scheduler.ReduceLROnPlateau(
        # optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    model_path = os.path.join(config.model_save_dir, 'phase1_best.pth')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"🚀 Starting unified training for {config.epochs} epochs...")
    print(f"📊 Learning Rate: {config.learning_rate}, Weight Decay: {config.weight_decay}")
    print(f"🎯 Using Label Smoothing (0.1) and ReduceLROnPlateau scheduler")
    
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config)
        val_loss = validate_one_epoch(model, val_loader, criterion, config)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"🏆 New best model saved! Val Loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break
    
    # --- Step 5: Final Evaluation ---
    print("\n" + "="*80 + "\nSTEP 5: FINAL EVALUATION ON TEST SET\n" + "="*80)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for (visual_data, audio_data), labels in tqdm(test_loader, desc="Final Evaluation"):
            visual_data = visual_data.to(config.device)
            audio_data = audio_data.to(config.device)
            
            outputs = model(visual_data, audio_data)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    preds_binary = (all_preds > 0.5).astype(int)
    
    accuracy = (preds_binary == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_preds)
    
    print(f"\n📈 PHASE 1 RESULTS:")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 AUC Score: {auc_score:.4f}")
    print(f"🔄 Training/Validation Loss Gap: {history['train_loss'][-1] - history['val_loss'][-1]:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, preds_binary, target_names=['Real (0)', 'Fake (1)']))
    
    # --- Step 6: Plot Loss Curve ---
    print("\n" + "="*80 + "\nSTEP 6: VISUALIZING TRAINING HISTORY\n" + "="*80)
    
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Phase 1: Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_dir, 'phase1_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Phase 1 training completed!")
    print("🔍 Check if the validation loss gap has improved compared to your original results.")
    print("📝 If overfitting persists, proceed to Phase 2 modifications.")

if __name__ == '__main__':
    main()
