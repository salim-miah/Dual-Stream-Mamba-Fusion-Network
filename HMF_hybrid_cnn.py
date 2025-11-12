# -*- coding: utf-8 -*-

"""
ABLATION STUDY - V1_HMF_HybridCNN: Hierarchical Mamba Fusion (Hybrid CNN)

[HYBRID-CNN CHANGE]: This script implements the "Hybrid CNN" architecture.
It uses a "heavy" ResNet-50 for the high-detail Local stream,
and lightweight MobileNetV3-Small for all other streams.

Visual (Local):   ResNet-50 (2048)   → Mamba (T=16) ↘
Visual (Context): MobileNetV3 (576)  → Mamba (T=16) → Concat (T=16) → Fusion Mamba → MLP
Visual (Global):  MobileNetV3 (576)  → Mamba (T=16) ↗
Audio:            MobileNetV3 (576)  → Mamba (T=16) ↗

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
from mamba_ssm import Mamba
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import torch.multiprocessing  # #[CRITICAL FIX]: Import multiprocessing

# --- Environment Setup ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
#[CRITICAL FIX]: Add sharing strategy as suggested by the error message.
# This helps prevent "Too many open files" errors with multiprocessing.
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    print("Note: Could not set multiprocessing sharing strategy (might be on Windows or already set).")
    
print("✅ Libraries imported successfully.")

# --- 2. CONFIGURATION ---

class Config:
    def __init__(self):
        # --- Paths and Device ---
        self.data_dir = "/home/affshafee/T2430421/datasets/AVLips/AVLips"
        self.model_save_dir = "/home/affshafee/T2430421/pipelines/novel_fusion/HMF_hybrid_cnn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_save_dir, exist_ok=True)

        # --- Data Sampling ---
        self.use_sampling = False
        self.num_samples_per_class = 500

        # --- [HYBRID-CNN CHANGE]: CNN Feature Dims ---
        self.vis_image_size = (128, 128)
        self.vis_num_frames = 16  # Sequence Length T=16
        
        # Heavy backbone (ResNet-50) for Local/Mouth stream
        self.cnn_feature_dim_heavy = 2048 
        # Light backbone (MobileNetV3-Small) for other streams
        self.cnn_feature_dim_light = 576  
        
        # Mamba projects to this dim
        self.vis_mamba_d_model = 160

        # --- Audio Stream (MobileNetV3-Small) ---
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 16  # Matches vis_num_frames
        self.aud_chunk_duration = 5.0 / 16.0 # Total duration 5s
        self.aud_n_mels = 128
        self.aud_cnn_feature_dim = 576  # MobileNetV3-Small output
        self.aud_mamba_d_model = 160

        # --- Training Parameters ---
        self.batch_size = 32 # [HYBRID-CNN CHANGE]: Lowered batch size for heavy ResNet-50
        self.accumulation_steps = 8 # (32*8 = 256 effective)
        self.epochs = 50
        self.learning_rate = 5e-4
        self.weight_decay = 0.05
        self.patience = 10

config = Config()
print(f"✅ Configuration loaded. Using device: {config.device}")
print(f"🔥 Effective Batch Size: {config.batch_size * config.accumulation_steps}")
print(f"⚡ ARCHITECTURE: [HYBRID-CNN] Hierarchical Mamba Fusion (T={config.vis_num_frames})")
print(f"   Visual (Local):   ResNet-50 (Heavy)")
print(f"   Visual (Context): MobileNetV3-Small (Light)")
print(f"   Visual (Global):  MobileNetV3-Small (Light)")
print(f"   Audio Stream:     MobileNetV3-Small (Light)")
print(f"⚠️  NOTE: Batch size likely reduced to {config.batch_size} to fit ResNet-50 in VRAM.")

# --- 3. LABEL SMOOTHING LOSS ---

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)

# --- 4. DATA PROCESSING & DATASET CLASSES ---

# This function (process_visual_stream) remains unchanged from HMF_superior
def process_visual_stream(video_path: str, config: Config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release() # Ensure file handle is closed
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < config.vis_num_frames:
        cap.release()
        return None
    frame_indices = np.linspace(0, total_frames - 1, config.vis_num_frames, dtype=int)
    local_frames_list, context_frames_list, global_frames_list = [], [], []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            mouth_crop = frame[y + int(h * 0.6):y + h, x + int(w * 0.25):x + int(w * 0.75)]
            context_crop = frame[y + int(h * 0.4):y + h, x:x + w]
            global_crop = frame[y:y + h, x:x + w]
            if mouth_crop.size > 0 and context_crop.size > 0 and global_crop.size > 0:
                resized_local = cv2.resize(mouth_crop, config.vis_image_size)
                local_frames_list.append(cv2.cvtColor(resized_local, cv2.COLOR_BGR2RGB))
                resized_context = cv2.resize(context_crop, config.vis_image_size)
                context_frames_list.append(cv2.cvtColor(resized_context, cv2.COLOR_BGR2RGB))
                resized_global = cv2.resize(global_crop, config.vis_image_size)
                global_frames_list.append(cv2.cvtColor(resized_global, cv2.COLOR_BGR2RGB))
    cap.release() # Ensure file handle is closed
    if (len(local_frames_list) == config.vis_num_frames and
        len(context_frames_list) == config.vis_num_frames and
        len(global_frames_list) == config.vis_num_frames):
        return (np.stack(local_frames_list), 
                np.stack(context_frames_list), 
                np.stack(global_frames_list))
    else:
        return None

# This function (process_audio_stream) remains unchanged
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
            if len(chunk) == 0: chunk = np.zeros(samples_per_chunk)
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=config.aud_n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            mel_list.append(torch.tensor(mel_db, dtype=torch.float32))
        if len(mel_list) == config.aud_num_chunks:
            return torch.stack(mel_list, axis=0)
        else:
            print(f"Warning: Audio processing yielded {len(mel_list)} chunks, expected {config.aud_num_chunks}.")
            return None
    except Exception as e:
        print(f"Error processing audio {video_path}: {e}")
        return None

# This class (DualStreamDataset) remains unchanged
class DualStreamDataset(Dataset):
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
            visual_crops_hwc = process_visual_stream(video_path, self.config)
            if visual_crops_hwc is None: return None
            local_frames_hwc, context_frames_hwc, global_frames_hwc = visual_crops_hwc
            local_frames_tchw = local_frames_hwc.transpose(0, 3, 1, 2)
            context_frames_tchw = context_frames_hwc.transpose(0, 3, 1, 2)
            global_frames_tchw = global_frames_hwc.transpose(0, 3, 1, 2)
            audio_mels = process_audio_stream(video_path, self.config)
            if audio_mels is None: return None
            audio_tensor = audio_mels.unsqueeze(1)
            visual_data_tuple = (local_frames_tchw, context_frames_tchw, global_frames_tchw)
            return (visual_data_tuple, audio_tensor), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error in getitem for {video_path}: {e}")
            return None

# This class (RAMCachedDataset) remains unchanged
class RAMCachedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def _apply_transform(self, frames_np_array):
        if self.transform:
            augmented_frames = []
            for frame_np in frames_np_array:
                frame_hwc = frame_np.transpose(1, 2, 0) 
                augmented_frames.append(self.transform(frame_hwc))
            return torch.stack(augmented_frames)
        else:
            return torch.from_numpy(frames_np_array).float()
    def __getitem__(self, idx):
        local_frames_np, context_frames_np, global_frames_np, audio_tensor = self.data[idx]
        label = self.labels[idx]
        visual_tensor_local = self._apply_transform(local_frames_np)
        visual_tensor_context = self._apply_transform(context_frames_np)
        visual_tensor_global = self._apply_transform(global_frames_np)
        visual_data_tuple = (visual_tensor_local, visual_tensor_context, visual_tensor_global)
        return (visual_data_tuple, audio_tensor), label


# --- 5. HIERARCHICAL FUSION MODEL ARCHITECTURE ---

#[HYBRID-CNN CHANGE]: Renamed to reflect "Light" backbone
class VisualStream_Light_MobileNetV3(nn.Module):
    """Lightweight MobileNetV3-Small stream for Context/Global views."""
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn_features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        # Project from light dim -> mamba dim
        self.proj = nn.Linear(config.cnn_feature_dim_light, config.vis_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)
        self.mamba = Mamba(d_model=config.vis_mamba_d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_dropout = nn.Dropout(0.2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.cnn_features(x)
        features = self.avgpool(features)
        features = features.view(b, t, -1)
        projected_features = self.proj_dropout(self.proj(features))
        temporal_out = self.mamba(projected_features)
        temporal_out = self.mamba_dropout(temporal_out)
        return temporal_out # Return (B, T, D_model)

#[HYBRID-CNN CHANGE]: NEW Class for the "Heavy" backbone
class VisualStream_Heavy_ResNet50(nn.Module):
    """Heavyweight ResNet-50 stream for Local/Mouth view."""
    def __init__(self, config):
        super().__init__()
        # Load ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # We'll use all layers *except* the final classification layer (fc)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = resnet.avgpool
        
        # Project from heavy dim -> mamba dim
        self.proj = nn.Linear(config.cnn_feature_dim_heavy, config.vis_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)
        self.mamba = Mamba(d_model=config.vis_mamba_d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_dropout = nn.Dropout(0.2)

        print("---")
        print("⚠️  NOTE: ResNet-50 is using default ImageNet weights.")
        print("   For best performance, retrain this backbone on a face dataset (e.g., VGGFace2).")
        print("---")

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.cnn_features(x)
        features = self.avgpool(features)
        features = features.view(b, t, -1)
        projected_features = self.proj_dropout(self.proj(features))
        temporal_out = self.mamba(projected_features)
        temporal_out = self.mamba_dropout(temporal_out)
        return temporal_out # Return (B, T, D_model)

#[HYBRID-CNN CHANGE]: Renamed to reflect "Light" backbone
class AudioStream_Light_MobileNetV3(nn.Module):
    """Lightweight MobileNetV3-Small stream for Audio."""
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn_features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.proj = nn.Linear(config.aud_cnn_feature_dim, config.aud_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)
        self.mamba = Mamba(d_model=config.aud_mamba_d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_dropout = nn.Dropout(0.2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w).repeat(1, 3, 1, 1)
        features = self.cnn_features(x)
        features = self.avgpool(features)
        features = features.view(b, t, -1)
        projected_features = self.proj_dropout(self.proj(features))
        temporal_out = self.mamba(projected_features)
        temporal_out = self.mamba_dropout(temporal_out)
        return temporal_out # Return (B, T, D_model)

#[HYBRID-CNN CHANGE]: This model now instantiates the new hybrid streams.
class HierarchicalFusionModel(nn.Module):
    """
    [HYBRID-CNN CHANGE]: This model instantiates the hybrid (Heavy/Light) streams.
    It has 1 "Heavy" (ResNet-50) visual stream for Local/Mouth.
    It has 2 "Light" (MobileNetV3) visual streams for Context/Global.
    It has 1 "Light" (MobileNetV3) audio stream.
    A final Mamba block models the temporal relationship between all 4 streams.
    """
    def __init__(self, config):
        super().__init__()
        # [HYBRID-CNN CHANGE]: Instantiate the Heavy stream for Local
        self.visual_stream_local = VisualStream_Heavy_ResNet50(config)
        
        # [HYBRID-CNN CHANGE]: Instantiate Light streams for Context and Global
        self.visual_stream_context = VisualStream_Light_MobileNetV3(config)
        self.visual_stream_global = VisualStream_Light_MobileNetV3(config)
        
        # Instantiate Light audio stream
        self.audio_stream = AudioStream_Light_MobileNetV3(config)

        # [HYBRID-CNN CHANGE]: This calculation is UNCHANGED.
        # All streams (heavy or light) project down to vis/aud_mamba_d_model
        # before the final mamba. The input dim is the sum of all stream outputs.
        fusion_mamba_d_model = (3 * config.vis_mamba_d_model) + config.aud_mamba_d_model
        
        self.fusion_mamba = Mamba(
            d_model=fusion_mamba_d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fusion_mamba_dropout = nn.Dropout(0.2)

        # The fusion_head processes the output of the new fusion_mamba block
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_mamba_d_model, 256), # Input dim is from fusion_mamba
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)
        )
    
    # This forward pass is UNCHANGED from HMF_superior
    def forward(self, visual_input_local, visual_input_context, visual_input_global, audio_input):
        
        # 1. Get full hidden state sequences from all streams
        # (The internals of local_seq are now "heavy", but output dim is the same)
        local_seq = self.visual_stream_local(visual_input_local)     # (B, T, D_vis)
        context_seq = self.visual_stream_context(visual_input_context) # (B, T, D_vis)
        global_seq = self.visual_stream_global(visual_input_global)   # (B, T, D_vis)
        audio_seq = self.audio_stream(audio_input)                   # (B, T, D_aud)

        # 2. Concatenate hidden states from ALL FOUR streams at each time step
        fused_seq = torch.cat((local_seq, context_seq, global_seq, audio_seq), dim=2)

        # 3. Feed the new "fused sequence" into the final Mamba block
        fusion_mamba_out = self.fusion_mamba(fused_seq)
        fusion_mamba_out = self.fusion_mamba_dropout(fusion_mamba_out)

        # 4. Take the final output vector from the *Fusion Mamba*
        final_fused_vector = fusion_mamba_out[:, -1, :]

        # 5. Feed this final vector to the MLP for classification
        return self.fusion_head(final_fused_vector)

# --- 6. UTILITY FUNCTIONS ---
# (Unchanged)
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

# --- 7. TRAINING AND EVALUATION ---
# (Unchanged from HMF_superior)
def train_one_epoch(model, loader, optimizer, criterion, scaler, config):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for i, ((visual_inputs, audio_data), labels) in enumerate(pbar):
        local_data, context_data, global_data = visual_inputs
        local_data = local_data.to(config.device, non_blocking=True)
        context_data = context_data.to(config.device, non_blocking=True)
        global_data = global_data.to(config.device, non_blocking=True)
        audio_data = audio_data.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()
        with autocast():
            outputs = model(local_data, context_data, global_data, audio_data)
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
        for (visual_inputs, audio_data), labels in tqdm(loader, desc="Validating"):
            local_data, context_data, global_data = visual_inputs
            local_data = local_data.to(config.device, non_blocking=True)
            context_data = context_data.to(config.device, non_blocking=True)
            global_data = global_data.to(config.device, non_blocking=True)
            audio_data = audio_data.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()
            with autocast():
                outputs = model(local_data, context_data, global_data, audio_data)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- 8. MAIN EXECUTION ---

def main():
    print("\n" + "="*80 + "\nSTEP 1: PREPARING FILE LISTS\n" + "="*80)
    # (No changes in data loading)
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
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.3, random_state=42, stratify=labels)
    val_files, test_files, val_labels, test_labels = train_test_split(
        test_files, test_labels, test_size=0.5, random_state=42, stratify=test_labels)
    print(f"Total Videos: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print("\n" + "="*80 + "\nSTEP 2: PRE-LOADING & CACHING DATA INTO RAM\n" + "="*80)
    
    def collate_fn_skip_errors(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

    def cache_data(files, labels, desc):
        dataset = DualStreamDataset(files, labels, config)
        
        #[CRITICAL FIX]: Set num_workers=0 for caching to prevent "Too many open files".
        # This is a one-time step, so speed is less important than stability.
        # The fast, multi-worker loaders are used for training (STEP 3).
        loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0, collate_fn=collate_fn_skip_errors)
        
        cached_data, cached_labels = [], []
        for data, batch_labels in tqdm(loader, desc=f"Caching {desc}"):
            if data is not None:
                (visual_data_tuple, audio_batch) = data
                local_batch, context_batch, global_batch = visual_data_tuple
                for i in range(local_batch.shape[0]):
                    cached_data.append((
                        local_batch[i].numpy(), 
                        context_batch[i].numpy(), 
                        global_batch[i].numpy(), 
                        audio_batch[i]
                    ))
                    cached_labels.append(batch_labels[i])
        return cached_data, torch.tensor(cached_labels)
    
    cached_train_data, cached_train_labels = cache_data(train_files, train_labels, "Train Set")
    cached_val_data, cached_val_labels = cache_data(val_files, val_labels, "Validation Set")
    cached_test_data, cached_test_labels = cache_data(test_files, test_labels, "Test Set")
    print(f"✅ Caching complete!")
    print(f" - Train samples: {len(cached_train_data)}")
    print(f" - Val samples: {len(cached_val_data)}")
    print(f" - Test samples: {len(cached_test_data)}")

    print("\n" + "="*80 + "\nSTEP 3: CREATING FINAL DATALOADERS WITH ENHANCED AUGMENTATION\n" + "="*80)
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = RAMCachedDataset(cached_train_data, cached_train_labels, transform=train_transform)
    val_dataset = RAMCachedDataset(cached_val_data, cached_val_labels, transform=val_test_transform)
    test_dataset = RAMCachedDataset(cached_test_data, cached_test_labels, transform=val_test_transform)
    
    # [CRITICAL FIX]: This is correct! These loaders use RAM data (no file I/O)
    # and *should* use many workers for speed.
    num_workers = os.cpu_count() // 2
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"✅ DataLoaders created successfully (using {num_workers} workers for training).")

    print("\n" + "="*80 + "\nSTEP 4: BUILDING [HYBRID-CNN] HIERARCHICAL FUSION MODEL\n" + "="*80)
    
    model = HierarchicalFusionModel(config).to(config.device)

    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)

    print(f"📊 [HYBRID-CNN] HierarchicalFusionModel Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    print(f"   Parameters (Millions): {total_params/1e6:.3f}M")
    print(f"   (Note: Size is now dominated by the new ResNet-50 backbone)")

    print("\n" + "="*80 + "\nSTEP 5: TRAINING [HYBRID-CNN] HIERARCHICAL FUSION MODEL\n" + "="*80)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    scaler = GradScaler()
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    model_path = os.path.join(config.model_save_dir, 'v1_hmf_hybrid_cnn_best.pth')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"🚀 Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config)
        val_loss = validate_one_epoch(model, val_loader, criterion, config)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
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

    print("\n" + "="*80 + "\nSTEP 6: FINAL EVALUATION ON TEST SET\n" + "="*80)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model from {model_path} for final evaluation.")
    else:
        print("Warning: Best model not saved. Evaluating with last epoch model.")
        
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for (visual_inputs, audio_data), labels in tqdm(test_loader, desc="Final Evaluation"):
            local_data, context_data, global_data = visual_inputs
            local_data = local_data.to(config.device)
            context_data = context_data.to(config.device)
            global_data = global_data.to(config.device)
            audio_data = audio_data.to(config.device)
            outputs = model(local_data, context_data, global_data, audio_data)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Handle case where no predictions were made (e.g., test set was empty or caching failed)
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("ERROR: No test samples were evaluated. Cannot generate report.")
        return

    preds_binary = (all_preds > 0.5).astype(int)
    accuracy = (preds_binary == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else -1
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else -1
    loss_gap = final_train_loss - final_val_loss if final_train_loss != -1 else -1

    print(f"\n" + "="*80)
    print(f"📈 V1_HMF_HybridCNN: HIERARCHICAL FUSION (HYBRID) - FINAL RESULTS")
    print(f"="*80)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 AUC Score: {auc_score:.4f} ({auc_score*100:.2f}%)")
    print(f"🔄 Training/Validation Loss Gap: {loss_gap:.4f}")
    print(f"📏 Model Size: {model_size_mb:.2f} MB")
    print(f"🔢 Parameters: {total_params/1e6:.3f}M")
    print(f"="*80)

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, preds_binary, target_names=['Real (0)', 'Fake (1)']))

    print("\n" + "="*80 + "\nSTEP 7: VISUALIZING TRAINING HISTORY\n" + "="*80)
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('V1_HMF_HybridCNN (Hierarchical Fusion): Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_dir, 'v1_hmf_hybrid_cnn_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ V1_HMF_HybridCNN (Hierarchical Fusion) ablation study completed!")
    print("📝 Record these metrics for your Excel comparison:")
    print(f"   - Variant: V1_HMF_HybridCNN")
    print(f"   - Size (MB): {model_size_mb:.2f}")
    print(f"   - Params (M): {total_params/1e6:.3f}")
    print(f"   - Accuracy: {accuracy*100:.2f}%")
    print(f"   - AUC: {auc_score*100:.2f}%")
    print(f"   - Loss Gap: {loss_gap:.4f}")

if __name__ == '__main__':
    main()