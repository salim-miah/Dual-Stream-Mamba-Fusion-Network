# -*- coding: utf-8 -*-

"""
ABLATION STUDY - V1_HMF_MultiDataset: Hierarchical Mamba Fusion (Multi-Dataset)

[MULTI-DATASET CHANGE]: This script is based on HMF_superior.py.
It is modified to load and process data from three different datasets:
1. AVLips (separate audio/video files)
2. DeepSpeak (stitched audio/video files)
3. LipSyncTIMIT (stitched audio/video files)

The goal is to train on a mixed domain to improve generalization.

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
import torch.multiprocessing

# --- Environment Setup ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass
print("✅ Libraries imported successfully.")

# --- 2. CONFIGURATION ---

class Config:
    def __init__(self):
        # [MULTI-DATASET CHANGE]: Paths for all three datasets
        # !! PLEASE VERIFY THESE PATHS ARE CORRECT FOR YOUR SYSTEM !!
        self.dataset_paths = {
            'avlips': "/home/affshafee/T2430421/datasets/AVLips/AVLips",
            'deepspeak': "/home/affshafee/T2430421/datasets/DeepSpeak/exported-dataset",
            'lipsynctimit': "/home/affshafee/T2430421/datasets/LipSyncTimit/Original Size"
        }
        
        self.model_save_dir = "/home/affshafee/T2430421/pipelines/novel_fusion/HMF_multi_dataset"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_save_dir, exist_ok=True)

        # --- Data Sampling ---
        self.use_sampling = False # Set to True for a quick debug run
        self.num_samples_per_class = 50

        # --- Visual Stream (MobileNetV3-Small) ---
        self.vis_image_size = (128, 128)
        self.vis_num_frames = 16
        self.vis_cnn_feature_dim = 576
        self.vis_mamba_d_model = 160

        # --- Audio Stream (MobileNetV3-Small) ---
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 16
        self.aud_chunk_duration = 5.0 / 16.0
        self.aud_n_mels = 128
        self.aud_cnn_feature_dim = 576
        self.aud_mamba_d_model = 160

        # --- Training Parameters ---
        self.batch_size = 64
        self.accumulation_steps = 4
        self.epochs = 50
        self.learning_rate = 5e-4
        self.weight_decay = 0.05
        self.patience = 10

config = Config()
print(f"✅ Configuration loaded. Using device: {config.device}")
print(f"🔥 Effective Batch Size: {config.batch_size * config.accumulation_steps}")
print(f"⚡ ARCHITECTURE: [MULTI-STREAM] Hierarchical Mamba Fusion (T={config.vis_num_frames})")
print(f"⚡ DATA: [MULTI-DATASET] Training on AVLips, DeepSpeak, and LipSyncTIMIT")

# --- 3. LABEL SMOOTHING LOSS ---
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)

# --- 4. DATA PROCESSING & DATASET CLASSES ---

# process_visual_stream is UNCHANGED. It just needs a video path.
def process_visual_stream(video_path: str, config: Config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release(); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < config.vis_num_frames:
        cap.release(); return None
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
    cap.release()
    if (len(local_frames_list) == config.vis_num_frames and
        len(context_frames_list) == config.vis_num_frames and
        len(global_frames_list) == config.vis_num_frames):
        return (np.stack(local_frames_list), 
                np.stack(context_frames_list), 
                np.stack(global_frames_list))
    else:
        return None

# [MULTI-DATASET CHANGE]: This function is HEAVILY modified.
def process_audio_stream(video_path: str, dataset_type: str, config: Config):
    try:
        audio_path = ""
        # --- Step 1: Get the audio file path or video path ---
        if dataset_type == 'avlips':
            # AVLips: Audio is in a separate .wav file
            parts = Path(video_path).parts
            audio_filename = Path(video_path).stem + ".wav"
            label_folder = parts[-2]
            base_data_dir = str(Path(video_path).parent.parent)
            audio_path = os.path.join(base_data_dir, "wav", label_folder, audio_filename)
        
        elif dataset_type in ['deepspeak', 'lipsynctimit']:
            # DeepSpeak/LipSyncTIMIT: Audio is stitched into the video file
            audio_path = video_path
        
        else:
            print(f"Unknown dataset_type: {dataset_type}")
            return None

        # --- Step 2: Load audio with Librosa ---
        # This now works for both .wav files and video files!
        y, sr = librosa.load(audio_path, sr=config.aud_sample_rate)
        
        # --- Step 3: Chunking and Mel Spectrogram (Unchanged) ---
        total_samples = int(config.aud_chunk_duration * config.aud_num_chunks * sr)
        if len(y) < total_samples:
            y = np.pad(y, (0, total_samples - len(y)), mode='constant')
        else:
            y = y[:total_samples]
        samples_per_chunk = int(config.aud_chunk_duration * sr)
        mel_list = []
        for i in range(config.aud_num_chunks):
            chunk = y[i*samples_per_chunk : (i+1)*samples_per_chunk]
            if len(chunk) == 0:
                chunk = np.zeros(samples_per_chunk)
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
        print(f"Error processing audio for {video_path} (Type: {dataset_type}). Error: {e}")
        return None

# [MULTI-DATASET CHANGE]: Modified __init__ and __getitem__
class DualStreamDataset(Dataset):
    def __init__(self, file_info, labels, config):
        # file_info is now a list of tuples: [(video_path, dataset_type), ...]
        self.file_info = file_info
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        # Unpack the file info
        video_path, dataset_type = self.file_info[idx]
        label = self.labels[idx]

        try:
            # --- Process Visual (Unchanged) ---
            visual_crops_hwc = process_visual_stream(video_path, self.config)
            if visual_crops_hwc is None:
                return None
            
            local_frames_hwc, context_frames_hwc, global_frames_hwc = visual_crops_hwc
            local_frames_tchw = local_frames_hwc.transpose(0, 3, 1, 2)
            context_frames_tchw = context_frames_hwc.transpose(0, 3, 1, 2)
            global_frames_tchw = global_frames_hwc.transpose(0, 3, 1, 2)
            
            # --- Process Audio (Now passes dataset_type) ---
            audio_mels = process_audio_stream(video_path, dataset_type, self.config)
            if audio_mels is None:
                return None

            audio_tensor = audio_mels.unsqueeze(1)
            
            visual_data_tuple = (local_frames_tchw, context_frames_tchw, global_frames_tchw)
            return (visual_data_tuple, audio_tensor), torch.tensor(label, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error in getitem for {video_path}: {e}")
            return None

# RAMCachedDataset is UNCHANGED. It's blissfully unaware of datasets.
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
# All model code is UNCHANGED from HMF_superior.py
class VisualStream_MobileNetV3Small(nn.Module):
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.cnn_features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.proj = nn.Linear(config.vis_cnn_feature_dim, config.vis_mamba_d_model)
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
        return temporal_out

class AudioStream_MobileNetV3Small(nn.Module):
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
        return temporal_out

class HierarchicalFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_stream_local = VisualStream_MobileNetV3Small(config)
        self.visual_stream_context = VisualStream_MobileNetV3Small(config)
        self.visual_stream_global = VisualStream_MobileNetV3Small(config)
        self.audio_stream = AudioStream_MobileNetV3Small(config)
        fusion_mamba_d_model = (3 * config.vis_mamba_d_model) + config.aud_mamba_d_model
        self.fusion_mamba = Mamba(
            d_model=fusion_mamba_d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fusion_mamba_dropout = nn.Dropout(0.2)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_mamba_d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)
        )
    def forward(self, visual_input_local, visual_input_context, visual_input_global, audio_input):
        local_seq = self.visual_stream_local(visual_input_local)
        context_seq = self.visual_stream_context(visual_input_context)
        global_seq = self.visual_stream_global(visual_input_global)
        audio_seq = self.audio_stream(audio_input)
        fused_seq = torch.cat((local_seq, context_seq, global_seq, audio_seq), dim=2)
        fusion_mamba_out = self.fusion_mamba(fused_seq)
        fusion_mamba_out = self.fusion_mamba_dropout(fusion_mamba_out)
        final_fused_vector = fusion_mamba_out[:, -1, :]
        return self.fusion_head(final_fused_vector)

# --- 6. UTILITY FUNCTIONS ---
# UNCHANGED
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

# --- 7. TRAINING AND EVALUATION ---
# UNCHANGED
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

# [MULTI-DATASET CHANGE]: This is the function with the BUG FIX
def load_all_datasets(paths_config, use_sampling, num_samples_per_class):
    all_files_info = [] # Will be list of (path, dataset_type)
    all_labels = []

    # --- 1. Load AVLips ---
    print(f"Loading AVLips from: {paths_config['avlips']}")
    avlips_real_dir = os.path.join(paths_config['avlips'], "0_real")
    avlips_fake_dir = os.path.join(paths_config['avlips'], "1_fake")
    avlips_real_files = [f for f in Path(avlips_real_dir).glob('*.mp4')]
    avlips_fake_files = [f for f in Path(avlips_fake_dir).glob('*.mp4')]
    
    for f in avlips_real_files:
        all_files_info.append((str(f), 'avlips'))
        all_labels.append(0)
    for f in avlips_fake_files:
        all_files_info.append((str(f), 'avlips'))
        all_labels.append(1)
    print(f"...found {len(avlips_real_files)} real, {len(avlips_fake_files)} fake.")

    # --- 2. Load DeepSpeak ---
    print(f"Loading DeepSpeak from: {paths_config['deepspeak']}")
    deepspeak_train_dir = Path(paths_config['deepspeak']) / 'train'
    # We'll just use the 'train' folder for now as it's the largest
    deepspeak_real_files = [f for f in (deepspeak_train_dir / 'real').glob('*.mp4')]
    deepspeak_fake_files = [f for f in (deepspeak_train_dir / 'fake').glob('*.mp4')]

    for f in deepspeak_real_files:
        all_files_info.append((str(f), 'deepspeak'))
        all_labels.append(0)
    for f in deepspeak_fake_files:
        all_files_info.append((str(f), 'deepspeak'))
        all_labels.append(1)
    print(f"...found {len(deepspeak_real_files)} real, {len(deepspeak_fake_files)} fake.")

    # --- 3. Load LipSyncTIMIT ---
    print(f"Loading LipSyncTIMIT from: {paths_config['lipsynctimit']}")
    timit_base_dir = Path(paths_config['lipsynctimit'])
    
    # [BUG FIX 1]: The real folder is 'RealVideo', not 'RealVideo-OriginalAudio'
    timit_real_dir = timit_base_dir / 'RealVideo'
    
    timit_fake_dirs = [
        timit_base_dir / 'FakeVideo-FakeAudio',
        timit_base_dir / 'FakeVideo-LSR2Audio',
        timit_base_dir / 'FakeVideo-OriginalAudio'
    ]
    
    # [BUG FIX 2]: Search recursively (**) for reals, just like we do for fakes.
    timit_real_files = [f for f in timit_real_dir.glob('**/*.mp4')]
    
    timit_fake_files = []
    for fake_dir in timit_fake_dirs:
        # Need to search recursively as fakes are in subfolders (Diff2Lip, etc.)
        timit_fake_files.extend([f for f in fake_dir.glob('**/*.mp4')])

    for f in timit_real_files:
        all_files_info.append((str(f), 'lipsynctimit'))
        all_labels.append(0)
    for f in timit_fake_files:
        all_files_info.append((str(f), 'lipsynctimit'))
        all_labels.append(1)
    print(f"...found {len(timit_real_files)} real, {len(timit_fake_files)} fake.")

    # --- 4. Final Sampling (if enabled) ---
    if use_sampling:
        print(f"🔥 Sampling {num_samples_per_class} videos per class from the combined dataset...")
        # Separate combined list into real and fake
        real_indices = [i for i, label in enumerate(all_labels) if label == 0]
        fake_indices = [i for i, label in enumerate(all_labels) if label == 1]
        
        # Check if we have enough samples
        num_real_available = len(real_indices)
        num_fake_available = len(fake_indices)
        
        if num_samples_per_class > num_real_available or num_samples_per_class > num_fake_available:
            print(f"Warning: Requested {num_samples_per_class} samples, but only found {num_real_available} real and {num_fake_available} fake.")
            # Adjust sample count to the minimum available
            num_samples_per_class = min(num_real_available, num_fake_available)
            print(f"Using {num_samples_per_class} samples per class instead.")
            if num_samples_per_class == 0:
                print("Error: No samples found for one or both classes.")
                return [], []

        # Sample indices
        sampled_real_indices = np.random.choice(real_indices, num_samples_per_class, replace=False)
        sampled_fake_indices = np.random.choice(fake_indices, num_samples_per_class, replace=False)
        
        # Re-build the lists
        sampled_files_info = [all_files_info[i] for i in sampled_real_indices] + [all_files_info[i] for i in sampled_fake_indices]
        sampled_labels = [0] * num_samples_per_class + [1] * num_samples_per_class
        
        print(f"Total files before sampling: {len(all_labels)}")
        print(f"Total files after sampling: {len(sampled_labels)}")
        return sampled_files_info, sampled_labels

    else:
        print("🎬 Using the full combined dataset.")
        print(f"Total files combined: {len(all_labels)}")
        return all_files_info, all_labels


def main():
    print("\n" + "="*80 + "\nSTEP 1: PREPARING FILE LISTS (MULTI-DATASET)\n" + "="*80)

    # [MULTI-DATASET CHANGE]: All file loading logic is now in this helper
    all_files_info, all_labels = load_all_datasets(
        config.dataset_paths, 
        config.use_sampling, 
        config.num_samples_per_class
    )
    
    if not all_files_info:
        print("🛑 ERROR: No files were loaded. Please check your dataset paths in the Config class.")
        return

    train_files_info, test_files_info, train_labels, test_labels = train_test_split(
        all_files_info, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
    val_files_info, test_files_info, val_labels, test_labels = train_test_split(
        test_files_info, test_labels, test_size=0.5, random_state=42, stratify=test_labels)

    print(f"Total Items: {len(all_labels)} | Train: {len(train_labels)} | Val: {len(val_labels)} | Test: {len(test_labels)}")

    print("\n" + "="*80 + "\nSTEP 2: PRE-LOADING & CACHING DATA INTO RAM\n" + "="*80)

    def collate_fn_skip_errors(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

    def cache_data(files_info, labels, desc):
        # [MULTI-DATASET CHANGE]: Pass file_info to the dataset
        dataset = DualStreamDataset(files_info, labels, config)
        
        # [MULTI-DATASET CHANGE]: Set num_workers=0 for caching to avoid file handle issues
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

    cached_train_data, cached_train_labels = cache_data(train_files_info, train_labels, "Train Set")
    cached_val_data, cached_val_labels = cache_data(val_files_info, val_labels, "Validation Set")
    cached_test_data, cached_test_labels = cache_data(test_files_info, test_labels, "Test Set")

    print(f"✅ Caching complete!")
    print(f" - Train samples: {len(cached_train_data)}")
    print(f" - Val samples: {len(cached_val_data)}")
    print(f" - Test samples: {len(cached_test_data)}")
    
    if len(cached_train_data) == 0:
        print("🛑 ERROR: Training cache is empty. Something went wrong during data loading/caching.")
        return

    print("\n" + "="*80 + "\nSTEP 3: CREATING FINAL DATALOADERS WITH ENHANCED AUGMENTATION\n" + "="*80)
    
    # We are NOT adding new augmentations yet, per our plan.
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
    
    num_workers = os.cpu_count() // 2
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"✅ DataLoaders created successfully (using {num_workers} workers).")

    print("\n" + "="*80 + "\nSTEP 4: BUILDING [MULTI-STREAM] HIERARCHICAL FUSION MODEL\n" + "="*80)
    
    model = HierarchicalFusionModel(config).to(config.device)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    print(f"📊 [MULTI-STREAM] HierarchicalFusionModel Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    print(f"   Parameters (Millions): {total_params/1e6:.3f}M")

    print("\n" + "="*80 + "\nSTEP 5: TRAINING [MULTI-STREAM] HIERARCHICAL FUSION MODEL\n" + "="*80)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    scaler = GradScaler()
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    model_path = os.path.join(config.model_save_dir, 'v1_hmf_multidataset_best.pth')
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
    
    if not os.path.exists(model_path):
        print("Warning: Best model path not found. Evaluating with last epoch model.")
    else:
        model.load_state_dict(torch.load(model_path))

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
    
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("ERROR: No test samples were evaluated. Cannot generate report.")
        return
        
    preds_binary = (all_preds > 0.5).astype(int)
    accuracy = (preds_binary == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_preds)
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else -1
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else -1
    loss_gap = final_train_loss - final_val_loss if final_train_loss != -1 else -1

    print(f"\n" + "="*80)
    print(f"📈 V1_HMF_MultiDataset: HIERARCHICAL FUSION (3-DATASET) - FINAL RESULTS")
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
    plt.title('V1_HMF_MultiDataset (Hierarchical Fusion): Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.model_save_dir, 'v1_hmf_multidataset_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ V1_HMF_MultiDataset (Hierarchical Fusion) ablation study completed!")
    print("📝 Record these metrics for your Excel comparison:")
    print(f"   - Variant: V1_HMF_MultiDataset")
    print(f"   - Size (MB): {model_size_mb:.2f}")
    print(f"   - Params (M): {total_params/1e6:.3f}")
    print(f"   - Accuracy: {accuracy*100:.2f}%")
    print(f"   - AUC: {auc_score*100:.2f}%")
    print(f"   - Loss Gap: {loss_gap:.4f}")

if __name__ == '__main__':
    main()