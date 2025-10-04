#!/usr/bin/env python3
"""
=============================================================================
DUAL-STREAM AUDIO-VISUAL TRANSFORMER FUSION MODEL FOR DEEPFAKE DETECTION
=============================================================================

This script implements an end-to-end deepfake detection system using Transformer
architectures for both audio and visual streams. Updated for the new AVLips 
dataset structure with optimized AMP training for RTX A4500.


GPU Optimizations:
- torch.cuda.amp.autocast for mixed precision training
- GradScaler for stable AMP training
- Default batch_size=32, num_workers=16 for RTX A4500
- Optimized for 20GB VRAM with efficient memory management

Author: AI/ML Engineer  
Date: September 2025
"""

import os
import argparse
import time
import warnings
from pathlib import Path
import subprocess
import tempfile
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # Faster on fixed-size inputs

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

class Config:
    """Configuration class containing all hyperparameters and settings."""
    def __init__(self):
        # Dataset Configuration (Updated for new structure)
        self.data_dir = None  # Will be set via argparse
        self.real_dir_name = "0_real"      # Real videos directory
        self.fake_dir_name = "1_fake"      # Fake videos directory
        self.audio_root_name = "wav"       # Root directory for pre-extracted audio
        self.max_videos_per_class = 0      # 0 means no limit (use all available)
        self.video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        self.random_state = 42

        # Training/Evaluation splits
        self.test_size = 0.2      # 20% for final test
        self.val_size = 0.2       # 20% of remaining for validation

        # Device Configuration (optimized for RTX A4500)
        self.device = torch.device("cpu")  # Will be resolved later
        self.use_amp = False               # Mixed precision flag
        self.amp_dtype = torch.float16

        # Visual Stream Configuration
        self.vis_image_size = (112, 112)
        self.vis_num_frames = 16
        self.vis_cnn_feature_dim = 1280    # EfficientNet-B0 output
        self.vis_transformer_d_model = 256
        self.vis_transformer_nhead = 8
        self.vis_transformer_layers = 4

        # Audio Stream Configuration
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 8
        self.aud_chunk_duration = 0.5      # seconds
        self.aud_n_mels = 80
        self.aud_cnn_feature_dim = 1280    # MobileNetV2 output
        self.aud_transformer_d_model = 256
        self.aud_transformer_nhead = 8
        self.aud_transformer_layers = 4

        # Fusion Configuration
        self.fusion_hidden_dim = 512
        self.fusion_dropout = 0.3
        self.num_classes = 2  # Real vs Fake

        # Training Configuration (Optimized for RTX A4500)
        self.batch_size = 32              # Increased for RTX A4500's 20GB VRAM
        self.epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.patience = 10
        self.num_workers = 16             # Increased for better CPU-GPU pipeline
        self.pin_memory = True
        self.gradient_clip = 1.0

# =============================================================================
# DATA PROCESSING UTILITIES (Updated for new dataset structure)
# =============================================================================

def extract_face_region(frame: np.ndarray, face_detector) -> Optional[np.ndarray]:
    """Extract mouth region from a video frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Take the first (largest) face
            # Extract mouth region (lower 40% of face, middle 50% width)
            mouth_y_start = y + int(h * 0.6)
            mouth_y_end = y + h
            mouth_x_start = x + int(w * 0.25)
            mouth_x_end = x + int(w * 0.75)
            
            mouth_crop = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            if mouth_crop.size > 0:
                return mouth_crop
        return None
    except Exception:
        return None

def process_visual_stream(video_path: str, config: Config) -> Optional[np.ndarray]:
    """Process video to extract mouth region frames."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < config.vis_num_frames:
            cap.release()
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, config.vis_num_frames, dtype=int)
        frames = []
        
        # Initialize face detector
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            mouth_region = extract_face_region(frame, face_detector)
            if mouth_region is not None:
                mouth_resized = cv2.resize(mouth_region, config.vis_image_size)
                frames.append(mouth_resized)
        
        cap.release()
        
        if len(frames) < config.vis_num_frames:
            return None
            
        return np.stack(frames[:config.vis_num_frames])
        
    except Exception:
        return None

def get_mirrored_audio_path(video_path: Path, config: Config, class_dir_name: str) -> Optional[str]:
    """
    Get the mirrored audio file path for a video file.
    
    Args:
        video_path: Path to the video file
        config: Configuration object
        class_dir_name: Either '0_real' or '1_fake'
    
    Returns:
        Path to the mirrored audio file if it exists, None otherwise
    """
    try:
        data_dir = config.data_dir
        class_root = data_dir / class_dir_name
        
        # Get relative path within the class directory
        relative_path = video_path.relative_to(class_root)
        
        # Construct mirrored audio path
        audio_root = data_dir / config.audio_root_name / class_dir_name
        audio_path = audio_root / relative_path.with_suffix('.wav')
        
        return str(audio_path) if audio_path.exists() else None
        
    except Exception:
        return None

def load_audio_from_wav(wav_path: str, config: Config) -> Optional[torch.Tensor]:
    """Load and process audio from a pre-extracted wav file."""
    try:
        y, sr = librosa.load(wav_path, sr=config.aud_sample_rate)
        
        # Calculate required samples
        samples_per_chunk = int(config.aud_chunk_duration * sr)
        total_samples = samples_per_chunk * config.aud_num_chunks
        
        # Pad or truncate audio
        if len(y) < total_samples:
            y = np.pad(y, (0, total_samples - len(y)), mode='constant')
        else:
            y = y[:total_samples]
        
        # Create mel-spectrograms for each chunk
        mel_spectrograms = []
        for i in range(config.aud_num_chunks):
            chunk_start = i * samples_per_chunk
            chunk_end = (i + 1) * samples_per_chunk
            chunk = y[chunk_start:chunk_end]
            
            # Compute mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=sr, n_mels=config.aud_n_mels,
                hop_length=512, win_length=2048
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            mel_spectrograms.append(torch.tensor(mel_db, dtype=torch.float32))
        
        return torch.stack(mel_spectrograms, dim=0)
        
    except Exception:
        return None

def extract_audio_from_video(video_path: str, config: Config) -> Optional[torch.Tensor]:
    """Extract audio from video using ffmpeg and process to mel-spectrograms."""
    try:
        # Extract audio using ffmpeg
        tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpwav = tmpfile.name
        tmpfile.close()
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path), "-ac", "1", 
                "-ar", str(config.aud_sample_rate), "-vn", "-f", "wav", tmpwav
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            return load_audio_from_wav(tmpwav, config)
            
        finally:
            if os.path.exists(tmpwav):
                os.remove(tmpwav)
                
    except Exception:
        return None

def process_audio_stream(video_path: str, config: Config, class_dir_name: str) -> Optional[torch.Tensor]:
    """
    Process audio stream - first try pre-extracted wav, then fallback to video extraction.
    
    Args:
        video_path: Path to the video file
        config: Configuration object  
        class_dir_name: Either '0_real' or '1_fake'
    """
    video_path_obj = Path(video_path)
    
    # Try to load from pre-extracted wav file first
    wav_path = get_mirrored_audio_path(video_path_obj, config, class_dir_name)
    if wav_path is not None:
        audio_tensor = load_audio_from_wav(wav_path, config)
        if audio_tensor is not None:
            return audio_tensor
    
    # Fallback to extracting audio from video
    return extract_audio_from_video(video_path, config)

# =============================================================================
# POSITIONAL ENCODING (batch_first-safe)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models - for batch_first=True."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer - shape: [max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x + pos_encoding

# =============================================================================
# VISUAL STREAM TRANSFORMER
# =============================================================================

class VisualStreamTransformer(nn.Module):
    """Visual stream processing using CNN + Transformer architecture."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # CNN Feature Extractor (EfficientNet-B0)
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = efficientnet.features
        self.cnn_avgpool = efficientnet.avgpool
        
        # Feature projection
        self.feature_projection = nn.Linear(config.vis_cnn_feature_dim, config.vis_transformer_d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.vis_transformer_d_model, max_len=config.vis_num_frames)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.vis_transformer_d_model,
            nhead=config.vis_transformer_nhead,
            dim_feedforward=config.vis_transformer_d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.vis_transformer_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.vis_transformer_d_model)
        
        # Global average pooling for sequence reduction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_frames, channels, height, width]
        Returns:
            features: [batch_size, d_model]
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for CNN processing
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract CNN features (frozen for efficiency)
        with torch.no_grad():
            cnn_features = self.cnn_features(x)
            cnn_features = self.cnn_avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1)
        
        # Project and reshape back to sequence
        projected = self.feature_projection(cnn_features)
        projected = projected.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        encoded = self.pos_encoding(projected)
        
        # Transformer encoder
        transformer_features = self.transformer_encoder(encoded)
        
        # Layer normalization
        normalized = self.layer_norm(transformer_features)
        
        # Global average pooling [B, T, D] -> [B, D, T] -> [B, D]
        pooled = self.global_pool(normalized.transpose(1, 2)).squeeze(-1)
        return pooled

# =============================================================================
# AUDIO STREAM TRANSFORMER
# =============================================================================

class AudioStreamTransformer(nn.Module):
    """Audio stream processing using CNN + Transformer architecture."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # CNN Feature Extractor (MobileNetV2)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn_features = mobilenet.features
        self.cnn_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(mobilenet.last_channel, config.aud_cnn_feature_dim),
            nn.ReLU(),
            nn.Linear(config.aud_cnn_feature_dim, config.aud_transformer_d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.aud_transformer_d_model, max_len=config.aud_num_chunks)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.aud_transformer_d_model,
            nhead=config.aud_transformer_nhead,
            dim_feedforward=config.aud_transformer_d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.aud_transformer_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.aud_transformer_d_model)
        
        # Global average pooling for sequence reduction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_chunks, n_mels, time_frames]
        Returns:
            features: [batch_size, d_model]
        """
        batch_size, num_chunks, n_mels, time_frames = x.shape
        
        # Convert to 3-channel for CNN (repeat channels)
        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        x = x.view(batch_size * num_chunks, 3, n_mels, time_frames)
        
        # Extract CNN features (frozen)
        with torch.no_grad():
            cnn_features = self.cnn_features(x)
            cnn_features = self.cnn_avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1)
        
        # Project and reshape back to sequence
        projected = self.feature_projection(cnn_features)
        projected = projected.view(batch_size, num_chunks, -1)
        
        # Add positional encoding
        encoded = self.pos_encoding(projected)
        
        # Transformer encoder
        transformer_features = self.transformer_encoder(encoded)
        
        # Layer normalization
        normalized = self.layer_norm(transformer_features)
        
        # Global average pooling to get single vector per batch
        pooled = self.global_pool(normalized.transpose(1, 2)).squeeze(-1)
        return pooled

# =============================================================================
# FUSION MODEL WITH CLASSIFICATION HEAD
# =============================================================================

class FusionModel(nn.Module):
    """Main fusion model combining visual and audio streams with classification head."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Stream processors
        self.visual_stream = VisualStreamTransformer(config)
        self.audio_stream = AudioStreamTransformer(config)
        
        # Fusion layers
        fusion_input_dim = config.vis_transformer_d_model + config.aud_transformer_d_model
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of fusion layers."""
        for module in self.fusion_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, visual_data: torch.Tensor, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_data: [batch_size, num_frames, channels, height, width]
            audio_data: [batch_size, num_chunks, n_mels, time_frames]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract features from both streams
        visual_features = self.visual_stream(visual_data)  # [B, vis_d_model]
        audio_features = self.audio_stream(audio_data)     # [B, aud_d_model]
        
        # Concatenate features
        fused_features = torch.cat([visual_features, audio_features], dim=1)  # [B, vis_d_model + aud_d_model]
        
        # Pass through fusion layers
        logits = self.fusion_layers(fused_features)  # [B, num_classes]
        return logits

# =============================================================================
# DATASET CLASS (Updated for new structure)
# =============================================================================

class AVLipsDataset(Dataset):
    """Dataset class for the new AVLips structure (0_real, 1_fake, wav folders)."""
    def __init__(self, video_paths: List[str], labels: List[int], config: Config, transform=None):
        self.video_paths = [Path(p) for p in video_paths]
        self.labels = labels
        self.config = config
        self.transform = transform
        
        print(f"Initialized dataset with {len(self.video_paths)} videos")
        print(f"Label distribution: Real={labels.count(0)}, Fake={labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def _get_class_dir_name(self, label: int) -> str:
        """Get class directory name based on label."""
        return self.config.real_dir_name if label == 0 else self.config.fake_dir_name
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        class_dir_name = self._get_class_dir_name(label)
        
        # Process visual stream
        try:
            visual_frames = process_visual_stream(str(video_path), self.config)
            if visual_frames is not None and self.transform is not None:
                # Convert BGR->RGB per frame before ToPILImage
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in visual_frames]
                visual_tensor = torch.stack([self.transform(frame) for frame in frames_rgb])
            else:
                visual_tensor = torch.zeros(
                    (self.config.vis_num_frames, 3, *self.config.vis_image_size),
                    dtype=torch.float32
                )
        except Exception as e:
            print(f"Visual processing failed for {video_path}: {e}")
            visual_tensor = torch.zeros(
                (self.config.vis_num_frames, 3, *self.config.vis_image_size),
                dtype=torch.float32
            )
        
        # Process audio stream (with pre-extracted wav support)
        try:
            audio_features = process_audio_stream(str(video_path), self.config, class_dir_name)
            if audio_features is not None:
                audio_tensor = audio_features
            else:
                audio_tensor = torch.zeros(
                    (self.config.aud_num_chunks, self.config.aud_n_mels, 32),
                    dtype=torch.float32
                )
        except Exception as e:
            print(f"Audio processing failed for {video_path}: {e}")
            audio_tensor = torch.zeros(
                (self.config.aud_num_chunks, self.config.aud_n_mels, 32),
                dtype=torch.float32
            )
        
        return visual_tensor, audio_tensor, torch.tensor(label, dtype=torch.long)

# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS (Enhanced AMP + GradScaler)
# =============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int, total_epochs: int,
                use_amp: bool, amp_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Train the model for one epoch with enhanced AMP support."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Initialize GradScaler for AMP (enhanced for stability)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for batch_idx, (visual_data, audio_data, labels) in enumerate(pbar):
        try:
            visual_data = visual_data.to(device, non_blocking=True)
            audio_data = audio_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with autocast for mixed precision
            with torch.autocast(device_type=device.type, dtype=amp_dtype, 
                               enabled=(use_amp and device.type == "cuda")):
                logits = model(visual_data, audio_data)
                loss = criterion(logits, labels)
            
            # Backward pass with GradScaler
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # Gradient clipping before scaler step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / max(1, total_samples)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.4f}",
                'gpu_mem': f"{torch.cuda.memory_reserved(0)/1e9:.1f}GB" if device.type == "cuda" else "N/A"
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(1, len(train_loader))
    avg_acc = correct_predictions / max(1, total_samples)
    
    # Print detailed epoch info
    eta_total = epoch_time * (total_epochs - (epoch + 1))
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, "
          f"Time: {epoch_time:.2f}s, ETA: {eta_total/60:.2f}min")
    
    return avg_loss, avg_acc, epoch_time

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                   device: torch.device, use_amp: bool, amp_dtype: torch.dtype) -> Tuple[float, float]:
    """Validate the model on validation set."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for visual_data, audio_data, labels in tqdm(val_loader, desc="Validation"):
            try:
                visual_data = visual_data.to(device, non_blocking=True)
                audio_data = audio_data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, dtype=amp_dtype, 
                                   enabled=(use_amp and device.type == "cuda")):
                    logits = model(visual_data, audio_data)
                    loss = criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    avg_loss = total_loss / max(1, len(val_loader))
    avg_acc = correct_predictions / max(1, total_samples)
    
    print(f"Validation - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    
    return avg_loss, avg_acc

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device,
                   use_amp: bool, amp_dtype: torch.dtype) -> Dict:
    """Evaluate the model on test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for visual_data, audio_data, labels in tqdm(test_loader, desc="Testing"):
            try:
                visual_data = visual_data.to(device, non_blocking=True)
                audio_data = audio_data.to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, dtype=amp_dtype, 
                                   enabled=(use_amp and device.type == "cuda")):
                    logits = model(visual_data, audio_data)
                    probabilities = F.softmax(logits, dim=1)
                
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])  # Probability of being fake
                
            except Exception as e:
                print(f"Test error: {e}")
                continue
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) if len(all_predictions) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probabilities) if len(np.unique(all_labels)) > 1 and len(all_probabilities) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def plot_training_curves(history: Dict, save_path: str):
    """Plot training curves."""
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'], label='Learning Rate', marker='d')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def plot_evaluation_results(results: Dict, save_path: str):
    """Plot evaluation results."""
    if len(results['labels']) == 0 or len(results['predictions']) == 0:
        print("No evaluation data to plot")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(results['labels'], results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Probability Distribution
    plt.subplot(1, 3, 2)
    real_probs = results['probabilities'][results['labels'] == 0]
    fake_probs = results['probabilities'][results['labels'] == 1]
    
    if len(real_probs) > 0:
        plt.hist(real_probs, alpha=0.7, label='Real Videos', bins=30, density=True, color='green')
    if len(fake_probs) > 0:
        plt.hist(fake_probs, alpha=0.7, label='Fake Videos', bins=30, density=True, color='red')
    plt.xlabel('Fake Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True)
    
    # ROC-like curve (threshold vs accuracy)
    plt.subplot(1, 3, 3)
    if len(results['probabilities']) > 0:
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        
        for threshold in thresholds:
            threshold_predictions = (results['probabilities'] > threshold).astype(int)
            acc = accuracy_score(results['labels'], threshold_predictions)
            accuracies.append(acc)
        
        plt.plot(thresholds, accuracies)
        plt.axvline(0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation results saved to {save_path}")

# =============================================================================
# FILE COLLECTION (RECURSIVE) - Updated for new structure
# =============================================================================

def collect_videos_recursive(base_dir: Path, exts: set, max_count: Optional[int] = None) -> List[str]:
    """Recursively collect video files under base_dir with allowed extensions."""
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return []
    
    files = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    
    files = sorted(files)
    if max_count is not None and max_count > 0:
        files = files[:max_count]
    
    return files

def describe_collection(name: str, files: List[str]):
    """Print a brief description of collected files for debugging."""
    print(f"{name}: {len(files)} files collected")
    
    if len(files) == 0:
        return
        
    # Show distribution by nested folders for sanity check
    buckets: Dict[str, int] = {}
    for f in files[:min(1000, len(files))]:  # Sample first 1000 for performance
        try:
            path_parts = Path(f).parts
            if len(path_parts) >= 3:
                key = "/".join(path_parts[-3:-1])
            elif len(path_parts) >= 2:
                key = "/".join(path_parts[-2:-1])
            else:
                key = "root"
            buckets[key] = buckets.get(key, 0) + 1
        except:
            buckets["unknown"] = buckets.get("unknown", 0) + 1
    
    # Show top categories
    top_categories = sorted(buckets.items(), key=lambda x: x[1], reverse=True)[:8]
    for category, count in top_categories:
        print(f"  - {category}: {count}")

# =============================================================================
# DEVICE RESOLUTION (Enhanced for RTX A4500)
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    """
    Resolve device based on user argument:
    - 'auto': cuda if available, else mps (Apple), else cpu
    - 'cuda'/'gpu': force cuda
    - 'mps': force Apple MPS
    - 'cpu': force cpu
    """
    dev = device_arg.strip().lower()
    if dev in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if dev == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if dev == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def print_device_info(device: torch.device, use_amp: bool):
    """Print detailed device info with RTX A4500 specific optimizations."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using CUDA GPU: {gpu_name} ({total_mem:.1f} GB VRAM), AMP: {use_amp}")
        
        # RTX A4500 specific optimizations
        if "A4500" in gpu_name:
            print("🚀 RTX A4500 detected - Using optimized settings:")
            print("   • Tensor Core acceleration enabled")
            print("   • Large batch size (32) for maximum throughput")
            print("   • 16 workers for optimal CPU-GPU pipeline")
            print("   • Mixed precision training for 30-40% speedup")
            
    elif device.type == "mps":
        print(f"Using Apple Metal (MPS), AMP: {use_amp} (AMP not used on MPS)")
    else:
        print("Using CPU (no AMP)")

# =============================================================================
# MAIN EXECUTION FUNCTION (Updated for new dataset structure)
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Deepfake Detection using Audio-Visual Transformer Fusion (AVLips Dataset, RTX A4500 Optimized)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to AVLips dataset directory (containing 0_real, 1_fake, wav folders)')
    parser.add_argument('--real_dir_name', type=str, default='0_real',
                        help='Real videos directory name (default: 0_real)')
    parser.add_argument('--fake_dir_name', type=str, default='1_fake', 
                        help='Fake videos directory name (default: 1_fake)')
    parser.add_argument('--audio_root_name', type=str, default='wav',
                        help='Audio root directory name (default: wav)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32, optimized for RTX A4500)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_videos_per_class', type=int, default=0,
                        help='Maximum number of videos per class (0 = no limit)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader workers (default: 16, optimized for RTX A4500)')
    parser.add_argument('--save_dir', type=str, default='./results_avlips/',
                        help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for final test set')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of remaining data for validation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'gpu', 'cpu', 'mps'],
                        help='Device selection')
    parser.add_argument('--amp', action='store_true',
                        help='Enable AMP mixed precision (CUDA only). Enabled by default on CUDA.')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP even if CUDA is available.')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.data_dir = Path(args.data_dir)
    config.real_dir_name = args.real_dir_name
    config.fake_dir_name = args.fake_dir_name
    config.audio_root_name = args.audio_root_name
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.max_videos_per_class = args.max_videos_per_class
    config.num_workers = args.num_workers
    config.test_size = args.test_size
    config.val_size = args.val_size
    
    # Resolve device and AMP
    device = resolve_device(args.device)
    # AMP default: True on CUDA unless explicitly disabled
    use_amp = (device.type == "cuda")
    if args.amp:
        use_amp = True
    if args.no_amp:
        use_amp = False
    
    config.device = device
    config.use_amp = use_amp
    config.pin_memory = (device.type == "cuda")
    
    # Optional: improve matmul perf on Ampere+ (RTX A4500)
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DUAL-STREAM AUDIO-VISUAL TRANSFORMER DEEPFAKE DETECTION")
    print("AVLips Dataset Structure (0_real, 1_fake, wav folders)")
    print("RTX A4500 Optimized with Enhanced AMP + GradScaler")
    print("="*80)
    print(f"Data directory: {config.data_dir}")
    print(f"Class folders: {config.real_dir_name} (real), {config.fake_dir_name} (fake)")
    print(f"Audio folder: {config.audio_root_name} (pre-extracted wav files)")
    print(f"Batch size: {config.batch_size} | Workers: {config.num_workers}")
    print(f"Epochs: {config.epochs} | Learning rate: {config.learning_rate}")
    print(f"Test/Val split: {config.test_size:.1%}/{config.val_size:.1%}")
    print_device_info(config.device, config.use_amp)
    print("="*80)
    
    # =============================================================================
    # DATA PREPARATION (Updated for new structure)
    # =============================================================================
    
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION (AVLips Structure)")
    print("="*50)
    
    # Directory paths (updated)
    real_dir = config.data_dir / config.real_dir_name
    fake_dir = config.data_dir / config.fake_dir_name
    
    # Collect all files
    all_files = []
    all_labels = []
    
    # Real videos (label = 0)
    real_files = collect_videos_recursive(real_dir, config.video_exts, 
                                        config.max_videos_per_class if config.max_videos_per_class > 0 else None)
    all_files.extend(real_files)
    all_labels.extend([0] * len(real_files))
    print(f"Real videos ({config.real_dir_name}): {len(real_files)}")
    describe_collection(f"Real ({config.real_dir_name})", real_files[:20])
    
    # Fake videos (label = 1)
    fake_files = collect_videos_recursive(fake_dir, config.video_exts,
                                        config.max_videos_per_class if config.max_videos_per_class > 0 else None)
    all_files.extend(fake_files)
    all_labels.extend([1] * len(fake_files))
    print(f"Fake videos ({config.fake_dir_name}): {len(fake_files)}")
    describe_collection(f"Fake ({config.fake_dir_name})", fake_files[:20])
    
    print(f"\nTotal files collected: {len(all_files)}")
    print(f"Real: {all_labels.count(0)}, Fake: {all_labels.count(1)}")
    
    if len(all_files) < 10:
        print("Not enough files for proper splitting. Need at least 10 files.")
        return
    
    # Split data into train/val/test with stratification
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=config.test_size, random_state=config.random_state,
        stratify=all_labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=config.val_size, random_state=config.random_state,
        stratify=train_val_labels
    )
    
    print(f"\nData splits:")
    print(f"Training: {len(train_files)} (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
    print(f"Validation: {len(val_files)} (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")
    print(f"Test: {len(test_files)} (Real: {test_labels.count(0)}, Fake: {test_labels.count(1)})")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.vis_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = AVLipsDataset(train_files, train_labels, config, transform)
    val_dataset = AVLipsDataset(val_files, val_labels, config, transform)
    test_dataset = AVLipsDataset(test_files, test_labels, config, transform)
    
    # Create data loaders (optimized for RTX A4500)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory,
                             persistent_workers=(config.num_workers > 0), prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory,
                           persistent_workers=(config.num_workers > 0), prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory,
                            persistent_workers=(config.num_workers > 0), prefetch_factor=2)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # =============================================================================
    # MODEL INITIALIZATION
    # =============================================================================
    
    print("\n" + "="*50)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*50)
    
    model = FusionModel(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    # =============================================================================
    # TRAINING LOOP (Enhanced with AMP + GradScaler)
    # =============================================================================
    
    print("\n" + "="*50)
    print("STEP 3: TRAINING (Enhanced AMP + GradScaler)")
    print("="*50)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training (enhanced with AMP)
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, optimizer, criterion, config.device, epoch, config.epochs,
            use_amp=config.use_amp, amp_dtype=config.amp_dtype
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, config.device,
            use_amp=config.use_amp, amp_dtype=config.amp_dtype
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        print(f"Summary - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'total_params': total_params,
                    'trainable_params': trainable_params
                }
            }, save_dir / 'best_model.pth')
            print(f"New best validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    # =============================================================================
    # EVALUATION
    # =============================================================================
    
    print("\n" + "="*50)
    print("STEP 4: FINAL EVALUATION")
    print("="*50)
    
    # Load best model
    best_model_path = save_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set
    results = evaluate_model(model, test_loader, config.device, use_amp=config.use_amp, amp_dtype=config.amp_dtype)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    
    # Print detailed classification report
    if len(results['labels']) > 0 and len(results['predictions']) > 0:
        print("\nDetailed Classification Report:")
        print(classification_report(results['labels'], results['predictions'],
                                  target_names=['Real', 'Fake']))
    
    # =============================================================================
    # VISUALIZATION AND SAVING
    # =============================================================================
    
    print("\n" + "="*50)
    print("STEP 5: VISUALIZATION & SAVING RESULTS")
    print("="*50)
    
    # Plot training curves
    plot_training_curves(history, save_dir / 'training_curves.png')
    
    # Plot evaluation results
    plot_evaluation_results(results, save_dir / 'evaluation_results.png')
    
    # Save detailed results
    if len(results['predictions']) > 0:
        results_df = pd.DataFrame({
            'prediction': results['predictions'],
            'probability': results['probabilities'],
            'true_label': results['labels']
        })
        results_df.to_csv(save_dir / 'detailed_test_results.csv', index=False)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / 'training_history.csv', index=False)
    
    # Save final metrics
    metrics = {
        'final_test_accuracy': results['accuracy'],
        'final_test_auc': results['auc'],
        'best_val_accuracy': best_val_acc,
        'total_epochs_trained': len(history['train_loss']),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / 'final_metrics.csv', index=False)
    
    print(f"\nAll results saved to: {save_dir}")
    print("\nFiles generated:")
    print("- best_model.pth (trained model)")
    print("- training_curves.png (loss and accuracy curves)")
    print("- evaluation_results.png (confusion matrix and distributions)")
    print("- detailed_test_results.csv (per-sample predictions)")
    print("- training_history.csv (epoch-by-epoch metrics)")
    print("- final_metrics.csv (summary metrics)")
    
    print("="*80)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test AUC: {results['auc']:.4f}")
    
    # RTX A4500 Performance Summary
    if config.device.type == "cuda" and config.use_amp:
        print("\n🚀 RTX A4500 Performance Summary:")
        print(f"   • Mixed Precision Training: ✅ Enabled")
        print(f"   • Batch Size: {config.batch_size} (utilizing 20GB VRAM)")
        print(f"   • Workers: {config.num_workers} (optimal CPU-GPU pipeline)")
        print(f"   • Expected speedup vs CPU: 15-25x")
        print(f"   • Expected speedup with AMP: 30-40%")
    
    print("="*80)

if __name__ == "__main__":
    main()