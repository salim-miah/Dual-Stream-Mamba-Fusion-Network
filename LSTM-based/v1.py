"""
14.1_lstm_v1cnn_fixed.py

Changes vs your original 14.1_lstm.py:
- Visual CNN: MobileNetV3-Small (features + avgpool)
- Audio CNN:  MobileNetV3-Small (features + avgpool) + channel repeat to 3 in forward()
- vis_cnn_feature_dim = aud_cnn_feature_dim = 576
- End-of-run metrics printed in V1-style (Size MB, Params M, Accuracy, AUC, Loss Gap)
- Startup banner moved under __main__ guard to avoid duplicate prints on Windows
"""

# --- 1. IMPORTS ---

import os
import cv2
import time
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


# --- 2. CONFIGURATION ---

class Config:
    def __init__(self):
        # --- Paths and Device ---
        self.data_dir = r"C:\Thesis\Thesis Team Deepfake Detection\Datasets\AVLips v1.0\AVLips"
        self.model_save_dir = r"C:\Thesis\Thesis Team Deepfake Detection\Test Runs\LSTM based v2\models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # --- Data Sampling (for quick debugging) ---
        self.use_sampling = True
        self.num_samples_per_class = 2000  # change this smaller for smoke tests
        
        # --- Visual Stream ---
        self.vis_image_size = (128, 128)
        self.vis_num_frames = 16
        self.vis_cnn_feature_dim = 576  # MobileNetV3-Small output
        self.vis_mamba_d_model = 128
        
        # --- Audio Stream ---
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 5
        self.aud_chunk_duration = 1.0
        self.aud_n_mels = 128
        self.aud_cnn_feature_dim = 576  # MobileNetV3-Small output
        self.aud_mamba_d_model = 128
        
        # --- Recurrent Temporal Modeling (LSTM replacement for Mamba) ---
        self.lstm_num_layers = 2
        self.lstm_bidirectional = True
        self.lstm_dropout = 0.2

        # --- Training Parameters (PHASE 1 CHANGES) ---
        self.batch_size = 32
        self.accumulation_steps = 4  # Effective batch size 128
        self.epochs = 25
        self.learning_rate = 5e-4
        self.weight_decay = 0.05
        
        # --- DataLoader ---
        self.num_workers = 4  # set 0 for debugging on Windows to avoid dup prints

        # --- Loss ---
        self.label_smoothing = 0.05


# --- 3. LABEL-SMOOTHED BCE (binary) ---

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # logits: (B,1), targets: (B,1)
        targets = targets.clamp(0, 1)
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


# --- 4. DATASETS ---

class DualStreamDataset(Dataset):
    """
    Reads video file paths and labels, returns:
      ((visual_tensor[T,C,H,W], audio_tensor[T,C,H,W]), label)
    Visual: sampled frames
    Audio: mel-spectrogram chunks stacked as pseudo-images
    """
    def __init__(self, file_paths, labels, config: Config, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def _read_video_frames(self, video_path):
        """
        Reads self.vis_num_frames frames resized to vis_image_size, RGB (C=3)
        Returns np.array shape [T, C, H, W]
        """
        T = self.config.vis_num_frames
        H, W = self.config.vis_image_size
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (T * 2)
        idxs = np.linspace(0, max(0, total-1), T).astype(int)
        ok_total = 0
        for i, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                # try reading next
                ok, frame = cap.read()
            if not ok:
                # pad with zeros if we fail
                frame = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                ok_total += 1
            frames.append(frame)
        cap.release()
        # to [T,C,H,W]
        arr = np.stack(frames, axis=0)  # [T,H,W,3]
        arr = arr.transpose(0, 3, 1, 2)  # [T,3,H,W]
        return arr

    def _read_audio_mels(self, video_path):
        """
        Extract audio from companion audio track if accessible, else load wav with same basename.
        Here we assume audio path is video_path with .wav or audio is embedded (simplified).
        Returns torch.Tensor [T, 1, H, W] -> we will repeat to 3 channels in the model.
        """
        sr = self.config.aud_sample_rate
        T = self.config.aud_num_chunks
        chunk_sec = self.config.aud_chunk_duration
        mel_bins = self.config.aud_n_mels

        # try sidecar .wav
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        if not os.path.exists(audio_path):
            # fallback: no audio file -> return zeros
            H, W = mel_bins, int(chunk_sec * 32)  # arbitrary width for plotting
            return torch.zeros((T, 1, H, W), dtype=torch.float32)

        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        total_needed = int(T * chunk_sec * sr)
        if len(y) < total_needed:
            pad = total_needed - len(y)
            y = np.pad(y, (0, pad), mode='constant')

        # chunk and make mels
        samples_per_chunk = int(chunk_sec * sr)
        mel_list = []
        for i in range(T):
            chunk = y[i*samples_per_chunk : (i+1)*samples_per_chunk]
            mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=mel_bins)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            # simple normalization
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            mel_t = torch.tensor(mel_db, dtype=torch.float32)[None, ...]  # [1,H,W]
            mel_list.append(mel_t)
        # [T,1,H,W]
        return torch.stack(mel_list, dim=0)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        # Visual
        visual_frames_np = self._read_video_frames(path)  # [T,3,H,W]
        if self.transform is not None:
            augmented_frames = []
            # ToPILImage expects [H,W,C]
            for frame_np in visual_frames_np:
                frame_hwc = frame_np.transpose(1, 2, 0)
                augmented_frames.append(self.transform(frame_hwc))
            visual_tensor = torch.stack(augmented_frames)  # [T,3,H,W]
        else:
            visual_tensor = torch.from_numpy(visual_frames_np).float()
        # Audio
        audio_tensor = self._read_audio_mels(path)  # [T,1,H,W]

        return (visual_tensor, audio_tensor), label


class RAMCachedDataset(Dataset):
    """ Same interface as DualStreamDataset, but uses pre-cached (visual, audio) tensors """
    def __init__(self, cached_data, cached_labels, transform=None):
        self.cached_data = cached_data
        self.cached_labels = cached_labels
        self.transform = transform

    def __len__(self):
        return len(self.cached_labels)

    def __getitem__(self, idx):
        visual_frames_np, audio_tensor = self.cached_data[idx]  # visual as numpy or tensor
        label = self.cached_labels[idx]
        if self.transform is not None:
            augmented_frames = []
            for frame_np in visual_frames_np:
                frame_hwc = frame_np.transpose(1, 2, 0)
                augmented_frames.append(self.transform(frame_hwc))
            visual_tensor = torch.stack(augmented_frames)
        else:
            if isinstance(visual_frames_np, np.ndarray):
                visual_tensor = torch.from_numpy(visual_frames_np).float()
            else:
                visual_tensor = visual_frames_np.float()
        return (visual_tensor, audio_tensor), label


# --- 5. MODEL (V1 CNNs + LSTM temporal) ---

class VisualStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.cnn_features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.proj = nn.Linear(config.vis_cnn_feature_dim, config.vis_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=config.vis_mamba_d_model,
            hidden_size=config.vis_mamba_d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.temporal_dropout = nn.Dropout(0.2)
        self.out_dim = config.vis_mamba_d_model * (2 if config.lstm_bidirectional else 1)

    def forward(self, x):
        # x: [B, T, 3, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.avgpool(self.cnn_features(x)).view(b, t, -1)   # [B,T,576]
        projected = self.proj_dropout(self.proj(features))             # [B,T,Dm]
        temporal, _ = self.lstm(projected)                             # [B,T,H]
        temporal = self.temporal_dropout(temporal)
        return temporal[:, -1, :]  # [B,H]


class AudioStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.cnn_features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.proj = nn.Linear(config.aud_cnn_feature_dim, config.aud_mamba_d_model)
        self.proj_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=config.aud_mamba_d_model,
            hidden_size=config.aud_mamba_d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.temporal_dropout = nn.Dropout(0.2)
        self.out_dim = config.aud_mamba_d_model * (2 if config.lstm_bidirectional else 1)

    def forward(self, x):
        # x: [B, T, C, H, W]; for audio C=1 (mel), need 3 channels for MobileNet
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # -> [B*T, 3, H, W]
        features = self.avgpool(self.cnn_features(x)).view(b, t, -1)   # [B,T,576]
        projected = self.proj_dropout(self.proj(features))             # [B,T,Dm]
        temporal, _ = self.lstm(projected)                             # [B,T,H]
        temporal = self.temporal_dropout(temporal)
        return temporal[:, -1, :]  # [B,H]


class FusionModel(nn.Module):
    """
    Concatenate final visual+audio LSTM features -> MLP head -> sigmoid (binary)
    """
    def __init__(self, config):
        super().__init__()
        self.visual_stream = VisualStream(config)
        self.audio_stream = AudioStream(config)
        fused_dim = self.visual_stream.out_dim + self.audio_stream.out_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # BCE logits
        )

    def forward(self, visual_input, audio_input):
        visual_features = self.visual_stream(visual_input)  # [B,Hv]
        audio_features = self.audio_stream(audio_input)     # [B,Ha]
        fused = torch.cat([visual_features, audio_features], dim=1)
        return self.fusion_head(fused)  # [B,1]


# --- 6. UTILITY FUNCTIONS ---

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


# --- 7. TRAINING / VALIDATION ---

def train_one_epoch(model, loader, optimizer, criterion, scaler, config: Config):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for step, ((visual_data, audio_data), labels) in enumerate(pbar):
        visual_data = visual_data.to(config.device, non_blocking=True)
        audio_data = audio_data.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()

        with autocast():
            outputs = model(visual_data, audio_data)  # [B,1]
            loss = criterion(outputs, labels) / config.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.accumulation_steps
        pbar.set_postfix({'loss': f"{loss.item() * config.accumulation_steps:.4f}"})

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, config: Config):
    model.eval()
    total_loss = 0.0
    for (visual_data, audio_data), labels in tqdm(loader, desc="Validating"):
        visual_data = visual_data.to(config.device, non_blocking=True)
        audio_data = audio_data.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True).unsqueeze(1).float()
        with autocast():
            outputs = model(visual_data, audio_data)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


# --- 8. MAIN PIPELINE ---

def main():
    config = Config()

    print("✅ Libraries imported successfully.")
    print(f"✅ Configuration loaded. Using device: {config.device.type}")
    print(f"🔥 Effective Batch Size: {config.batch_size * config.accumulation_steps}")

    # --- Step 1: Prepare File Lists ---
    print("\n" + "="*80 + "\nSTEP 1: PREPARING FILE LISTS\n" + "="*80)
    real_dir = os.path.join(config.data_dir, "0_real")
    fake_dir = os.path.join(config.data_dir, "1_fake")

    all_real = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    all_fake = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')]

    if config.use_sampling:
        sample_n = min(config.num_samples_per_class, len(all_real), len(all_fake))
        print(f"🔥 Sampling {sample_n} videos per class...")
        real_files = np.random.choice(all_real, sample_n, replace=False).tolist()
        fake_files = np.random.choice(all_fake, sample_n, replace=False).tolist()
    else:
        print("🎬 Using the full dataset.")
        real_files, fake_files = all_real, all_fake

    all_paths = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    # split: train / (val+test); then val/test
    train_files, vt_files, train_labels, vt_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        vt_files, vt_labels, test_size=0.5, random_state=43, stratify=vt_labels
    )
    print(f"Total Videos: {len(all_paths)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # --- Step 2: Pre-load & Cache Data into RAM ---
    print("\n" + "="*80 + "\nSTEP 2: PRE-LOADING & CACHING DATA INTO RAM\n" + "="*80)

    def cache_split(paths, labels, desc):
        ds = DualStreamDataset(paths, labels, config, transform=None)
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )  # warm, deterministic
        cached_data, cached_lbls = [], []
        for (v, a), y in tqdm(loader, desc=f"Caching {desc}"):
            # store as numpy for visuals & tensor for audio (saves a bit of RAM)
            cached_data.append((v.squeeze(0).numpy(), a.squeeze(0)))
            cached_lbls.append(int(y.item()))
        return cached_data, cached_lbls

    cached_train_data, cached_train_labels = cache_split(train_files, train_labels, "Train Set")
    cached_val_data, cached_val_labels = cache_split(val_files, val_labels, "Val Set")
    cached_test_data, cached_test_labels = cache_split(test_files, test_labels, "Test Set")

    print("✅ Caching complete!")
    print(f" - Train samples: {len(cached_train_data)}")
    print(f" - Val samples:   {len(cached_val_data)}")
    print(f" - Test samples:  {len(cached_test_data)}")

    # --- Step 3: Dataloaders with transforms ---
    print("\n" + "="*80 + "\nSTEP 3: CREATING FINAL DATALOADERS WITH ENHANCED AUGMENTATION\n" + "="*80)

    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RAMCachedDataset(cached_train_data, cached_train_labels, transform=train_transform)
    val_dataset   = RAMCachedDataset(cached_val_data,   cached_val_labels,   transform=val_test_transform)
    test_dataset  = RAMCachedDataset(cached_test_data,  cached_test_labels,  transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )

    # --- Step 4: Model / Optim / Loss ---
    print("\n" + "="*80 + "\nSTEP 4: TRAINING\n" + "="*80)
    model = FusionModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler()
    criterion = LabelSmoothingBCELoss(config.label_smoothing)

    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)

    print(f"📊 Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Approx Size: {model_size_mb:.2f} MB")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        print(f"\n-- Epoch {epoch}/{config.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config)
        val_loss   = validate_one_epoch(model, val_loader, criterion, config)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # --- Step 5: Evaluation on Test Set ---
    print("\n" + "="*80 + "\nSTEP 5: EVALUATION ON TEST SET\n" + "="*80)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (visual_data, audio_data), labels in tqdm(test_loader, desc="Testing"):
            visual_data = visual_data.to(config.device, non_blocking=True)
            audio_data  = audio_data.to(config.device, non_blocking=True)
            logits = model(visual_data, audio_data)  # [B,1]
            probs = torch.sigmoid(logits).squeeze(1)  # [B]
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds_binary = (all_preds > 0.5).astype(int)

    accuracy = (preds_binary == all_labels).mean()
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc_score = 0.0

    print(f"\n📈 PHASE 1 RESULTS:")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 AUC Score: {auc_score:.4f}")
    print(f"🔄 Training/Validation Loss Gap: {history['train_loss'][-1] - history['val_loss'][-1]:.4f}")

    print("\n📋 Classification Report:")
    try:
        print(classification_report(all_labels, preds_binary, target_names=['Real (0)', 'Fake (1)']))
    except Exception:
        print(classification_report(all_labels, preds_binary))

    # --- Step 6: Plot Loss Curves ---
    print("\n" + "="*80 + "\nSTEP 6: VISUALIZING TRAINING HISTORY\n" + "="*80)
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title('Phase 1: Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(config.model_save_dir, 'lstm_v1cnn_loss_curve.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved loss curve → {out_png}")

    print("\n✅ Phase 1 training completed!")
    print("🔍 Check if the validation loss gap has improved compared to your original results.")
    print("📝 If overfitting persists, proceed to Phase 2 modifications.")

    # === V1-style summary (matches V1.py fields) ===
    try:
        loss_gap = history['train_loss'][-1] - history['val_loss'][-1]
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80)
        print(f"   - Variant: V1")
        print(f"   - Size (MB): {model_size_mb:.2f}")
        print(f"   - Params (M): {total_params/1e6:.3f}")
        print(f"   - Accuracy: {accuracy*100:.2f}%")
        print(f"   - AUC: {auc_score*100:.2f}%")
        print(f"   - Loss Gap: {loss_gap:.4f}")
    except Exception as e:
        print("[WARN] Could not print V1-style summary:", e)


if __name__ == '__main__':
    # For Windows worker spawn; keep all prints/setup under this guard to avoid duplicates
    # If you're doing tiny smoke tests, you can also set Config.num_workers = 0 above.
    main()
