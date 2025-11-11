# -*- coding: utf-8 -*-

"""
CROSS-DATASET EVALUATION: V1d (DeepSpeak) → KoDF
This script evaluates a model trained on DeepSpeak against the KoDF dataset.
"""

# --- 1. IMPORTS ---
import os
import cv2
import time
import torch
import librosa
import numpy as np
import torch.nn as nn
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
print("✅ Libraries imported successfully.")

# --- 2. CONFIGURATION ---
class Config:
    def __init__(self):
        # --- Paths ---
        self.kodf_base_dir = "/home/affshafee/T2430421/datasets/KoDF"
        self.model_path = "/home/affshafee/T2430421/pipelines/V1d_on_DS/v1d_deepspeak_best.pth"
        self.output_dir = "/home/affshafee/T2430421/pipelines/V1d_DS_to_KoDF"
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Sampling Strategy ---
        self.num_samples_per_class = 3000  # 3000 real + 3000 fake = 6000 videos
        self.use_sampling = True

        # --- Visual Stream (MobileNetV3-Small) ---
        self.vis_image_size = (128, 128)
        self.vis_num_frames = 16
        self.vis_cnn_feature_dim = 576
        self.vis_mamba_d_model = 160

        # --- Audio Stream (MobileNetV3-Small) ---
        self.aud_sample_rate = 16000
        self.aud_num_chunks = 5
        self.aud_chunk_duration = 1.0
        self.aud_n_mels = 128
        self.aud_cnn_feature_dim = 576
        self.aud_mamba_d_model = 160

        # --- Evaluation Parameters ---
        self.batch_size = 32  # Reduced for evaluation
        self.num_workers = 8

config = Config()
print(f"✅ Configuration loaded. Using device: {config.device}")
print(f"📊 Sampling {config.num_samples_per_class} videos per class from KoDF")
print(f"🔥 Model: V1d trained on DeepSpeak")

# --- 3. AUDIO EXTRACTION FROM VIDEO ---
def extract_audio_from_video(video_path: str, output_path: str, config: Config):
    """Extract audio from video using FFmpeg."""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(config.aud_sample_rate), '-ac', '1', '-y', output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False

# --- 4. DATA PROCESSING FUNCTIONS ---
def process_visual_stream(video_path: str, config: Config):
    """Extract and process visual frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < config.vis_num_frames:
        cap.release()
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
            mouth_crop = frame[y + int(h * 0.6):y + h, x + int(w * 0.25):x + int(w * 0.75)]
            if mouth_crop.size > 0:
                resized_crop = cv2.resize(mouth_crop, config.vis_image_size)
                resized_crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                frames.append(resized_crop_rgb)

    cap.release()
    return np.stack(frames) if len(frames) == config.vis_num_frames else None

def process_audio_stream(video_path: str, config: Config):
    """Extract and process audio from video."""
    try:
        # Create temporary audio file
        temp_audio = video_path.replace('.mp4', '_temp.wav')

        if not extract_audio_from_video(video_path, temp_audio, config):
            return None

        # Load and process audio
        y, sr = librosa.load(temp_audio, sr=config.aud_sample_rate)

        # Clean up temp file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

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
    except Exception as e:
        return None

# --- 5. DATASET CLASS ---
class KoDFDataset(Dataset):
    def __init__(self, file_paths, labels, config, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Process visual stream
            visual_frames_hwc = process_visual_stream(video_path, self.config)
            if visual_frames_hwc is None:
                return None

            # Process audio stream
            audio_mels = process_audio_stream(video_path, self.config)
            if audio_mels is None:
                return None

            # Transform visual frames
            if self.transform:
                transformed_frames = []
                for frame in visual_frames_hwc:
                    transformed_frames.append(self.transform(frame))
                visual_tensor = torch.stack(transformed_frames)
            else:
                visual_frames_tchw = visual_frames_hwc.transpose(0, 3, 1, 2)
                visual_tensor = torch.from_numpy(visual_frames_tchw).float()

            audio_tensor = audio_mels.unsqueeze(1)

            return (visual_tensor, audio_tensor), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            return None

# --- 6. MODEL ARCHITECTURE ---
class VisualStream_MobileNetV3Small(nn.Module):
    """V1: MobileNetV3-Small for visual feature extraction."""
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
        return temporal_out[:, -1, :]

class AudioStream_MobileNetV3Small(nn.Module):
    """V1: MobileNetV3-Small for audio feature extraction."""
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
        return temporal_out[:, -1, :]

class FusionModel_V1(nn.Module):
    """V1: Fusion model with MobileNetV3-Small CNNs."""
    def __init__(self, config):
        super().__init__()
        self.visual_stream = VisualStream_MobileNetV3Small(config)
        self.audio_stream = AudioStream_MobileNetV3Small(config)
        fusion_input_dim = config.vis_mamba_d_model + config.aud_mamba_d_model
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1)
        )

    def forward(self, visual_input, audio_input):
        visual_features = self.visual_stream(visual_input)
        audio_features = self.audio_stream(audio_input)
        fused_features = torch.cat((visual_features, audio_features), dim=1)
        return self.fusion_head(fused_features)

# --- 7. FILE DISCOVERY FUNCTION ---
def discover_kodf_videos(base_dir, num_samples_per_class):
    """Recursively discover videos in nested KoDF structure."""
    print("\n" + "="*80)
    print("DISCOVERING KODF VIDEOS")
    print("="*80)

    real_dir = os.path.join(base_dir, "original_videos")
    fake_dir = os.path.join(base_dir, "synthesized")

    print(f"🔍 Searching for real videos in: {real_dir}")
    real_videos = []
    for root, dirs, files in os.walk(real_dir):
        for file in files:
            if file.endswith('.mp4'):
                real_videos.append(os.path.join(root, file))

    print(f"🔍 Searching for fake videos in: {fake_dir}")
    fake_videos = []
    for root, dirs, files in os.walk(fake_dir):
        for file in files:
            if file.endswith('.mp4'):
                fake_videos.append(os.path.join(root, file))

    print(f"\n📊 Total videos found:")
    print(f"   Real videos: {len(real_videos):,}")
    print(f"   Fake videos: {len(fake_videos):,}")

    # Sample videos
    if len(real_videos) > num_samples_per_class:
        real_videos = np.random.choice(real_videos, num_samples_per_class, replace=False).tolist()
        print(f"\n✂️ Sampled {num_samples_per_class:,} real videos")

    if len(fake_videos) > num_samples_per_class:
        fake_videos = np.random.choice(fake_videos, num_samples_per_class, replace=False).tolist()
        print(f"✂️ Sampled {num_samples_per_class:,} fake videos")

    all_videos = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)

    print(f"\n✅ Final dataset size: {len(all_videos):,} videos")
    print(f"   Real: {len(real_videos):,} | Fake: {len(fake_videos):,}")

    return all_videos, labels

# --- 8. EVALUATION FUNCTION ---
def evaluate_model(model, dataloader, config):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch[0] is None:
                continue

            (visual_data, audio_data), labels = batch
            visual_data = visual_data.to(config.device)
            audio_data = audio_data.to(config.device)

            with autocast():
                outputs = model(visual_data, audio_data)
                probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    return all_labels, all_preds, all_probs

# --- 9. METRICS CALCULATION ---
def calculate_metrics(labels, preds, probs):
    """Calculate all evaluation metrics."""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

# --- 10. VISUALIZATION FUNCTIONS ---
def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix: V1d (DeepSpeak) → KoDF')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {output_path}")

# --- 11. MAIN EXECUTION ---
def main():
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION: V1d (DeepSpeak) → KoDF")
    print("="*80)

    # Step 1: Discover videos
    video_paths, labels = discover_kodf_videos(
        config.kodf_base_dir, 
        config.num_samples_per_class
    )

    # Step 2: Create dataset and dataloader
    print("\n" + "="*80)
    print("CREATING DATALOADER")
    print("="*80)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

    dataset = KoDFDataset(video_paths, labels, config, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"✅ DataLoader created with {len(dataset)} samples")

    # Step 3: Load model
    print("\n" + "="*80)
    print("LOADING V1d MODEL (TRAINED ON DEEPSPEAK)")
    print("="*80)

    model = FusionModel_V1(config).to(config.device)
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    print(f"✅ Model loaded from: {config.model_path}")

    # Step 4: Evaluate
    labels_true, preds, probs = evaluate_model(model, dataloader, config)

    # Step 5: Calculate metrics
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)

    metrics = calculate_metrics(labels_true, preds, probs)

    # Step 6: Print results
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("="*80)
    print(f"\n🔥 Model: V1d trained on DeepSpeak")
    print(f"🎯 Evaluated on: KoDF ({config.num_samples_per_class*2:,} videos)")
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   🎯 Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   📈 AUC Score:  {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    print(f"   🎯 Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   🔍 Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   ⚖️  F1-Score:   {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

    print("\n📋 Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\n   [[TN  FP]")
    print("    [FN  TP]]")

    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    print(f"\n   True Negatives (Real as Real):  {tn:,}")
    print(f"   False Positives (Real as Fake): {fp:,}")
    print(f"   False Negatives (Fake as Real): {fn:,}")
    print(f"   True Positives (Fake as Fake):  {tp:,}")

    print("\n📊 Detailed Classification Report:")
    print(classification_report(labels_true, preds, target_names=['Real (0)', 'Fake (1)']))

    # Step 7: Save confusion matrix
    cm_path = os.path.join(config.output_dir, 'confusion_matrix_v1d_ds_kodf.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

    # Step 8: Save metrics to file
    results_path = os.path.join(config.output_dir, 'v1d_ds_kodf_evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-DATASET EVALUATION: V1d (DeepSpeak) → KoDF\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: V1d (MobileNetV3-Small + Mamba)\n")
        f.write(f"Trained on: DeepSpeak\n")
        f.write(f"Evaluated on: KoDF ({config.num_samples_per_class*2:,} videos)\n")
        f.write(f"Sampling: {config.num_samples_per_class:,} videos per class\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"AUC Score: {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n")
        f.write(f"   TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(labels_true, preds, target_names=['Real (0)', 'Fake (1)']))

    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Confusion matrix saved to: {cm_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n📝 Summary:")
    print(f"   Training Dataset: DeepSpeak")
    print(f"   Test Dataset: KoDF")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   AUC: {metrics['auc']*100:.2f}%")
    print(f"   Results: {config.output_dir}")

if __name__ == '__main__':
    main()
