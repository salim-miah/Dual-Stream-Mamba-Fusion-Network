# Dual-Stream Mamba Fusion Network

This repository implements experiments and training pipelines for a dual-stream system (visual + audio) for lip-sync / deepfake detection. It contains several variants: LSTM-based prototypes and Tiny CNN / Mamba-style variants (MobileNet + Mamba blocks).

## Repository layout (high level)
- LSTM-based/
  - v1.py, v4.py — LSTM temporal modeling experiments and entry scripts
- scripts for V1d and Transformer variant/
  - V1d.py — MobileNet/Mamba tiny variant and entry script
- results of ablation study/ — logs and CSV results
- Python notebooks used for statistical analysis - 5 random seeds/ — analysis notebooks

## Short overview of the main pipeline
The pipeline follows these high-level stages:

1. Prepare dataset
   - Organize data into class folders under a data directory, typically:
     - <data_dir>/0_real/...
     - <data_dir>/1_fake/...
   - Each video should have associated audio (sidecar .wav or extractable via ffmpeg).

2. Preprocess & cache
   - Visual: sample frames, detect face / mouth region and crop frames into fixed-size tensors.
   - Audio: load audio, create mel-spectrogram chunks aligned to video frames.
   - Caching: expensive preprocessing should be cached (RAMCachedDataset or disk cache) before training.

3. DataLoader & transforms
   - Build datasets that return (visual_tensor, audio_tensor, label).
   - Use collate functions that skip failed items (scripts already include collate_fn_skip_errors).
   - Apply train / val transforms and normalization.

4. Model construction
   - LSTM variants: CNN backbones (MobileNet/EfficientNet) -> projection -> LSTM temporal model -> fusion head.
   - Tiny Mamba variants: MobileNetV3 small + Mamba blocks -> per-stream encoders -> fusion head that combines audio + visual features.

5. Training
   - Training loop supports gradient accumulation, mixed precision, gradient clipping.
   - Use label-smoothed BCE loss and appropriate scheduler (ReduceLROnPlateau used in examples).
   - Save best model by validation metric (loss / AUC).

6. Final evaluation
   - Load best checkpoint, run test set, compute sigmoid outputs and metrics (accuracy, AUC).
   - Save predictions and logs.

## Steps to create & run the main pipeline (example commands for Windows)
1. Create environment and install basics:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install torch torchvision librosa opencv-python tqdm scikit-learn ffmpeg-python
   ```
   (Add mamba-ssm or other project-specific packages if required by V1d scripts.)

2. Edit config paths
   - Open the entry script you will use and set paths in its Config:
     - LSTM-based/v4.py -> Config().data_dir, Config().model_save_dir, other hyperparams
     - LSTM-based/v1.py or scripts for V1d and Transformer variant/V1d.py similarly

3. Preprocess & cache (recommended)
   - Run caching flow or a small preprocessing run to generate cached tensors. Example: run entry with a flag or a short dataset split configured in Config. If no dedicated cache script exists, run the dataset creation once with num_workers=0 to populate cache.

4. Train
   - From repo root run the chosen entry script:
     ```powershell
     python "LSTM-based\v4.py"
     ```
     or
     ```powershell
     python "scripts for V1d and Transformer variant\V1d.py"
     ```
   - Monitor training logs and validation metrics. Adjust num_workers (Windows: start with 0 or small values).

5. Evaluate / test
   - After training finishes and the best checkpoint is saved, run the evaluation block or the same script with a --evaluate flag (if implemented) pointing to the saved model.

## Tips & notes
- Caching preprocessing is critical — face detection and mel extraction are slow. Use RAMCachedDataset or persistent cache.
- On Windows, set DataLoader num_workers=0 if you encounter spawn issues.
- Use collate_fn_skip_errors provided in scripts to avoid crashes on corrupt files.
- Experiment scripts include label smoothing + ReduceLROnPlateau — these help stabilize training.

## Useful entry points & files
- LSTM-based/v1.py — prototype pipeline and config
- LSTM-based/v4.py — improved LSTM pipeline (recommended for LSTM experiments)
- scripts for V1d and Transformer variant/V1d.py — MobileNet + Mamba tiny variant
- results of ablation study/ — example logs and CSVs for reference
- analysis notebooks in the notebooks folder for results inspection

## Reproducibility suggestions
- Fix random seeds in config and disable worker randomness for exact reproducibility.
- Save full checkpoint (model + optimizer + scheduler + rng states).
- Cache preprocessed datasets to avoid nondeterministic preprocessing differences.