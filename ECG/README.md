# ECG Signal Processing Pipeline

This repository contains an ECG-focused signal processing pipeline adapted from an EMG/startle analysis script.

## Contents

- `ecg_config.py` — configurable parameters for ECG channel selection, filtering, event windows, and plotting
- `ecg_processor.py` — core ECG loading, filtering, event extraction, and CSV metadata handling
- `ecg_main.py` — main orchestration script that runs the pipeline and generates outputs
- `emg_raw_potentiation.py` — original EMG/startle analysis script preserved in the repo

## How to use

1. Adjust `ecg_config.py` for your data paths and ECG channel.
2. Run:
   ```bash
   ./.venv/bin/python ecg_main.py
   ```

## GitHub remote setup

This repo is initialized locally. To publish it to GitHub:

1. Create a new repository on GitHub (for example `yotameviatar/ecg`).
2. Add the remote:
   ```bash
   git remote add origin git@github.com:<your-username>/<repo-name>.git
   git push -u origin main
   ```

If you want, I can also help configure the GitHub remote automatically once you have the GitHub CLI installed or provide a repository URL.
