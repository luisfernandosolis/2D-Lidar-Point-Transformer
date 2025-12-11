# 2D-Lidar-Point-Transformer

Tiny project scaffold with scripts for datasets, utilities, training and a minimal model for LiDAR-to-image point projection and region-aware training.

Usage summary
- Create a virtual environment and install requirements
  ```powershell
  python -m venv venv; .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- Download dataset
  ```powershell
  python dataset/download.py --output-dir dataset --unzip
  ```
- Prepare dataset
  ```powershell
  python training/prepare_data.py --lidar-csv dataset/lidar_data.csv --masks-dir masked_regions --out-json final_processed_dataset.json
  ```
- Train (very minimal)
  ```powershell
  python training/train.py --dataset-json final_processed_dataset.json --out-dir models --epochs 2
  ```

Notes
- This repo contains minimal placeholders and a basic pipeline. You may want to integrate SAM, BLIP-2, or CLIP for better annotation and semantic labeling.
