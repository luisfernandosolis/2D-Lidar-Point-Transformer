# 2D LiDAR Point Transformer

Overview
--------
- `dataset/` — dataset download and helper scripts.
- `utils/` — utility functions for LiDAR projection, KMeans clustering, and operations that match LiDAR points to image masks.
- `training/` — `prepare_data.py` and `train.py` to create a JSON dataset mapping and a simple training loop.
- `model/` — minimal PyTorch implementation for `LidarPointTransformer`.
- `assets/` — example images/visualizations (the repo includes a placeholder `nuscenes_results.svg`).

Requirements
------------
Create a Python virtual environment and install the basic dependencies:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```