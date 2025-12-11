# 2D LiDAR Point Transformer

This repository contains a minimal working pipeline to project 2D LiDAR points into image coordinates, associate points with image regions (segmentation masks), and train a small Transformer-based model to classify or otherwise learn from region-level LiDAR aggregation.

Overview
--------
- `dataset/` — dataset download and helper scripts.
- `utils/` — utility functions for LiDAR projection, KMeans clustering, and operations that match LiDAR points to image masks.
- `training/` — `prepare_data.py` and `train.py` to create a JSON dataset mapping and a simple training loop.
- `model/` — minimal PyTorch implementation for `LidarPointTransformer`.
- `assets/` — example images/visualizations (the repo includes a placeholder `nuscenes_results.svg`).

This is intentionally an example implementation that you can extend for production-level training pipelines. The scripts are written such that you can enable optional components (SAM/BLIP/CLIP) if needed.

Requirements
------------
Create a Python virtual environment and install the basic dependencies:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (for mask generation and semantic labeling):

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers blip2
pip install gdown
```

If you will use the GPU, ensure you install the correct PyTorch version for your CUDA setup (see https://pytorch.org/).

Dataset: nuScenes example
-------------------------
The pipeline is intended to work with LiDAR-to-image paired data. As an example, nuScenes is a popular dataset: you will usually do the following steps for nuScenes or a similar dataset:

1. Download the dataset according to the nuScenes instructions and export or convert the relevant LiDAR frames to a CSV containing columns like `id`, `angle`, `distance`, and `img_front`.
2. Collect or generate segmentation masks for the frontal images and store them in `masked_regions/` or a structure where `training/prepare_data.py` can find them.
3. Run `training/prepare_data.py` to create a JSON mapping of scenes, masks, and LiDAR indices used as the dataset for training.

Note: this repo includes a placeholder example image (`assets/nuscenes_results.svg`) that demonstrates the kind of visualization you might generate for nuScenes results.

How to use
----------
1) Download datasets (example using gdown or a URL):

```powershell
python dataset/download.py --output-dir dataset --unzip --gdrive-ids "[{'id':'FILE_ID','name':'images.zip'}]"
```

2) Prepare the dataset (project LiDAR into images and match masks):

```powershell
python training/prepare_data.py --lidar-csv dataset/lidar_data.csv --masks-dir masked_regions --out-json final_processed_dataset.json
```

3) Train the model (very minimal example):

```powershell
python training/train.py --dataset-json final_processed_dataset.json --out-dir models --epochs 10 --batch-size 8
```

Examples and visualization
--------------------------
The repository contains `assets/nuscenes_results.svg` as a sample visualization for nuScenes as a placeholder — replace this image with your actual output visualization (PNG/JPG/SVG). The pipeline in `utils/` includes functions you can use to overlay LiDAR points on images and display associated regions.

Next steps & recommendations
----------------------------
- Integrate Segment Anything (SAM) as an optional stage to generate masks automatically from images, and store masks in `masked_regions/`.
- Use CLIP/BLIP to annotate masks with semantic class labels automatically if ground-truth labels are unavailable.
- Implement per-point labels (if available) and per-point loss functions instead of sample-level mean pooling.
- Add evaluation metrics (mAP, IoU per class) and a validation loop for better model supervision.
- Add unit tests and a demo notebook that runs the pipeline end-to-end.


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
- Train
  ```powershell
  python training/train.py --dataset-json final_processed_dataset.json --out-dir models --epochs 2
  ```