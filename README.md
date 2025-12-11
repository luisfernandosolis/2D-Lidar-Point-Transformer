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

Contact
-------
If you want, I can adapt this pipeline to a specific dataset layout (for example, the exact nuScenes folder structure), add SAM/CLIP/BLIP integration, or add tests & metrics. Let me know what you prefer, and I’ll continue iterating.
# 2D-Lidar-Point-Transformer

Repositorio con una implementación y scripts mínimos para proyectar puntos LiDAR 2D sobre imágenes (frontal) y entrenar un modelo estilo Transformer que aprenda a clasificar/regresar información por regiones segmentadas en la imagen.

## Resumen

Este proyecto incluye:

- Preprocesado de datos de LiDAR (CSV) y máscaras segmentadas (SAM/otros) para asociar puntos de LiDAR con regiones de imágenes.
- Un conjunto de utilidades en `utils/` para proyección, clustering (KMeans) y vinculaciones entre máscaras y puntos LiDAR.
- Un modelo básico `LidarPointTransformer` implementado en PyTorch para clasificación por región/punto.
- Script de entrenamiento `training/train.py` y script de preparación `training/prepare_data.py`.
- Descarga básica de datasets con `dataset/download.py` (se usa gdown si está disponible).

---

## Estructura del repositorio

- `dataset/` - scripts para descargar y preparar el dataset.
- `model/` - contiene `LidarPointTransformer.py`.
- `training/` - scripts `prepare_data.py` y `train.py`.
- `utils/` - funciones utilitarias para proyecciones, KMeans, y asociación de máscaras.
- `requirements.txt` - dependencias mínimas.

---

## Requisitos previos

Instala un entorno virtual y las dependencias:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si desea generar máscaras con Segment-Anything (SAM) o descripciones con BLIP/CLIP, instale esas dependencias adicionalmente:

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers blip2
pip install gdown
```

---

## Dataset - nuScenes (ejemplo)

El pipeline está orientado a trabajar con una estructura donde existe:

- Un CSV con lecturas LiDAR por frame (`id, angle, distance, img_front` o similar).
- Imágenes frontales (ej. `images/front/image_000123.png`).
- Máscaras por imagen (salidas de SAM o máscaras manuales) en `masked_regions/`.

Para usar nuScenes:

1. Regístrate y descarga los archivos siguiendo las instrucciones oficiales: https://www.nuscenes.org/
2. Extrae las imágenes y los datos LiDAR (o transforma las anotaciones a un CSV que contenga columnas `id, angle, distance, img_front`).
3. Si no tienes máscaras segmentadas, crea un proceso con SAM para generar máscaras por imagen y guardarlas en `masked_regions/`.

---

## Flujo de trabajo (ejemplo)

1. Descargar dataset (ejemplo, usando `gdown` o URLs):

```powershell
python dataset/download.py --output-dir dataset --unzip --gdrive-ids "[{'id':'FILE_ID','name':'images.zip'}]"
```

2. Preparar los datos: proyectar LiDAR a coordenadas de imagen y asociar con máscaras.

```powershell
python training/prepare_data.py --lidar-csv dataset/lidar_data.csv --masks-dir masked_regions --out-json final_processed_dataset.json
```

3. Entrenar el modelo (ejemplo):

```powershell
python training/train.py --dataset-json final_processed_dataset.json --out-dir models --epochs 10 --batch-size 8
```

4. Visualizar resultados: el repo incluye un ejemplo de visualización en `assets/nuscenes_results.svg`. Reemplaza con tus resultados reales.

---

## Sobre los resultados - Ejemplo en nuScenes

El siguiente ejemplo muestra una visualización simulada de resultados en nuScenes (puntos LiDAR proyectados + regiones segmentadas):

![Ejemplo de resultados en nuScenes](assets/nuscenes_results.svg)

> Nota: Reemplaza `assets/nuscenes_results.svg` con una visualización real exportada por `utils` o con tus propias figuras (PNG, JPG o SVG). Para imágenes reales se recomienda exportar con alta resolución y añadir info: mAP, recall y precisión por clase.

---

## Recomendaciones y pasos futuros

- Integrar SAM correctamente como proceso opcional dentro de `training/prepare_data.py` para generar máscaras automáticamente.
- Añadir integración con CLIP/BLIP-2 para etiquetar regiones automáticamente con descripciones/texto semántico.
- Reforzar el entrenamiento: implementar pérdida por punto (si tienes anotaciones por punto), evaluación y métricas (mAP, IoU por región).
- Añadir tests unitarios y un notebook de demo que recorra el pipeline completo (download -> prepare -> train -> infer).

---

## Contacto

Si necesitas que adapte el pipeline a la estructura exacta de tus archivos de nuScenes o a la salida de tus generadores de máscaras, dímelo y lo adapto para automatizar todo el flujo.
# 2D-Lidar-Point-Transformer

Project scaffold with scripts for datasets, utilities, training and a minimal model for LiDAR-to-image point projection and region-aware training.

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