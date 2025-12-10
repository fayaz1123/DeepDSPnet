# DeepDSPNet: Monocular Satellite Terrain Estimation

DeepDSPNet is a deep learning framework for estimating 3D Digital
Elevation Models (DEMs) directly from **single-view satellite RGB
images**. It uses a **multi-stream architecture** that merges
photometric texture (RGB) with geometric priors (Edges + MiDaS Monocular
Depth) to produce accurate terrain height maps.

## 📂 Project Structure

``` text
project-root/
│
├── data/                   # Raw Input GeoTIFFs (RGB + DEM)
├── dataset/                # Auto-generated training patches
│   ├── train/
│   │   ├── A/              # RGB patches
│   │   ├── B/              # Elevation (Float32 TIF)
│   │   └── C/              # MiDaS relative depth maps
│
├── checkpoints/            # Saved model weights (.pth)
├── results/                # Inference visualizations
│
├── src/                    # Core source code
│   ├── dataset.py          # Dataset & dataloader logic
│   ├── model.py            # DeepDSPNet architecture (Gated Fusion)
│   ├── losses.py           # Custom Losses (TV, Wavelet, L1)
│   └── __init__.py
│
├── preprocess_data.py      # Step 1: Slice GeoTIFFs into patches
├── generate_depth.py       # Step 2: Create MiDaS depth priors
├── train.py                # Step 3: Training pipeline
├── inference.py            # Step 4: Model inference & visualization
└── requirements.txt        # Dependencies
```

## 🛠️ Installation

Clone the repository:

``` bash
git clone https://github.com/yourusername/DeepDSPNet.git
cd DeepDSPNet
```

Install dependencies (recommended: use a virtual environment):

``` bash
pip install -r requirements.txt
```

> **Note:** A CUDA-enabled GPU is strongly recommended. CPU-only
> training will be extremely slow.

## 🚀 Usage Pipeline

Follow the steps **in order** to prepare data, train the network, and
run inference.

### 1. Data Preparation

1.  Place your RGB & DEM GeoTIFFs into the `data/` folder.\
2.  Edit `preprocess_data.py` and update the `DATA_SOURCES` list with
    your filenames.
3.  Run the patch generator:

Or you can just run the following: 
``` bash
python preprocess_data.py
```

``` bash
python preprocess_data.py
```

**Output:**\
`dataset/train/A` (RGB) and `dataset/train/B` (Elevation).

### 2. Generate Geometric Priors (MiDaS Depth)

Generate relative depth maps using the MiDaS model:

``` bash
python generate_depth.py
```

**Output:**\
`dataset/train/C` (Depth maps)

### 3. Training

``` bash
python train.py
```

-   Best model saved at:

```{=html}
<!-- -->
```
    checkpoints/best_deepdsp_model.pth

-   Modify learning rate, batch size, and epochs inside `train.py`.

### 4. Inference & Evaluation

``` bash
python inference.py
```

**Output:** Visualizations saved in `results/`, including:

-   Input RGB\
-   Ground Truth DEM\
-   Predicted DEM\
-   Absolute Error Heatmap

## 🧠 Model Architecture

### Texture Stream (RGB)

-   ResNet34 backbone (ImageNet pretrained)
-   Captures semantic terrain cues

### Geometry Stream (Edges + Depth)

-   Canny edges + MiDaS depth\
-   Provides structural shape priors

### Gated Fusion

-   Multi-scale gating at 1/4, 1/8, 1/16 resolutions\
-   Dynamically balances texture vs. geometry

### Loss Functions

-   L1 Loss\
-   Total Variation (TV) Loss\
-   Wavelet Loss

## 📝 License

Licensed under the **MIT License**. See the `LICENSE` file.
