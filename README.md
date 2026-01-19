# ECE443 Audio Genre Classification (GMM)

Music genre classification using **MFCC** features and **Gaussian Mixture Models (GMMs)**.

This project was developed for the **ECE443 (Voice and Sound Recognition)** course and implements a classic statistical pattern-recognition pipeline for supervised classification of music genres.

---

## 📌 Project Overview

### Pipeline
1. Load audio signals (`.wav`)
2. Extract frame-level features (MFCC)
3. Standardize features with `StandardScaler`
4. Train one **GMM per genre**
5. Classify tracks by **maximum log-likelihood** across genre models
6. Evaluate with accuracy + confusion matrix

---

## 📂 Datasets

This notebook supports **two datasets** and can run **one or both** in a single execution.

### 1) Default Dataset (course/instructor-provided)
- Downloaded automatically from Kaggle using `kagglehub`
- Stored locally under: `data-default/`
- Expected structure:

```
data-default/
  train/
    blues/  *.wav
    ...
  test/
    blues/  *.wav
    ...
```

### 2) GTZAN Dataset
- Downloaded automatically from Kaggle using `kagglehub`
- The original Kaggle dataset is split into a local train/test structure under: `data-gtzan/`
- Train/test split is controlled by `GTZAN_TRAIN_RATIO` (seed fixed to 42 for reproducibility)

```
data-gtzan/
  train/
    blues/  *.wav
    ...
  test/
    blues/  *.wav
    ...
```

> Note: the split code **copies** files into `data-gtzan/train` and `data-gtzan/test` (it does not modify the downloaded source). If you re-run the split with different settings, delete `data-gtzan/` first to avoid leftover files from older runs.

---

## 🔧 Notebook Parameters

In `project.ipynb`, you control the experiments via these parameters:

```python
DATASETS_FETCH = True          # download/copy datasets (run once, then set False)
RUN_DATASETS = ["default", "gtzan"]  # run one or both

GTZAN_TRAIN_RATIO = 0.8        # only affects GTZAN split

BUILD_FEATURES = True          # extract MFCC features into .npz
TRAIN = True                   # train GMM models
Ks = [2, 4, 8, 16, 32]         # number of mixture components
```

Typical usage:
- **First run**: `DATASETS_FETCH=True`, `BUILD_FEATURES=True`, `TRAIN=True`
- **Later runs** (same data/features): `DATASETS_FETCH=False`, `BUILD_FEATURES=False`, `TRAIN=True` (or `TRAIN=False` if you only want to re-plot / re-evaluate)

---

## 📁 Features, Models, and Results (where files go)

For each dataset, the notebook creates **dataset-specific** outputs.

### Features
Saved under each dataset directory:

```
<DATA_DIR>/features/
  train/<genre>/*.npz
  test/<genre>/*.npz
```

### Models
Saved under each dataset directory:

```
<DATA_DIR>/models/
  scaler.joblib
  gmm_<genre>_K<K>.joblib
```

### Results (report-ready)
Saved under:

```
results/<dataset_name>/
  accuracy_overall.csv
  accuracy_per_genre.csv
  accuracy_vs_K.png
  accuracy_per_K.png
  per_genre_grouped_by_K.png
  cm_K2.png
  cm_K4.png
  ...
```

Confusion matrices are saved **with counts printed in each cell**.

---

## 🛠️ Installation

### Option A: Using Conda
```bash
conda create -n ece443 python=3.10
conda activate ece443
pip install -r requirements.txt
```

### Option B: Plain pip
```bash
pip install numpy pandas matplotlib librosa soundfile scipy scikit-learn tqdm joblib kagglehub
```

> Kaggle downloads via `kagglehub` require Kaggle authentication configured on your machine.

---

## ▶️ Running the project

1. Open `project.ipynb`
2. Set parameters (datasets, `Ks`, training flags)
3. Run all cells

---

## 📊 Evaluation

The notebook reports and saves:
- Overall accuracy per `K`
- Confusion matrix per `K` (with counts)
- Per-genre accuracy tables
- Grouped bar plots: **per-genre accuracy for each K**

---

## 👥 Authors

- [Dimitris Kleidonaris](https://github.com/dkleidonaris)
- [Nikolaos-Panagiotis Rinos](https://github.com/NRinos)

---

## 📜 License

This project is intended for **educational and academic use**.

Dataset licensing follows the terms specified on Kaggle.
