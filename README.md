# ECE443 Audio Genre Classification

Music genre classification using statistical audio features and **Gaussian Mixture Models (GMMs)**.

This project was developed as part of the **ECE443** (Voice and Sound Recognition) course and focuses on supervised classification of music genres from audio signals.

---

## 📌 Project Overview

The goal of this project is to automatically classify music tracks into genres based on extracted audio features. The approach follows a **statistical pattern recognition pipeline**:

1. Audio loading and preprocessing
2. Feature extraction (e.g. MFCCs)
3. Feature normalization
4. Genre modeling using **Gaussian Mixture Models (GMMs)**
5. Evaluation on a held-out test set

---

## 📂 Datasets

This project supports **two alternative datasets**. The user can select which dataset to use by setting a parameter in the notebook.

### 1️⃣ Default Dataset (Instructor-provided)

This is the **default dataset provided by the course instructor**, intended mainly for testing and benchmarking.

* Hosted on **Kaggle**
* Downloaded automatically using `kagglehub`
* Fixed train / test split

**Structure:**

```
train/
  ├── blues/
  ├── rock/
  └── ...

test/
  ├── blues/
  ├── rock/
  └── ...
```

Each genre folder contains `.wav` audio files.

### 2️⃣ GTZAN Dataset

The well-known **GTZAN music genre dataset** can also be used.

* Downloaded automatically
* Split into training and test sets
* Train/test split ratio is configurable

---

### 🔧 Dataset Selection

Dataset selection is controlled by the following parameters in the notebook:

```python
DATASET = "default"  # "default" or "gtzan"

DEFAULT_FETCH = True
GTZAN_FETCH = False
GTZAN_TRAIN_RATIO = 0.8
```

Only one dataset is active at a time.

---

## 🧠 Methodology

* Audio loading with `librosa`
* Feature extraction (MFCCs, statistics over frames)
* Feature standardization using `StandardScaler`
* Genre-wise GMM training
* Classification via maximum likelihood

---

## 🛠️ Installation

### 1️⃣ Create Conda environment

```bash
conda create -n ece443 python=3.10
conda activate ece443
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the project

Typical workflow:

1. Download dataset from Kaggle
2. Copy dataset into `data/`
3. Train GMM models
4. Evaluate on test set
5. Visualize results (confusion matrices, accuracy per genre)

---

## 📊 Evaluation

* Overall classification accuracy
* Confusion matrix per experiment
* Per-genre performance analysis

---

## 👥 Authors

* [Dimitris Kleidonaris](https://github.com/dkleidonaris)
* [Nikolaos-Panagiotis Rinos](https://github.com/NRinos)

---

## 📜 License

This project is intended for **educational and academic use**.

Dataset licensing follows the terms specified on Kaggle.
