# Détection des Mauvaises Herbes par Deep Learning

Multi-class plant seedling classifier that distinguishes **9 weed species** from **3 crop species** using transfer learning with MobileNetV2. Achieves **88.86% test accuracy** on the V2 Plant Seedlings Dataset.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 88.86% |
| Macro F1-Score | 0.81 |
| Test Loss | 2.03 |

## Model Architecture

```
MobileNetV2 (ImageNet, frozen)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dense(256, ReLU, L2)  →  Dropout(0.5)
    → Dense(128, ReLU, L2)  →  Dropout(0.3)
    → Dense(12, Softmax)
```

Training uses a **two-phase strategy**:
- **Phase 1** — Feature extraction: base model frozen, Adam lr=1e-4, up to 30 epochs
- **Phase 2** — Fine-tuning: last 30 layers unfrozen, Adam lr=1e-5, up to 20 epochs

## Classes

| Weeds (9) | Crops (3) |
|---|---|
| Black-grass | Common wheat |
| Charlock | Maize |
| Cleavers | Sugar beet |
| Common Chickweed | |
| Fat Hen | |
| Loose Silky-bent | |
| Scentless Mayweed | |
| Shepherds Purse | |
| Small-flowered Cranesbill | |

## Dataset

**V2 Plant Seedlings Dataset** — ~5,300 images across 12 classes.

- Add it to your Kaggle notebook via the **Data** tab → search `v2-plant-seedlings-dataset`
- Or download from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset)

## How to Run

This notebook is designed to run on **Kaggle with GPU**. Training was performed on dual Tesla T4 GPUs provided by Kaggle.

1. Create a new Kaggle notebook
2. **Data** tab → **Add Data** → search and add `v2-plant-seedlings-dataset`
3. **Settings** → **Accelerator** → select **GPU T4 x2**
4. Upload `weed-detection.ipynb` and run all cells in order

## Tech Stack

| | Library | Version |
|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.10+ |
| Backbone | MobileNetV2 | ImageNet weights |
| ML utilities | scikit-learn | 1.8 |
| Image processing | Pillow, NumPy | — |
| Visualization | Matplotlib, Seaborn | — |

## Output Artifacts

After training, the following files are saved to `/kaggle/working/`:

| File | Description |
|---|---|
| `weed_detection_model.h5` | Final trained model |
| `best_model.h5` | Best checkpoint from Phase 1 |
| `best_model_finetuned.h5` | Best checkpoint from Phase 2 |
| `01_distribution_classes.png` | Class distribution bar chart + pie chart |
| `02_training_curves.png` | Accuracy & loss curves across both phases |
| `03_confusion_matrix.png` | 12×12 confusion matrix |
| `04_per_class_metrics.png` | Precision / Recall / F1 per class |
| `05_prediction_examples.png` | 12 sample predictions (correct in green, errors in red) |

## Project Structure

```
.
├── weed-detection.ipynb    # Full pipeline — data loading to evaluation
├── requirements.txt        # Pinned dependencies
└── data/
    └── v2-plant-seedlings-dataset/
        ├── Black-grass/
        ├── Charlock/
        └── ...             # one folder per class
```
