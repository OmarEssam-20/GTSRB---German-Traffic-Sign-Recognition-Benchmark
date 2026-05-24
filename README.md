# 🚦 German Traffic Sign Recognition Benchmark (GTSRB)

A deep learning project that classifies **43 categories of German traffic signs** using three progressively more powerful approaches: a CNN trained from scratch, transfer learning with EfficientNetB0, and fine-tuning. The project culminates in a deployed **Streamlit web app** where users can upload a traffic sign image and receive an instant classification.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach & Methodology](#approach--methodology)
  - [Phase 1 — CNN from Scratch](#phase-1--cnn-from-scratch)
  - [Phase 2 — Transfer Learning (EfficientNetB0)](#phase-2--transfer-learning-efficientnetb0)
  - [Phase 3 — Fine-Tuning](#phase-3--fine-tuning)
- [Problems & Solutions](#problems--solutions)
- [Results](#results)
- [Streamlit Web App](#streamlit-web-app)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)

---

## Project Overview

Traffic sign recognition is a critical component of autonomous driving and advanced driver-assistance systems (ADAS). This project benchmarks three different training strategies on the GTSRB dataset to answer a practical question:

> **How much does transfer learning and fine-tuning improve accuracy over a baseline CNN?**

The pipeline covers the full machine learning workflow: data loading and augmentation, model architecture design, training with callbacks, evaluation on a held-out test set, and deployment as an interactive web application.

---

## Dataset

**Source:** [GTSRB — German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) on Kaggle

| Property | Value |
|---|---|
| Total classes | 43 |
| Training images | ~39,000 |
| Test images | ~12,630 |
| Image format | PNG (variable size) |
| Labels | Integer class IDs (0–42) |

Classes include speed limit signs (20–120 km/h), prohibitory signs (no passing, no entry), mandatory signs (turn ahead, roundabout), and warning signs (road work, slippery road, wild animals, etc.).

---

## Project Structure

```
gtsrb-german-traffic-sign/
├── Train/                      # 43 subdirectories, one per class
│   ├── 0/
│   ├── 1/
│   └── ...
├── Test/                       # Test images
├── Test.csv                    # Ground-truth labels for test set
scratch_cnn_gtsrb.keras         # Saved CNN-from-scratch model
tl_model_best.keras             # Best transfer learning checkpoint
finetuned_model_best.keras      # Best fine-tuned model checkpoint
class_names.json                # Class ID → name mapping
app.py                          # Streamlit web application
requirements.txt
```

---

## Approach & Methodology

### Phase 1 — CNN from Scratch

**Goal:** Establish a baseline using a small convolutional network trained entirely on the GTSRB data.

**Architecture:**

```
Input (32×32×3)
  → Rescaling (pixel values to [0, 1])
  → Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool
  → Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool
  → Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool
  → Flatten
  → Dense(256, ReLU) → Dropout(0.5)
  → Dense(43, Softmax)
```

**Why this design:**
- Images were resized to **32×32** — small enough for fast training, and traffic signs carry enough information at that resolution.
- Three progressively deeper conv blocks capture low-level edges → mid-level shapes → high-level symbol patterns.
- `BatchNormalization` after each conv block stabilises training and speeds up convergence.
- `Dropout(0.5)` in the dense layers combats overfitting on a relatively small dataset.
- `sparse_categorical_crossentropy` is used because labels are raw integers (not one-hot encoded).

**Training:** Adam optimizer, 10 epochs, 90/10 train-validation split, batch size 32.

---

### Phase 2 — Transfer Learning (EfficientNetB0)

**Goal:** Leverage ImageNet-pretrained weights to dramatically boost accuracy without training from scratch.

**Why EfficientNetB0:**
EfficientNet models are designed with a compound scaling strategy that jointly scales network depth, width, and resolution. Even the smallest variant (B0) contains rich feature representations that generalise well to new domains. Crucially, EfficientNetB0 expects **224×224** inputs, so the dataset was reloaded at that size.

**Setup:**
- The base model was loaded with `include_top=False` and `weights="imagenet"`.
- `base_model.trainable = False` — all pretrained weights were frozen; only the new classification head was trained.
- The new head: `GlobalAveragePooling2D → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Dense(43, Softmax)`.
- Data augmentation (`RandomRotation`, `RandomZoom`, `RandomContrast`) was applied inline to improve generalisation.

**Callbacks used:**
| Callback | Purpose |
|---|---|
| `EarlyStopping(patience=5)` | Stop training when validation accuracy plateaus |
| `ModelCheckpoint` | Save the best weights automatically |
| `ReduceLROnPlateau(factor=0.5, patience=3)` | Halve the learning rate when validation loss stalls |

**Training:** Adam (lr=1e-3), up to 20 epochs, batch size 16.

---

### Phase 3 — Fine-Tuning

**Goal:** Squeeze out additional accuracy by allowing the top layers of EfficientNetB0 to adapt to traffic sign features.

**Why fine-tune only the last 30 layers:**
The early layers of a CNN learn universal features (edges, textures, gradients) that transfer well to any vision task. The deeper layers learn more task-specific patterns. Unfreezing only the last 30 layers allows the model to specialise those higher-level representations to traffic signs, while preserving the stable low-level knowledge.

**Critical detail — learning rate:**
Fine-tuning must use a **much smaller learning rate** (1e-5 vs. 1e-3) than the initial transfer learning phase. A large learning rate at this stage would overwrite the carefully tuned pretrained weights with random noise, destroying the learned representations and causing performance collapse.

**Training:** Adam (lr=1e-5), up to 20 epochs, same callbacks with `patience=7`.

---

## Problems & Solutions

### 1. Memory pressure with 224×224 images
**Problem:** Reloading the dataset at 224×224 with a batch size of 32 caused out-of-memory errors on Google Colab's GPU.  
**Solution:** Reduced batch size to **16** for the transfer learning and fine-tuning phases. Also used `prefetch(AUTOTUNE)` and `shuffle(500)` (smaller buffer than the CNN phase) to reduce peak memory usage during loading.

---

### 2. Finding the EfficientNetB0 layer inside the compiled model
**Problem:** After reloading the best transfer learning checkpoint from disk (`tl_model_best.keras`), the model is a functional `tf.keras.Model` wrapper. Trying to access `base_model` directly no longer works — it's embedded as a named layer.  
**Solution:** Iterated over `tl_model.layers` and matched by name:
```python
for layer in tl_model.layers:
    if "efficientnetb0" in layer.name:
        efficientnet_layer = layer
        break
```
Then unfroze selectively:
```python
efficientnet_layer.trainable = True
for layer in efficientnet_layer.layers[:-30]:
    layer.trainable = False
```

---

### 3. Test set evaluation required a custom data pipeline
**Problem:** The test images are not organised into class subdirectories like the training data — they are listed in a `Test.csv` file with a `Path` and `ClassId` column. `image_dataset_from_directory` cannot be used here.  
**Solution:** Built a manual `tf.data` pipeline:
```python
path_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
ds = path_ds.map(load_image).batch(batch_size).prefetch(AUTOTUNE)
```
Two separate test datasets were created: one at 32×32 for the scratch CNN and one at 224×224 for the transfer learning / fine-tuned models.

---

### 4. Preprocessing mismatch between training and inference
**Problem:** During transfer learning, images were passed through `efficientnet.preprocess_input()` before entering the base model. Forgetting this step during inference causes a distribution shift — the model sees pixel values in a range it was never trained on.  
**Solution:** In the Streamlit app, `preprocess_input` is explicitly applied to every image before prediction:
```python
from tensorflow.keras.applications.efficientnet import preprocess_input
img_array = preprocess_input(img_array)
```

---

### 5. Slow dataset loading
**Problem:** Without any pipeline optimisation, each training epoch spent significant time reading and decoding images from disk.  
**Solution:** Applied `.cache().shuffle(1000).prefetch(AUTOTUNE)` on the CNN-from-scratch pipeline, and `.shuffle(500).prefetch(AUTOTUNE)` on the transfer learning pipeline. `cache()` stores decoded batches in memory after the first epoch, eliminating redundant disk I/O.

---

## Results

| Model | Test Accuracy |
|---|---|
| CNN from Scratch | ~92–94% |
| EfficientNetB0 Transfer Learning | ~97–98% |
| EfficientNetB0 Fine-Tuned | ~98–99% |

> *Exact values depend on the random seed and early stopping epoch. The table reflects typical results from this pipeline.*

Fine-tuning consistently outperforms frozen transfer learning, which in turn far surpasses the from-scratch baseline. The gap is largest on rare classes where the scratch model had limited training examples.

---

## Streamlit Web App

The deployed app (`app.py`) allows users to upload any traffic sign image (JPG, PNG, or PPM) and receive:

- The **predicted sign name** with confidence score.
- A **top-3 ranked list** of predictions with progress bars.

The fine-tuned EfficientNetB0 model (`finetuned_model_best.keras`) is loaded with `@st.cache_resource` so it is only read from disk once per session.

**Screenshot preview:**
```
🚦 German Traffic Sign Recognition
Upload a traffic sign image — the model will identify it from 43 possible classes.

[Image preview]    Prediction Result
                   ✅ Speed limit (50km/h)
                   Confidence: 98.71%

                   Top 3 Predictions
                   1. Speed limit (50km/h) — 98.71%
                   2. Speed limit (80km/h) — 0.89%
                   3. Speed limit (30km/h) — 0.31%
```

---

## How to Run

### 1. Train the models (Google Colab)

1. Upload the notebook to Google Colab.
2. Add your Kaggle API credentials.
3. Run all cells in order. The notebook will:
   - Download and unzip the GTSRB dataset.
   - Train all three models.
   - Save `.keras` model files and `class_names.json`.
   - Generate and download `app.py` and `requirements.txt`.

### 2. Run the web app locally

```bash
# Install dependencies
pip install -r requirements.txt

# Place these files in the same folder as app.py:
#   finetuned_model_best.keras
#   class_names.json

# Launch the app
streamlit run app.py
```

### requirements.txt

```
streamlit
tensorflow-cpu
numpy
pillow
```

---

## Technologies Used

| Technology | Role |
|---|---|
| TensorFlow / Keras | Model building and training |
| EfficientNetB0 | Pretrained backbone for transfer learning |
| Streamlit | Interactive web application |
| Kaggle API | Dataset download |
| Google Colab | GPU training environment |
| Matplotlib | Training curve visualisation |
| Pandas | Test CSV parsing |
| PIL (Pillow) | Image preprocessing in the app |

---

## Author

Built as a computer vision benchmark project exploring the progression from custom CNNs to state-of-the-art transfer learning on a real-world multi-class classification task.
