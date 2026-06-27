# Pothole Detection using RF-DETR Segmentation

This project implements an RF-DETR-based object detection and segmentation model specifically designed for pothole detection in road images. The model is trained on a custom dataset from Roboflow and achieves real-time detection capabilities with high accuracy.

## 🚀 Overview

RF-DETR (Roboflow Detection Transformer) is a state-of-the-art object detection model that combines the power of transformer architectures with efficient training. This implementation focuses on detecting and segmenting potholes and related road anomalies.

## 📋 Features

- **Object Detection & Segmentation**: Simultaneously detects and segments potholes in road images
- **Multiple Class Support**: Detects various road anomalies including:
  - Potholes
  - Manholes
  - Unmarked bumps
- **Real-time Inference**: Optimized for efficient inference
- **Visualization**: Automatic annotation and visualization of detections

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Supervision
- Roboflow API Key
- RF-DETR library (>=1.4.0)

### Installation

```bash
pip install -q rfdetr>=1.4.0 supervision roboflow
```

## 📊 Dataset

The model is trained on the "Pothole-Segmentation" dataset from Roboflow, which includes:
- **Training images**: 1,242
- **Training annotations**: 2,648
- **Classes**: 6 categories including different types of potholes and road anomalies

### Dataset Classes
- Manhole
- Potholes
- Unmarked Bump

## 🏗️ Model Architecture

The model uses **RFDETRSegNano**, a lightweight variant of RF-DETR optimized for segmentation tasks with:
- 33.3M trainable parameters
- Efficient transformer architecture
- Optimized for real-time inference

## 🔧 Usage

### Training

```python
from rfdetr import RFDETRSegNano

model = RFDETRSegNano()
model.train(
    dataset_dir="/content/Pothole-Segmentation-2",
    epochs=5,
    batch_size=4,
    grad_accum_steps=4
)
```

### Inference

```python
# Load trained model
model = RFDETRSegNano(pretrain_weights="/content/output/checkpoint_best_total.pth")
model.optimize_for_inference()

# Run inference
detections = model.predict(image, threshold=0.5)
```

### Visualization

The notebook includes automatic visualization with:
- Bounding boxes with class labels and confidence scores
- Instance segmentation masks
- Confidence threshold filtering

## 📈 Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|-------|
| mAP 50:95 | 0.1380 |
| mAP 50 | 0.2125 |
| mAP 75 | 0.1664 |
| mAR | 0.5158 |
| F1 Score | 0.2101 |
| Precision | 0.2242 |
| Recall | 0.2190 |

### Per-Class Performance

| Class | AP 50:95 | AR | F1 | Precision | Recall |
|-------|----------|-----|-----|-----------|--------|
| Manhole | 0.0007 | 0.2000 | 0.0000 | 0.0000 | 0.0000 |
| Pothole | 0.0826 | 0.5324 | 0.1616 | 0.2857 | 0.1127 |
| Potholes | 0.4637 | 0.7306 | 0.6788 | 0.6111 | 0.7633 |
| pothole | 0.0049 | 0.6000 | 0.0000 | 0.0000 | 0.0000 |

## 🖼️ Results

The model demonstrates:
- **Strong performance** on the "Potholes" class with 46.37% mAP
- **Good recall** across most classes, especially for "Potholes" (73.06% AR)
- **Real-time inference** capability after optimization

<Figure size 1200x1200 with 16 Axes><img width="1196" height="1196" alt="image" src="https://github.com/user-attachments/assets/13b667f4-7a6c-4664-9c83-1507ac93f7c0" />


## 📝 Notes

- The model shows varying performance across different pothole class variants
- "Potholes" class (plural) shows the best performance
- Some classes have limited training data, affecting performance

## 🔮 Future Improvements

- Data augmentation to improve generalization
- Fine-tuning hyperparameters
- Addressing class imbalance
- Ensemble methods for better detection

## 📄 License

This project is for research and educational purposes. Please ensure compliance with dataset licensing terms.

---

**Note**: This project uses the RF-DETR library from Roboflow and requires a Roboflow API key for dataset access.
