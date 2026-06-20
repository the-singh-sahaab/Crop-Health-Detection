# Crop-Health-Detection

## Apple, Cucumber, Tomato, and Grape Crop Health Detection (YOLOv12)

This repository contains the training workflow and validation results for a custom agricultural computer vision model designed to monitor crop health. The model is trained to detect diseases and assess the health status of leaves and crops across four main categories: **Apple, Cucumber, Tomato, and Grape**. A research Paper about the same is underreview, will include it in future commits. 

---

## 🔒 Data Privacy & Proprietary Information Notice

> [!IMPORTANT]  
> The training, validation, and testing datasets used in this project contain proprietary agricultural data. Consequently, **the raw dataset files, sample images, and specific data annotations are private and cannot be shared or demonstrated here**. 
> 
> To respect data privacy:
> - No raw images from the dataset are stored or previewed in this repository.
> - The directory paths referenced in the code (e.g., `/content/apple-cucumber-tomato-grape.v1i.yolov12/`) point to secure environments where the private dataset was mounted during model execution.
> - Analysis and metrics presented below are extracted directly from the training and validation execution logs.

---

## 🛠️ Model Architecture & Training Setup

The model leverages the state-of-the-art **YOLOv12 Large (`yolo12l.pt`)** object detection architecture, fine-tuned for high-precision multi-class agricultural detection. 

### Training Configuration
* **Framework**: Ultralytics (v8.3.222)
* **Base Model**: YOLO12l (Large)
* **Input Resolution**: $640 \times 640$ pixels
* **Epochs**: 100
* **Batch Size**: 16
* **Optimization**: Automatic optimizer selection (with learning rate $\text{lr0}=0.01$, momentum $= 0.937$, weight decay $= 0.0005$)
* **Mixed Precision (AMP)**: Enabled (`amp=True`)
* **Hardware**: Trained on an `NVIDIA A100-SXM4-40GB` GPU (CUDA:0)

### Class Structure
The dataset defines **22 distinct classes** (`nc=22`) mapping to various health conditions of the target crops. These cover healthy leaves, as well as specific diseases, pests, and physiological disorders, including:
* **Grape**: `GrapeLeaf_healthy`, `GrapeLeaf_unhealthy_blackMeasles`, etc.
* **Cucumber**: `CucumberLeaf_unhealthy`, etc.
* **Tomato**: `Tomato_healthy`, `TomatoLeaf_unhealthy_spiderMites`, `TomatoLeaf_unhealthy_targetSpot`, etc.
* **Apple**: `Apple_healthy`, etc.

---

## 📈 Performance & Evaluation Summary

Validation of the trained YOLOv12 model was performed programmatically on the holdout validation subset. The model demonstrated excellent classification and localization capabilities.

### Overall Metrics
| Metric | Value | Description |
| :--- | :---: | :--- |
| **Precision** | **94.3%** | Proportion of true positive detections out of all positive predictions. |
| **Recall** | **93.6%** | Proportion of true positive detections identified out of all actual objects. |
| **mAP@50** | **94.4%** | mean Average Precision computed at an Intersection over Union (IoU) threshold of 0.50. |
| **mAP@50-95** | **85.5%** | mean Average Precision averaged over IoU thresholds ranging from 0.50 to 0.95. |

### Class-Specific Highlights
The model achieved near-perfect average precision on several challenging classification tasks:
* **Tomato Leaf Spider Mites (`TomatoLeaf_unhealthy_spiderMites`)**: 
  * mAP@50: **99.5%** | mAP@50-95: **89.7%**
* **Tomato Leaf Target Spot (`TomatoLeaf_unhealthy_targetSpot`)**: 
  * mAP@50: **99.4%** | mAP@50-95: **87.1%**
* **Grape Leaf Healthy (`GrapeLeaf_healthy`)**: 
  * mAP@50: **99.1%**
* **Grape Leaf Black Measles (`GrapeLeaf_unhealthy_blackMeasles`)**: 
  * mAP@50: **97.7%**
* **Cucumber Leaf Unhealthy (`CucumberLeaf_unhealthy`)**: 
  * mAP@50: **97.6%**
* **Tomato Healthy (`Tomato_healthy`)**:
  * mAP@50: **70.9%** | mAP@50-95: **58.0%** *(lower due to variations in healthy fruit appearance and leaf occlusion)*

*Note: Preprocessing, inference, and postprocessing speeds clocked in at an average of **0.1ms, 4.2ms, and 1.4ms** per image respectively during batch validation.*

---

## 🔍 Inference & Usage Example

The trained weights can be loaded using the Ultralytics Python API to run inference on new imagery. The following snippet outlines the inference process used in the notebook:

```python
import cv2
from ultralytics import YOLO

# 1. Load the fine-tuned YOLOv12 model weights
model = YOLO('path/to/weights/best.pt')

# 2. Perform object detection on a test crop image
image_path = 'path/to/test_image.jpg'
img = cv2.imread(image_path)
results = model.predict(source=img, imgsz=640)

# 3. Process detection results
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = box.conf[0]
    print(f"Detected class ID: {class_id} | Confidence: {confidence:.2f}")
```

### Demonstration Log Output
An execution of the inference pipeline on a test image containing healthy apples yielded the following metrics:
* **Detection Outcome**: Detected **5 instances of `Apple_healthy`**
* **Confidence Scores**: `0.52`, `0.36`, `0.36`, `0.31`, `0.29`
* **Inference Speed**: 
  * Preprocess: **3.0ms**
  * Inference: **103.7ms**
  * Postprocess: **3.1ms**
  *(Recorded at an image resolution of 480 * 640)*
