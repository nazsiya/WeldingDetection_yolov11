# 🧠 Weld Defect Detection using YOLOv11

This repository presents a solution for detecting weld defects using the YOLOv11 object detection model. The project includes scripts for dataset download, model training, and a Streamlit application for real-time crack detection from uploaded images.

## Table of Contents

- [🧠 Weld Defect Detection using YOLOv11](#-weld-defect-detection-using-yolov11)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [1. Download the Dataset](#1-download-the-dataset)
    - [2. Train the YOLOv11 Model](#2-train-the-yolov11-model)
    - [3. Run the Streamlit Application](#3-run-the-streamlit-application)
  - [Usage](#usage)
  - [Deployment (Streamlit Cloud)](#deployment-streamlit-cloud)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)

## Features

* **Automated Dataset Download:** Easily download the weld defect dataset from Roboflow.
* **YOLOv11 Model Training:** Train a custom YOLOv11 model for weld defect detection.
* **Interactive Streamlit Application:** Upload images and visualize crack detections with bounding boxes and confidence scores.
* **Detection Metrics:** Get real-time feedback on the number of defects found and class-wise confidence.

## Project Structure

```plaintext
.
├── dataset.py
├── train.py
├── stream.py
├── Weld-Classifier-1/  (Downloaded dataset will reside here)
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── best.pt  (Trained model weights will be saved here after training)
└── README.md
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.8+
* `pip` (Python package installer)

It is highly recommended to create a virtual environment to manage project dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install ultralytics streamlit roboflow pillow numpy
```

### 1. Download the Dataset

The dataset used for training is hosted on Roboflow. You will need a Roboflow API key to download it.

1. **Obtain Roboflow API Key:** If you don't have one, sign up at Roboflow and navigate to your workspace settings to find your API key.
2. **Update dataset.py:** Open dataset.py and replace "b27p4MNEi5OqwUKepU06" with your actual Roboflow API key.

```python
# dataset.py
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY") # <--- REPLACE THIS
project = rf.workspace("defspace").project("weld-classifier")
version = project.version(1)
dataset = version.download("yolov11")
```

3. **Run the script to download the dataset:**

```bash
python dataset.py
```

This will download the dataset into a directory named Weld-Classifier-1 in your project root. The data.yaml file, crucial for training, will be located inside this directory.

### 2. Train the YOLOv11 Model

Once the dataset is downloaded, you can proceed to train the YOLOv11 model.

1. **Review train.py:** The train.py script is configured to train a yolov11m.pt model for 100 epochs with an image size of 640 and a batch size of 16. You can adjust these parameters based on your computational resources and desired training duration.

```python
# train.py
from ultralytics import YOLO
import os

model = YOLO("yolov11m.pt") # You can change to "yolov11n.pt" or any other variant

results = model.train(
    data="Weld-Classifier-1/data.yaml", # Relative path from current directory
    epochs=100,
    imgsz=640,
    batch=16, # Adjust as per your GPU capacity
    name="weld_defect_v1" # Custom run name
)

best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
print(f"✅ Training complete. Best model saved at: {best_model_path}")

model = YOLO(best_model_path)
metrics = model.val()
print(f"📊 Validation complete.")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

2. **Start Training:**

```bash
python train.py
```

Training can take a significant amount of time depending on your hardware (GPU recommended). Upon completion, the trained model's weights (best.pt) will be saved in a directory structure like runs/detect/weld_defect_v1/weights/best.pt. The script will also print the exact path to the best.pt file. For the Streamlit application, you will need to ensure best.pt is directly accessible or its path is correctly specified in stream.py. It's recommended to copy best.pt to the root of your project for simplicity.

### 3. Run the Streamlit Application

The stream.py script provides a user-friendly interface for crack detection.

1. **Ensure Model Availability:** Make sure the best.pt file (the trained model weights) is in the same directory as stream.py, or update the model_path variable in stream.py to point to its correct location.

```python
# stream.py
# ...
model_path = "best.pt" # Ensure this path is correct relative to stream.py
model = YOLO(model_path)
# ...
```

2. **Launch the Streamlit App:**

```bash
streamlit run stream.py
```

This command will open the Streamlit application in your web browser, typically at http://localhost:8501.

## Usage

* **Upload Image:** In the Streamlit application, use the "Choose an image" button to upload a JPG, JPEG, or PNG file.
* **View Detections:** The application will process the image using the trained YOLOv11 model and display the image with detected cracks, bounding boxes, and confidence scores.
* **Review Metrics:** Below the detected image, you will see the total number of defects found and a class-wise breakdown with individual confidence scores.

## Deployment (Streamlit Cloud)

You can deploy this application live using Streamlit Cloud.

1. **Create a GitHub Repository:** Ensure your project is pushed to a public GitHub repository.
2. **Go to Streamlit Cloud:** Visit share.streamlit.io.
3. **Connect to GitHub:** Link your Streamlit Cloud account to your GitHub repository.
4. **New App:** Click "New app" and select your repository and the stream.py file as the main file.
5. **Environment Variables (if any):** If your dataset.py (or any other script) relies on sensitive information like API keys, consider using Streamlit's secrets management. For the provided dataset.py, if you only run it locally for training, you don't need to add the Roboflow API key as a secret for the deployed Streamlit app, as the deployed app only uses best.pt.
6. **Specify Python Version and Dependencies:** Streamlit will automatically try to detect your dependencies from a requirements.txt file. It's highly recommended to create one:

```bash
pip freeze > requirements.txt
```

Ensure ultralytics, streamlit, Pillow, numpy, and roboflow are listed in requirements.txt.

7. **DeployLink:** https://weldingdetectionyolov11-ddxhagvjbwflxhysi4uf8y.streamlit.app/

## Acknowledgments

* Ultralytics YOLOv11 for the powerful object detection framework.
* Roboflow for dataset management and hosting.
* Streamlit for providing an excellent framework for building interactive web applications.
