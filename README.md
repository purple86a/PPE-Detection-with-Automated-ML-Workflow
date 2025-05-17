# YOLOv8 MLOps Pipeline 🚀

A complete end-to-end MLOps pipeline for training, evaluating, and serving YOLOv8 object detection models with integrated experiment tracking through MLflow. Gradio App for trying the model: [Link](https://huggingface.co/spaces/purplewater/Personal-protective-equipment-checker-app)

## 📋 Project Overview

This project implements a modular MLOps pipeline for YOLOv8 models with automated data preprocessing, hyperparameter tuning, model training, evaluation, and deployment. Everything is tracked with MLflow to ensure experiment reproducibility and model versioning.

## 🗂️ Data Structure

The project expects data in standard YOLOv8 format:
```
project/
├── data.yaml         # Dataset configuration
├── datasets/
│   ├── images/
│   │   ├── train/    # Training images
│   │   ├── val/      # Validation images
│   │   └── test/     # Test images
│   └── labels/
│       ├── train/    # Training labels (YOLO format)
│       ├── val/      # Validation labels
│       └── test/     # Test labels
└── orig_dataset/     # Optional backup of original data
```

## 🔧 Quick Start

Run the complete pipeline with:
```bash
python run_mlflow_pipeline.py
```

Or run individual components as needed:
```bash
python preprocessing.py --src orig_dataset --dst datasets
python hparam_tune.py --monitor
python predict_and_compare.py --data data.yaml
python serve_model.py --register --serve
```

## 📚 Script Documentation

### run_mlflow_pipeline.py
**The main orchestrator that runs all pipeline steps in sequence.**

Arguments:
- `--skip-preprocessing`: Skip data preprocessing step
- `--skip-tuning`: Skip hyperparameter tuning step
- `--skip-training`: Skip model training step
- `--skip-evaluation`: Skip model evaluation step
- `--skip-serving`: Skip model serving step
- `--cpu`: Force using CPU even if CUDA is available
- `--keep-examples`: Number of example directories to keep (default: 10)

Example:
```bash
python run_mlflow_pipeline.py --cpu --skip-preprocessing
```

---

### preprocessing.py
**Applies noise filtering to images and prepares datasets.**

Arguments:
- `--src`: Source directory (default: ".")
- `--dst`: Destination directory (default: "datasets")
- `--noise`: Noise level for filtering (default: 0.0)
- `--run-name`: Custom name for the MLflow run

Example:
```bash
python preprocessing.py --src orig_dataset --dst datasets --noise 0.0
```

---

### augment_data.py
**Augments training data with various transformations to improve model robustness.**

Arguments:
- `--src`: Source directory (default: "orig_dataset")
- `--dst`: Destination directory (default: "datasets")
- `--level`: Augmentation level - light, medium, or heavy (default: medium)
- `--noise`: Enable noise filtering (default: True)
- `--num-aug`: Number of augmentations per image (default: 2)
- `--run-name`: Custom name for the MLflow run

Example:
```bash
python augment_data.py --level heavy --num-aug 3
```

---

### hparam_tune.py
**Performs hyperparameter tuning using Optuna and trains the final model.**

Arguments:
- `--cpu`: Force using CPU even if CUDA is available
- `--new-db`: Delete existing database and start fresh
- `--deploy`: Deploy the best model with MLflow using serve_model.py
- `--monitor`: Plot the loss during training

Example:
```bash
python hparam_tune.py --monitor --cpu
```

---

### predict_and_compare.py
**Evaluates model performance against ground truth and generates visualizations.**

Arguments:
- `--model`: Path to the model weights file
- `--model-path`: Alternative parameter for model path
- `--model-dir`: Directory containing the model (used for naming)
- `--data`: Path to the data YAML file (default: "data.yaml")
- `--conf`: Confidence threshold for predictions (default: 0.25)
- `--output`: Output directory for results (default: "artifacts/prediction_analysis")
- `--run-name`: Custom name for the MLflow run

Example:
```bash
python predict_and_compare.py --data data.yaml --conf 0.3
```

---

### evaluate_augmented_data.py
**Compares model performance on original vs. augmented test data.**

Arguments:
- `--model`: Path to the model weights file
- `--model-path`: Alternative parameter for model path
- `--model-dir`: Directory containing the model (used for naming)
- `--data`: Path to the data YAML file (default: "data_aug.yaml")
- `--batch`: Batch size (default: 8)
- `--workers`: Number of workers (default: 2)
- `--device`: Device to use (default: "0")
- `--metrics-dir`: Directory for test metrics (default: "artifacts/test_metrics")

Example:
```bash
python evaluate_augmented_data.py --model-path runs/train/yolo_best_params1/weights/best.pt
```

---

### class_distribution.py
**Analyzes and visualizes the class distribution in your dataset.**

No arguments; automatically reads data from 'data.yaml' and datasets directory.

Example:
```bash
python class_distribution.py
```

---

### serve_model.py
**Registers the model with MLflow and/or serves it via FastAPI.**

Arguments:
- `--model`: Path to the model weights file
- `--model-path`: Alternative parameter for model path
- `--model-dir`: Directory containing the model (used for naming)
- `--data`: Path to the data YAML file (default: "data.yaml")
- `--register`: Register the model with MLflow
- `--serve`: Serve the model with FastAPI
- `--port`: Port to serve the model on (default: 8000)

Example:
```bash
python serve_model.py --register --serve --port 8000
```

## 🌐 Web Interface

When serving the model, a user-friendly web interface is available at http://localhost:8000 that allows users to:
- Upload images for object detection
- Adjust confidence thresholds
- View detection results with bounding boxes

## 📈 MLflow Tracking

Access the MLflow UI to view experiments, compare runs, and examine metrics:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then open http://localhost:5000 in your browser.

## 📦 Dependencies

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- MLflow
- FastAPI
- Optuna
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests for any improvements or bug fixes.

## 📄 License

This project is open-source and available under the MIT License. 
