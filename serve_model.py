import os
import sys
import argparse
import mlflow
import mlflow.pyfunc
import json
import yaml
import numpy as np
import cv2
from PIL import Image
import io
import base64
from ultralytics import YOLO
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time

class YOLOv8Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        self.class_names = None
        
    def load_context(self, context):
        """Load the model from the MLflow artifact."""
        model_path = context.artifacts["model"]
        
        # Load class names from data.yaml if available
        data_yaml_path = context.artifacts.get("data_yaml")
        if data_yaml_path and os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', {})
        
        # Load the model
        self.model = YOLO(model_path)
        
    def predict(self, context, model_input):
        """Make predictions using the model."""
        # Check input type
        if isinstance(model_input, np.ndarray):
            # Input is a numpy array (image)
            return self._predict_image(model_input)
        elif isinstance(model_input, dict):
            # Input is a dictionary
            if "image" in model_input:
                # Process base64 encoded image
                if isinstance(model_input["image"], str):
                    try:
                        # Decode base64 image
                        img_data = base64.b64decode(model_input["image"])
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return self._predict_image(img, model_input.get("conf", 0.25))
                    except Exception as e:
                        return {"error": f"Failed to decode image: {str(e)}"}
                else:
                    return {"error": "Image must be base64 encoded string"}
            elif "image_path" in model_input:
                # Process image from path
                try:
                    img = cv2.imread(model_input["image_path"])
                    if img is None:
                        return {"error": f"Failed to load image from {model_input['image_path']}"}
                    return self._predict_image(img, model_input.get("conf", 0.25))
                except Exception as e:
                    return {"error": f"Failed to load image: {str(e)}"}
            else:
                return {"error": "Input must contain 'image' or 'image_path' key"}
        else:
            return {"error": "Unsupported input type"}
    
    def _predict_image(self, img, conf=0.25):
        """Make predictions on a single image."""
        try:
            # Run inference
            results = self.model.predict(img, conf=conf)
            
            # Process results
            detections = []
            for i, result in enumerate(results):
                boxes = result.boxes
                
                for j, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get class and confidence
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    
                    # Get class name if available
                    if self.class_names and cls_id < len(self.class_names):
                        cls_name = self.class_names[cls_id]
                    else:
                        cls_name = f"class_{cls_id}"
                    
                    # Add detection
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })
            
            return {
                "detections": detections,
                "count": len(detections)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

def register_model(model_path, data_yaml_path="data.yaml", model_name="yolov8-object-detection"):
    """Register the model with MLflow."""
    # Set MLflow tracking URI from environment variable
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create a new experiment for model serving
    experiment_name = "model-serving"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            # If experiment is deleted, create a new one with a timestamp
            import time
            new_experiment_name = f"{experiment_name}-{int(time.time())}"
            print(f"Experiment '{experiment_name}' is deleted. Creating new experiment '{new_experiment_name}'")
            mlflow.create_experiment(new_experiment_name)
            experiment_name = new_experiment_name
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        # Fall back to default experiment
        print("Using default experiment")
    
    # Create a meaningful run name based on the model
    model_basename = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    parent_dir = os.path.basename(os.path.dirname(model_dir)) if os.path.dirname(model_dir) else ""
    run_name = f"Model-Registration-{parent_dir}-{model_basename.replace('.pt', '')}"
    
    # Start a new run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "model_path": model_path,
            "data_yaml": data_yaml_path
        })
        
        # Log artifacts
        mlflow.log_artifact(model_path, "model")
        if os.path.exists(data_yaml_path):
            mlflow.log_artifact(data_yaml_path, "data")
        
        # Define artifacts to pass to the model
        artifacts = {
            "model": model_path,
            "data_yaml": data_yaml_path if os.path.exists(data_yaml_path) else None
        }
        
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YOLOv8Wrapper(),
            artifacts=artifacts,
            pip_requirements=[
                "ultralytics>=8.0.0",
                "torch>=1.7.0",
                "opencv-python>=4.1.2",
                "pyyaml>=5.1",
                "numpy>=1.18.5",
                "pillow>=7.1.2",
                "fastapi>=0.68.0",
                "uvicorn>=0.15.0",
                "python-multipart>=0.0.5"
            ],
            code_path=["serve_model.py"]  # Include this file in the model
        )
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"Model registered as: {model_name} (version {registered_model.version})")
        return model_uri

# Pydantic model for prediction request
class PredictionRequest(BaseModel):
    image: Optional[str] = None
    image_path: Optional[str] = None
    conf: Optional[float] = 0.25

def create_fastapi_app(model_uri=None):
    """Create a FastAPI app for serving the model."""
    app = FastAPI(title="YOLOv8 Object Detection API")
    
    # Create templates directory
    os.makedirs("templates", exist_ok=True)
    templates = Jinja2Templates(directory="templates")
    
    # Global variable to store the model
    model = None
    
    # Create index.html if it doesn't exist
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
        }
        #resultImage {
            max-width: 100%;
            margin-top: 10px;
        }
        canvas {
            border: 1px solid #ddd;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>YOLOv8 Object Detection</h1>
    
    <div class="upload-form">
        <h2>Upload an image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageInput" name="file" accept="image/*">
            <br><br>
            <label for="confThreshold">Confidence threshold:</label>
            <input type="range" id="confThreshold" name="conf" min="0.1" max="1.0" step="0.05" value="0.25">
            <span id="confValue">0.25</span>
            <br><br>
            <button type="submit">Detect Objects</button>
        </form>
    </div>
    
    <div class="result" id="resultContainer" style="display: none;">
        <h2>Detection Results</h2>
        <p id="detectionCount"></p>
        <canvas id="resultCanvas"></canvas>
    </div>

    <script>
        // Update confidence value display
        document.getElementById('confThreshold').addEventListener('input', function() {
            document.getElementById('confValue').textContent = this.value;
        });
        
        // Handle form submission
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const imageFile = document.getElementById('imageInput').files[0];
            const confThreshold = document.getElementById('confThreshold').value;
            
            if (!imageFile) {
                alert('Please select an image file');
                return;
            }
            
            formData.append('file', imageFile);
            formData.append('conf', confThreshold);
            
            // Show loading indicator
            document.getElementById('resultContainer').style.display = 'block';
            document.getElementById('detectionCount').textContent = 'Processing...';
            
            // Send request to server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display results
                document.getElementById('detectionCount').textContent = 
                    `Found ${data.count} objects`;
                
                // Draw image and bounding boxes
                const canvas = document.getElementById('resultCanvas');
                const ctx = canvas.getContext('2d');
                
                const img = new Image();
                img.onload = function() {
                    // Set canvas size to match image
                    canvas.width = img.width;
                    canvas.height = img.height;
                    
                    // Draw image
                    ctx.drawImage(img, 0, 0);
                    
                    // Draw bounding boxes
                    data.detections.forEach(det => {
                        const [x1, y1, x2, y2] = det.bbox;
                        const width = x2 - x1;
                        const height = y2 - y1;
                        
                        // Draw rectangle
                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x1, y1, width, height);
                        
                        // Draw label
                        ctx.fillStyle = 'red';
                        ctx.font = '16px Arial';
                        ctx.fillText(
                            `${det.class_name} (${det.confidence.toFixed(2)})`, 
                            x1, y1 > 20 ? y1 - 5 : y1 + 20
                        );
                    });
                };
                
                // Load the image from the file input
                const reader = new FileReader();
                reader.onload = function(e) {
                    img.src = e.target.result;
                };
                reader.readAsDataURL(imageFile);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during processing');
            });
        });
    </script>
</body>
</html>
            """)
    
    # Use modern lifespan approach instead of on_event
    @asynccontextmanager
    async def lifespan(app):
        # Load the model at startup
        nonlocal model
        nonlocal model_uri
        
        print("Loading model...")
        if model_uri is None:
            # Try to find the latest version of the registered model
            try:
                client = mlflow.tracking.MlflowClient()
                model_name = "yolov8-object-detection"
                
                # Check if model exists in registry
                try:
                    latest_versions = client.get_latest_versions(model_name)
                    if latest_versions:
                        model_uri = f"models:/{model_name}/{latest_versions[0].version}"
                        print(f"Loading model from: {model_uri}")
                        model = mlflow.pyfunc.load_model(model_uri)
                        print("Model loaded successfully!")
                    else:
                        print(f"No versions found for model {model_name}")
                except mlflow.exceptions.MlflowException as e:
                    if "not found" in str(e):
                        print(f"Model {model_name} not found in registry.")
                        # Try to find the model in the local file system
                        best_params_dirs = sorted([d for d in os.listdir("runs/train") if d.startswith("yolo_best_params")]) if os.path.exists("runs/train") else []
                        if best_params_dirs:
                            latest_dir = best_params_dirs[-1]
                            model_path = f"runs/train/{latest_dir}/weights/best.pt"
                            if os.path.exists(model_path):
                                print(f"Loading model from file: {model_path}")
                                model = YOLO(model_path)
                                print("Model loaded successfully from local path!")
                    else:
                        raise e
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            # Direct model path provided
            try:
                if os.path.exists(model_uri):
                    print(f"Loading model from file: {model_uri}")
                    model = YOLO(model_uri)
                    print("Model loaded successfully from provided path!")
                else:
                    print(f"Model file not found at: {model_uri}")
            except Exception as e:
                print(f"Error loading model from path: {e}")
        
        yield
        
        # Cleanup (if needed)
        print("Shutting down and cleaning up...")
    
    # Set the lifespan handler
    app.router.lifespan_context = lifespan
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Render the home page."""
        return templates.TemplateResponse("index.html", {"request": request})
    
    @app.post("/predict")
    async def predict(
        file: Optional[UploadFile] = File(None), 
        conf: Optional[float] = Form(None),
        request: Request = None
    ):
        """Make predictions on the input data."""
        nonlocal model
        
        if model is None:
            # Try to load model one more time
            try:
                client = mlflow.tracking.MlflowClient()
                model_name = "yolov8-object-detection"
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    model_uri = f"models:/{model_name}/{latest_versions[0].version}"
                    model = mlflow.pyfunc.load_model(model_uri)
                    print("Model loaded successfully on demand!")
                else:
                    # Try to find the model in the local file system
                    best_params_dirs = sorted([d for d in os.listdir("runs/train") if d.startswith("yolo_best_params")]) if os.path.exists("runs/train") else []
                    if best_params_dirs:
                        latest_dir = best_params_dirs[-1]
                        model_path = f"runs/train/{latest_dir}/weights/best.pt"
                        if os.path.exists(model_path):
                            print(f"Loading model from file: {model_path}")
                            model = YOLO(model_path)
                            print("Model loaded successfully from local path on demand!")
            except Exception as e:
                print(f"Error loading model on demand: {e}")
                
        if model is None:
            raise HTTPException(status_code=500, detail="No model loaded. Please register a model first.")
        
        # Set default confidence if not provided
        if conf is None:
            conf = 0.25
        
        # Set up MLflow tracking for this prediction
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = "model-serving"
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error setting up MLflow experiment: {e}")
        
        # Create a meaningful run name
        timestamp = int(time.time())
        if file:
            source_type = "UploadedFile"
            source_name = file.filename
        else:
            try:
                json_data = await request.json()
                if "image" in json_data:
                    source_type = "Base64Image"
                    source_name = "api_request"
                elif "image_path" in json_data:
                    source_type = "ImagePath"
                    source_name = os.path.basename(json_data["image_path"])
                else:
                    source_type = "Unknown"
                    source_name = "api_request"
            except:
                source_type = "Unknown"
                source_name = "api_request"
        
        run_name = f"Prediction-{source_type}-{source_name}-Conf{conf}"
            
        # Handle file upload
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("confidence_threshold", conf)
            mlflow.log_param("source_type", source_type)
            
            result = None
            
            if file:
                contents = await file.read()
                img = Image.open(io.BytesIO(contents))
                
                # Convert to RGB if image has alpha channel (RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    
                img_array = np.array(img)
                
                # Make prediction
                if isinstance(model, YOLO):
                    results = model.predict(img_array, conf=conf)
                    result = process_yolo_results(results, model)
                else:
                    result = model.predict({"image": img_array, "conf": conf})
                
                # Log the image as an artifact
                img_path = f"temp_{timestamp}.jpg"
                img.save(img_path)
                mlflow.log_artifact(img_path, "input_image")
                os.remove(img_path)
            
            # Handle JSON request
            else:
                try:
                    # Try to parse JSON body
                    json_data = await request.json()
                    
                    # Handle base64 image
                    if "image" in json_data and isinstance(json_data["image"], str):
                        try:
                            # Decode base64 image
                            img_data = base64.b64decode(json_data["image"])
                            nparr = np.frombuffer(img_data, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Force 3 channels (BGR)
                            
                            # Get confidence threshold
                            request_conf = json_data.get("conf", conf)
                            mlflow.log_param("confidence_threshold", request_conf)
                            
                            # Make prediction
                            if isinstance(model, YOLO):
                                results = model.predict(img, conf=request_conf)
                                result = process_yolo_results(results, model)
                            else:
                                result = model.predict({"image": img, "conf": request_conf})
                            
                            # Log a sample of the image
                            img_path = f"temp_{timestamp}.jpg"
                            cv2.imwrite(img_path, img)
                            mlflow.log_artifact(img_path, "input_image")
                            os.remove(img_path)
                        except Exception as e:
                            raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
                    
                    # Handle image path
                    elif "image_path" in json_data:
                        image_path = json_data["image_path"]
                        if not os.path.exists(image_path):
                            raise HTTPException(status_code=404, detail=f"Image not found at {image_path}")
                        
                        # Get confidence threshold
                        request_conf = json_data.get("conf", conf)
                        mlflow.log_param("confidence_threshold", request_conf)
                        mlflow.log_param("image_path", image_path)
                        
                        # Make prediction
                        if isinstance(model, YOLO):
                            results = model.predict(image_path, conf=request_conf)
                            result = process_yolo_results(results, model)
                        else:
                            result = model.predict({"image_path": image_path, "conf": request_conf})
                        
                        # Log the image as an artifact
                        mlflow.log_artifact(image_path, "input_image")
                    else:
                        raise HTTPException(status_code=400, detail="Request must contain either 'image' (base64) or 'image_path'")
                
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON format")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
            
            # Log results
            if result:
                # Log detection count as a metric
                if isinstance(result, dict) and "count" in result:
                    mlflow.log_metric("detection_count", result["count"])
                
                # Log result as JSON artifact
                result_path = f"result_{timestamp}.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f)
                mlflow.log_artifact(result_path, "prediction_result")
                os.remove(result_path)
            
            return result
            
    def process_yolo_results(results, model_obj):
        """Process YOLOv8 results into a standardized format."""
        # Get class names if available
        class_names = {}
        if os.path.exists("data.yaml"):
            with open("data.yaml", 'r') as f:
                data = yaml.safe_load(f)
                class_names = data.get('names', {})
                
        detections = []
        for i, result in enumerate(results):
            boxes = result.boxes
            
            for j, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get class and confidence
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                
                # Get class name if available
                if class_names and cls_id < len(class_names):
                    cls_name = class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"
                
                # Add detection
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })
        
        return {
            "detections": detections,
            "count": len(detections)
        }
            
    return app

def main():
    parser = argparse.ArgumentParser(description="Serve YOLOv8 model with FastAPI")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to the model weights file (if not provided, will use the latest registered model)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the model weights file (alternative to --model)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing the model (used for naming)")
    parser.add_argument("--data", type=str, default="data.yaml", 
                        help="Path to the data YAML file")
    parser.add_argument("--register", action="store_true", 
                        help="Register the model with MLflow")
    parser.add_argument("--serve", action="store_true", 
                        help="Serve the model with FastAPI")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to serve the model on")
    args = parser.parse_args()
    
    # Use model-path if provided (for compatibility with run_mlflow_pipeline.py)
    if args.model_path is not None:
        args.model = args.model_path
    
    # Register the model if requested
    if args.register:
        if args.model is None:
            # Try to find the latest best_params directory
            best_params_dirs = sorted([d for d in os.listdir("runs/train") if d.startswith("yolo_best_params")]) if os.path.exists("runs/train") else []
            if best_params_dirs:
                latest_dir = best_params_dirs[-1]
                args.model = f"runs/train/{latest_dir}/weights/best.pt"
                print(f"Using latest model: {args.model}")
            else:
                print("No model found. Please specify a model path with --model")
                return
        
        # Check if model file exists
        if not os.path.exists(args.model):
            print(f"Error: Model file not found at {args.model}")
            return
            
        try:
            model_uri = register_model(args.model, args.data)
            print(f"Model registered with URI: {model_uri}")
            print("To serve the model, run: mlflow models serve -m models:/yolov8-object-detection/latest -p 5000")
        except Exception as e:
            print(f"Error registering model: {e}")
            return
    
    # Serve the model if requested
    if args.serve:
        try:
            # Create the FastAPI app
            app = create_fastapi_app(args.model)
            
            # Run the app
            print(f"Starting FastAPI server on port {args.port}...")
            print(f"Open http://localhost:{args.port} in your browser to use the web interface")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except Exception as e:
            print(f"Error serving model: {e}")
            return
    
    # If neither register nor serve was specified, show help
    if not args.register and not args.serve:
        parser.print_help()

if __name__ == "__main__":
    main() 