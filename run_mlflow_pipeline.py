#!/usr/bin/env python
import os
import argparse
import subprocess
import time
import mlflow
import sys
import shutil
import signal
import atexit
import requests
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_command(cmd, description=None, run_name=None):
    """
    Run a command and print its output.
    """
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}")
    
    python_exe = sys.executable
    cmd = cmd.replace("python ", f'"{python_exe}" ')
    
    if run_name:
        if "preprocessing.py" in cmd:
           
            run_name_value = f"Preprocessing-NoiseFiltering-{int(time.time())}"
            cmd = cmd.replace("preprocessing.py", f"preprocessing.py --run-name \"{run_name_value}\"")
            
        elif "augment_data.py" in cmd:
            
            import re
            level_match = re.search(r'--level\s+(\w+)', cmd)
            level = level_match.group(1) if level_match else "medium"
            run_name_value = f"Augmentation-{level}-filter-{int(time.time())}"
            cmd = cmd.replace("augment_data.py", f"augment_data.py --run-name \"{run_name_value}\"")
            
        elif "predict_and_compare.py" in cmd:

            import re
            data_match = re.search(r'--data\s+(\S+)', cmd)
            output_match = re.search(r'--output\s+(\S+)', cmd)
            data_yaml = data_match.group(1) if data_match else "data.yaml"
            output_dir = output_match.group(1) if output_match else "prediction_analysis"
            data_type = "Original" if "data.yaml" in data_yaml else "Augmented"
            cmd = cmd.replace("predict_and_compare.py", f"predict_and_compare.py --run-name \"Evaluation-{data_type}-Dataset\"")
    
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}  # Optional
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def setup_mlflow_server():
    """
    Setup and start the MLflow tracking server.
    """

    if not os.path.exists("mlruns.db"):
        print("Creating MLflow database...")
        with open("mlruns.db", "w") as f:
            pass
    
    print("Starting MLflow server...")
    mlflow_cmd = f'"{sys.executable}" -m mlflow ui --backend-store-uri sqlite:///mlruns.db'
    server_process = subprocess.Popen(
        mlflow_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}  # Optional
    )
    
    time.sleep(5)
    print("MLflow server started. Access the UI at http://localhost:5000")
    
    return server_process

def ensure_directories():
    """
    Ensure all required directories exist.
    """
    required_dirs = [
        "runs",
        "runs/train",
        "runs/test",
        "artifacts",
        "artifacts/preprocessing",
        "artifacts/test_metrics",
        "artifacts/prediction_analysis",
        "templates",
        "datasets/images/train",
        "datasets/images/val",
        "datasets/images/test",
        "datasets/labels/train",
        "datasets/labels/val",
        "datasets/labels/test",
        "orig_dataset/images/train",
        "orig_dataset/images/val", 
        "orig_dataset/images/test",
        "orig_dataset/labels/train",
        "orig_dataset/labels/val",
        "orig_dataset/labels/test",
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def prepare_orig_dataset():
    """
    Check if orig_dataset directory exists or prepare it from datasets directory.
    """
    
    if os.path.exists('orig_dataset'):
        print("Using orig_dataset directory for augmentation...")
        return True
    elif os.path.exists('datasets'):
        print("Copying datasets to orig_dataset directory for augmentation...")
        
        os.makedirs('orig_dataset', exist_ok=True)
        
        dirs_to_copy = [
            ('datasets/images/train', 'orig_dataset/images/train'),
            ('datasets/images/val', 'orig_dataset/images/val'),
            ('datasets/images/test', 'orig_dataset/images/test'),
            ('datasets/labels/train', 'orig_dataset/labels/train'),
            ('datasets/labels/val', 'orig_dataset/labels/val'),
            ('datasets/labels/test', 'orig_dataset/labels/test')
        ]
        
        for src, dst in dirs_to_copy:
            if os.path.exists(src):
                print(f"Copying {src} to {dst}...")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                for item in os.listdir(src):
                    s = os.path.join(src, item)
                    d = os.path.join(dst, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
        
        if os.path.exists('datasets/data.yaml'):
            shutil.copy2('datasets/data.yaml', 'orig_dataset/data.yaml')
        elif os.path.exists('data.yaml'):
            shutil.copy2('data.yaml', 'orig_dataset/data.yaml')
            
        print("Dataset copied successfully to orig_dataset.")
        return True
    else:
        print("Warning: Neither orig_dataset nor datasets directory found.")
        return False

def cleanup_old_examples(max_dirs=10):
    """
    Clean up old example directories if there are too many.
    """
    import glob
    import os
    
    artifacts_dir = 'artifacts'
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        print(f"Created artifacts directory at '{artifacts_dir}'")
        return  # No cleanup needed for a fresh directory
        
    artifact_structure = {
        'augmentation': {'pattern': 'examples_*'},
        'preprocessing': {'pattern': 'examples_*'},
        'test_metrics': {'pattern': 'run_*'},
        'prediction_analysis': {'pattern': '*_run_*'} 
    }
    
    for artifact_type, config in artifact_structure.items():
        type_dir = os.path.join(artifacts_dir, artifact_type)
        
        if not os.path.exists(type_dir):
            os.makedirs(type_dir, exist_ok=True)
            continue
            
        pattern = config['pattern']
        example_dirs = [d for d in os.listdir(type_dir) 
                       if os.path.isdir(os.path.join(type_dir, d)) and 
                       (pattern == '*_run_*' or d.startswith(pattern.replace('*', '')))]
        
        if artifact_type == 'prediction_analysis':
            by_type = {"original": [], "augmented": []}
            for dir_name in example_dirs:
                if dir_name.startswith("original"):
                    by_type["original"].append(dir_name)
                elif dir_name.startswith("augmented"):
                    by_type["augmented"].append(dir_name)
            
            for data_type, dirs in by_type.items():
                sorted_dirs = sorted(dirs, key=lambda x: int(x.split("_run_")[1]) if "_run_" in x else 0)
                
                num_dirs_to_remove = max(0, len(sorted_dirs) - max_dirs)
                if num_dirs_to_remove > 0:
                    print(f"Cleaning up {num_dirs_to_remove} old {artifact_type}/{data_type} directories...")
                    for dir_name in sorted_dirs[:num_dirs_to_remove]:
                        dir_path = os.path.join(type_dir, dir_name)
                        try:
                            shutil.rmtree(dir_path)
                            print(f"  Removed {dir_path}")
                        except Exception as e:
                            print(f"  Failed to remove {dir_path}: {e}")
        else:
            sorted_dirs = sorted(example_dirs)
            
            num_dirs_to_remove = max(0, len(sorted_dirs) - max_dirs)
            if num_dirs_to_remove > 0:
                print(f"Cleaning up {num_dirs_to_remove} old {artifact_type} directories...")
                for dir_name in sorted_dirs[:num_dirs_to_remove]:
                    dir_path = os.path.join(type_dir, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  Removed {dir_path}")
                    except Exception as e:
                        print(f"  Failed to remove {dir_path}: {e}")
    
    old_dirs_patterns = [
        'augmentation_examples_*', 
        'preprocessing_examples_*', 
        'test_metrics_*', 
        'prediction_analysis_*'
    ]
    
    old_dirs = []
    for pattern in old_dirs_patterns:
        old_dirs.extend(glob.glob(pattern))
    
    if old_dirs:
        print(f"Found {len(old_dirs)} old-style example directories in root. Cleaning up...")
        for old_dir in old_dirs:
            try:
                shutil.rmtree(old_dir)
                print(f"  Removed {old_dir}")
            except Exception as e:
                print(f"  Failed to remove {old_dir}: {e}")

def find_latest_model():
    """
    Find the path to the latest trained model.
    """
    if not os.path.exists("runs/train"):
        return None, None
        
    best_params_dirs = [d for d in os.listdir("runs/train") if d.startswith("yolo_best_params")]
    
    if not best_params_dirs:
        return None, None
    
    best_params_dirs = sorted(best_params_dirs, key=lambda x: int(x.replace("yolo_best_params", "")))
    
    latest_dir = best_params_dirs[-1]
    model_path = f"runs/train/{latest_dir}/weights/best.pt"
    
    if os.path.exists(model_path):
        return model_path, latest_dir
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Run the complete MLflow pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip data preprocessing step")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip-serving", action="store_true", help="Skip model serving step")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if CUDA is available")
    parser.add_argument("--keep-examples", type=int, default=10, help="Number of example directories to keep (default: 10)")
    args = parser.parse_args()
    
    # Ensure required directories exist
    ensure_directories()
    
    # Clean up old example directories
    cleanup_old_examples(max_dirs=args.keep_examples)
    
    # Start MLflow server
    server_process = setup_mlflow_server()
    
    try:
        # Step 1: Data Preprocessing
        if not args.skip_preprocessing:
            # Prepare or check orig_dataset directory for preprocessing
            if prepare_orig_dataset():
                run_command(
                    "python preprocessing.py --src orig_dataset --dst datasets --noise 0.0 --",
                    "Step 1: Data Preprocessing (Noise Filtering and Augmentation)",
                    run_name=True
                )
            else:
                print("Skipping preprocessing step because no dataset was found.")
                args.skip_preprocessing = True
        
        # Step 2: Hyperparameter Tuning
        cpu_flag = "--cpu" if args.cpu else ""
        if not args.skip_tuning:
            run_command(
                f"python hparam_tune.py --monitor {cpu_flag}",
                "Step 2: Hyperparameter Tuning & Model Training"
            )
        
        # Find the latest model
        model_path, model_dir = find_latest_model()
        model_exists = model_path is not None
        
        if model_exists:
            print(f"Found latest model at {model_path} in directory {model_dir}")
        else:
            print("Warning: No trained model found in runs/train directory")
        
        # Step 3: Model Evaluation on Test Data
        if not args.skip_evaluation:
            if model_exists:
                # Regular evaluation on test data
                run_command(
                    f"python predict_and_compare.py --data data.yaml --output artifacts/prediction_analysis --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                    "Step 3a: Model Evaluation on Original Test Data",
                    run_name=True
                )
                
                # Evaluation on augmented test data
                run_command(
                    f"python predict_and_compare.py --data data_aug.yaml --output artifacts/prediction_analysis --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                    "Step 3b: Model Evaluation on Augmented Test Data",
                    run_name=True
                )
                
                # Compare performance on original vs augmented test data
                run_command(
                    f"python evaluate_augmented_data.py --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                    "Step 3c: Performance Comparison on Original vs Augmented Test Data"
                )
            else:
                print("\n" + "="*80)
                print("ERROR: Cannot run evaluation because no trained model was found.")
                print("You need to train a model first by running the hyperparameter tuning step.")
                print("Run the pipeline without --skip-tuning flag or run 'python hparam_tune.py --monitor' separately.")
                print("="*80 + "\n")
                
                if args.skip_tuning:
                    print("You specified --skip-tuning but are trying to evaluate a model.")
                    user_input = input("Would you like to run the hyperparameter tuning step now? (y/n): ")
                    if user_input.lower() == 'y':
                        run_command(
                            f"python hparam_tune.py --monitor {cpu_flag}",
                            "Step 2: Hyperparameter Tuning (Requested after evaluation error)"
                        )
                        # Check again if model exists
                        model_path, model_dir = find_latest_model()
                        model_exists = model_path is not None
                        
                        if model_exists:
                            print(f"Found latest model at {model_path}")
                            # Now run evaluation
                            run_command(
                                f"python predict_and_compare.py --data data.yaml --output artifacts/prediction_analysis --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                                "Step 3a: Model Evaluation on Original Test Data",
                                run_name=True
                            )
                            
                            run_command(
                                f"python predict_and_compare.py --data data_aug.yaml --output artifacts/prediction_analysis --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                                "Step 3b: Model Evaluation on Augmented Test Data",
                                run_name=True
                            )
                            
                            run_command(
                                f"python evaluate_augmented_data.py --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                                "Step 3c: Performance Comparison on Original vs Augmented Test Data"
                            )
                    else:
                        print("Skipping evaluation step because no model was found and tuning was declined.")
                else:
                    print("Skipping evaluation step because no model was found.")
        
        # Step 4: Register and Serve Model
        if not args.skip_serving:
            if model_exists:
                # Pass the model path and directory to serve_model.py
                run_command(
                    f"python serve_model.py --register --model-path \"{model_path}\" --model-dir \"{model_dir}\"",
                    "Step 4: Register Model with MLflow"
                )
                
                print("\n\nTo serve the model, run one of the following commands:")
                print("1. Using MLflow's built-in server:")
                print("   mlflow models serve -m models:/yolov8-object-detection/latest -p 5000")
                print("2. Using the FastAPI app:")
                print(f"   python serve_model.py --serve --port 8000 --model-path \"{model_path}\" --model-dir \"{model_dir}\"")
            else:
                print("Skipping model serving step because no model was found")
        
        print("\n\nPipeline completed successfully!")
        print("Access the MLflow UI at http://localhost:5000 to view experiments and models.")
        
        # Keep the server running until the user presses Ctrl+C
        print("\nPress Ctrl+C to stop the MLflow server and exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")
        server_process.terminate()
        print("MLflow server stopped. Exiting.")

if __name__ == "__main__":
    main() 