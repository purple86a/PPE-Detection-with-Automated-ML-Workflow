import os
import sys
import argparse
import mlflow
import torch
import yaml
import json
import time
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def extract_metrics_from_results(results, prefix="metrics/"):
    """
    Extract metrics directly from YOLO results object
    """
    metrics = {}

    metrics["precision"] = float(results.results_dict[f"{prefix}precision(B)"])
    metrics["recall"] = float(results.results_dict[f"{prefix}recall(B)"])
    metrics["mAP50"] = float(results.results_dict[f"{prefix}mAP50(B)"])
    metrics["mAP50-95"] = float(results.results_dict[f"{prefix}mAP50-95(B)"])
    
    print(f"Extracted metrics: {metrics}")
    return metrics

def plot_metrics_comparison(metrics_original, metrics_augmented, save_path):
    """
    Create a bar chart comparing metrics between original and augmented test sets
    """
    metrics_to_plot = ["precision", "recall", "mAP50", "mAP50-95"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(metrics_to_plot))
    
    original_bars = ax.bar(index - bar_width/2, 
                          [metrics_original[m] for m in metrics_to_plot], 
                          bar_width, 
                          label='Original Test Data')
    
    augmented_bars = ax.bar(index + bar_width/2, 
                           [metrics_augmented[m] for m in metrics_to_plot], 
                           bar_width, 
                           label='Augmented Test Data')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison: Original vs Augmented Test Data')
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    add_labels(original_bars)
    add_labels(augmented_bars)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on augmented test data")
    parser.add_argument("--model", type=str, default=None, help="Path to the model weights file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model weights file (alternative to --model)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing the model (used for naming)")
    parser.add_argument("--data", type=str, default="data_aug.yaml", help="Path to the data YAML file")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--device", type=str, default="0", help="Device to use (0, cpu, etc.)")
    parser.add_argument("--metrics-dir", type=str, default="artifacts/test_metrics", help="Directory for test metrics")
    parser.add_argument("--predictions-dir", type=str, default="artifacts/prediction_analysis", help="Directory for prediction analysis")
    parser.add_argument("--output", type=str, default=None, help="Legacy output directory (use --metrics-dir instead)")
    args = parser.parse_args()
    
    if args.model_path is not None:
        args.model = args.model_path
    
    if args.output is not None:
        print(f"Warning: --output is deprecated, use --metrics-dir instead. Using '{args.output}' as metrics directory.")
        args.metrics_dir = args.output
    
    timestamp = int(time.time())
    
    metrics_dir_original = os.path.join(args.metrics_dir, f"plain_run_{timestamp}")
    metrics_dir_augmented = os.path.join(args.metrics_dir, f"augmented_run_{timestamp}")
    
    os.makedirs(metrics_dir_original, exist_ok=True)
    os.makedirs(metrics_dir_augmented, exist_ok=True)
    
    print(f"Saving test metrics to:\n  Original: {metrics_dir_original}\n  Augmented: {metrics_dir_augmented}")
    
    # Find the latest model if not specified
    if args.model is None:
        model_path, model_dir = find_latest_model()
        if model_path:
            args.model = model_path
            args.model_dir = model_dir
            print(f"Using latest model: {args.model} from directory {args.model_dir}")
        else:
            print("No model found. Please specify a model path with --model")
            return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        return
    
    # Get run number from model path or model directory
    if args.model_dir and "yolo_best_params" in args.model_dir:
        try:
            run_number = int(args.model_dir.split("yolo_best_params")[1])
        except (ValueError, IndexError):
            run_number = int(time.time())
    elif "yolo_best_params" in args.model:
        try:
            run_number = int(args.model.split("yolo_best_params")[1].split("/")[0])
        except (ValueError, IndexError):
            run_number = int(time.time())
    else:
        run_number = int(time.time())
    
    # Set up MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "yolov8-augmented-test-evaluation"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            # If experiment is deleted, create a new one with a timestamp
            new_experiment_name = f"{experiment_name}-{int(time.time())}"
            print(f"Experiment '{experiment_name}' is deleted. Creating new experiment '{new_experiment_name}'")
            mlflow.create_experiment(new_experiment_name)
            experiment_name = new_experiment_name
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        # Fall back to default experiment
        print("Using default experiment")
    
    # Load the model
    model = YOLO(args.model)
    
    # Create temporary YAML files with correct paths
    temp_data_yaml = "temp_data.yaml"
    temp_data_aug_yaml = "temp_data_aug.yaml"
    
    # Create temp data.yaml for original test data
    with open("data.yaml", "r") as f:
        data_yaml_content = yaml.safe_load(f)
    
    # Fix paths if needed
    data_yaml_content["train"] = data_yaml_content["train"].replace("datasets/", "")
    data_yaml_content["val"] = data_yaml_content["val"].replace("datasets/", "")
    data_yaml_content["test"] = data_yaml_content["test"].replace("datasets/", "")
    
    # Save temporary data.yaml
    with open(temp_data_yaml, "w") as f:
        yaml.dump(data_yaml_content, f)
    
    # Create temp data_aug.yaml for augmented test data
    with open("data_aug.yaml", "r") as f:
        data_aug_yaml_content = yaml.safe_load(f)
    
    # Fix paths if needed
    data_aug_yaml_content["train"] = data_aug_yaml_content["train"].replace("datasets/", "")
    data_aug_yaml_content["val"] = data_aug_yaml_content["val"].replace("datasets/", "")
    data_aug_yaml_content["test"] = data_aug_yaml_content["test"].replace("datasets/", "")
    
    # Save temporary data_aug.yaml
    with open(temp_data_aug_yaml, "w") as f:
        yaml.dump(data_aug_yaml_content, f)
    
    # First evaluate on original test data
    print("\nEvaluating on original test data...")
    with mlflow.start_run(run_name=f"test_original_data_run_{run_number}"):
        # Log model path
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("data_yaml", temp_data_yaml)
        
        # Run validation on original test data
        test_results_original = model.val(
            data=temp_data_yaml,
            split="test",
            batch=args.batch,
            workers=args.workers,
            device=args.device
        )
        
        # Extract metrics
        test_metrics_original = extract_metrics_from_results(test_results_original)
        
        # Log metrics to MLflow
        if test_metrics_original:
            mlflow.log_metrics(test_metrics_original)
        
        # Save metrics to file
        metrics_file_original = f"{metrics_dir_original}/metrics_original_run_{run_number}.txt"
        with open(metrics_file_original, "w") as f:
            for key, value in sorted(test_metrics_original.items()):
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"Original test metrics saved to {metrics_file_original}")
        mlflow.log_artifact(metrics_file_original)
    
    # Then evaluate on augmented test data
    print("\nEvaluating on augmented test data...")
    with mlflow.start_run(run_name=f"test_augmented_data_run_{run_number}"):
        # Log model path
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("data_yaml", temp_data_aug_yaml)
        
        # Run validation on augmented test data
        test_results_augmented = model.val(
            data=temp_data_aug_yaml,
            split="test",  # data_aug.yaml points to test directory which contains augmented test images
            batch=args.batch,
            workers=args.workers,
            device=args.device
        )
        
        # Extract metrics
        test_metrics_augmented = extract_metrics_from_results(test_results_augmented)
        
        # Log metrics to MLflow
        if test_metrics_augmented:
            mlflow.log_metrics(test_metrics_augmented)
        
        # Save metrics to file
        metrics_file_augmented = f"{metrics_dir_augmented}/metrics_augmented_run_{run_number}.txt"
        with open(metrics_file_augmented, "w") as f:
            for key, value in sorted(test_metrics_augmented.items()):
                f.write(f"{key}: {value:.6f}\n")
        
        print(f"Augmented test metrics saved to {metrics_file_augmented}")
        mlflow.log_artifact(metrics_file_augmented)
    
    # Clean up temporary files
    os.remove(temp_data_yaml)
    os.remove(temp_data_aug_yaml)
    
    # Create comparison visualization
    plot_metrics_comparison(
        test_metrics_original, 
        test_metrics_augmented, 
        f"{metrics_dir_original}/metrics_comparison_run_{run_number}.png"
    )
    
    # Save comparison to JSON for easy access
    comparison = {
        "original": test_metrics_original,
        "augmented": test_metrics_augmented,
        "run_number": run_number,
        "model_path": args.model,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{metrics_dir_original}/comparison_run_{run_number}.json", "w") as f:
        json.dump(comparison, f, indent=4)
    
    # Print comparison summary
    print("\nMetrics Comparison:")
    print(f"{'Metric':<10} {'Original':<10} {'Augmented':<10} {'Difference':<10}")
    print("-" * 40)
    for metric in ["precision", "recall", "mAP50", "mAP50-95"]:
        orig = test_metrics_original[metric]
        aug = test_metrics_augmented[metric]
        diff = aug - orig
        diff_percent = (diff / orig) * 100 if orig != 0 else 0
        print(f"{metric:<10} {orig:.4f}      {aug:.4f}      {diff:.4f} ({diff_percent:+.1f}%)")

if __name__ == "__main__":
    main() 