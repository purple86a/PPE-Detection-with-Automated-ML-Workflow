import os
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import torch
import multiprocessing
import sys
import time
import random
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Disable YOLO's built-in MLflow integration : Prevent double-logging or conflicting runs (you'll otherwise see duplicated metrics or "not tracking this run" warnings).
SETTINGS.update({'mlflow': False})

TUNNING_EPOCHS = 1
TRAINING_EPOCHS = 60
BATCH_SIZE = 8
WORKERS = 2
TUNNING_TRIALS = 12

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

def log_plots_as_artifacts(run_dir, mlflow_run=None):
    """
    Log all plots and visualizations from a training run as MLflow artifacts.
    """

    if not os.path.exists(run_dir):
        print(f"Warning: Run directory {run_dir} does not exist.")
        return
    
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    confusion_matrix_path = os.path.join(run_dir, "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        mlflow.log_artifact(confusion_matrix_path, "visualizations")
    
    pr_curve_path = os.path.join(run_dir, "PR_curve.png")
    if os.path.exists(pr_curve_path):
        mlflow.log_artifact(pr_curve_path, "visualizations")
    
    f1_curve_path = os.path.join(run_dir, "F1_curve.png")
    if os.path.exists(f1_curve_path):
        mlflow.log_artifact(f1_curve_path, "visualizations")
    
    p_curve_path = os.path.join(run_dir, "P_curve.png")
    if os.path.exists(p_curve_path):
        mlflow.log_artifact(p_curve_path, "visualizations")
    
    r_curve_path = os.path.join(run_dir, "R_curve.png")
    if os.path.exists(r_curve_path):
        mlflow.log_artifact(r_curve_path, "visualizations")
    
    results_path = os.path.join(run_dir, "results.png")
    if os.path.exists(results_path):
        mlflow.log_artifact(results_path, "visualizations")
    
    val_images_dir = os.path.join(run_dir, "val_batch0_pred.jpg")
    if os.path.exists(val_images_dir):
        mlflow.log_artifact(val_images_dir, "validation_images")
    
    # Log training results CSV
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        mlflow.log_artifact(results_csv, "data")
        
        try:
            import pandas as pd
            
            df = pd.read_csv(results_csv)
            
            # Plot training and validation losses
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Box Loss')
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(plots_dir, "box_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()
            mlflow.log_artifact(loss_plot_path, "visualizations")
            
            # Plot mAP metrics
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('mAP Metrics')
            plt.legend()
            plt.grid(True)
            map_plot_path = os.path.join(plots_dir, "map_metrics.png")
            plt.savefig(map_plot_path)
            plt.close()
            mlflow.log_artifact(map_plot_path, "visualizations")
            
            # Plot precision and recall
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Precision and Recall')
            plt.legend()
            plt.grid(True)
            pr_plot_path = os.path.join(plots_dir, "precision_recall.png")
            plt.savefig(pr_plot_path)
            plt.close()
            mlflow.log_artifact(pr_plot_path, "visualizations")
            
        except Exception as e:
            print(f"Error generating additional plots: {e}")
    

    hyp_path = os.path.join(run_dir, "args.yaml")
    if os.path.exists(hyp_path):
        mlflow.log_artifact(hyp_path, "config")
    
    print(f"Logged all visualizations from {run_dir} to MLflow")

def plot_loss(run_number):
    directory = f"runs/train/yolo_best_params{run_number}"
    csv_path = f"{directory}/results.csv"
    while not os.path.exists(csv_path):
        time.sleep(1)

    print(f"{csv_path} detected. Starting real-time plot...")
    
    fig, ax = plt.subplots()
    train_line, = ax.plot([], [], 'b-', label='Train Loss')
    val_line, = ax.plot([], [], 'r--', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Real-time Training & Validation Loss')
    ax.legend()

    # Update function
    def update(frame):
        try:
            data = pd.read_csv(csv_path)
            x = data['epoch']
            train_y = data['train/box_loss']
            val_y = data['val/box_loss']

            train_line.set_data(x, train_y)
            val_line.set_data(x, val_y)
            ax.relim()
            ax.autoscale_view()
        except Exception as e:
            print(f"Error reading file: {e}")
        return train_line, val_line

    # Start animation
    ani = FuncAnimation(fig, update, interval=1000)
    plt.tight_layout()
    plt.show()

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning
    """
    force_cpu = "--cpu" in sys.argv
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "yolov8-object-detection-training-tuning"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":

            new_experiment_name = f"{experiment_name}-{int(time.time())}"
            print(f"Experiment '{experiment_name}' is deleted. Creating new experiment '{new_experiment_name}'")
            mlflow.create_experiment(new_experiment_name)
            experiment_name = new_experiment_name
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")

        print("Using default experiment")
    
    with mlflow.start_run(nested=True):

        lr0 = trial.suggest_categorical("lr0", [0.001, 0.01])   
        weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.5])
            
        # Fixed parameters (not tuned)
        batch_size = BATCH_SIZE    # Batch size
        lrf = 0.01                 # Final learning rate factor
        momentum = 0.937           # SGD momentum/Adam beta1
        warmup_epochs = 3.0        # Warmup epochs
            
        epochs = TUNNING_EPOCHS    # Keep epochs reasonable for tuning
        imgsz = 640                # Reduced image size to save memory
            

        run_name = f"HPTuning-Trial{trial.number}-LR{lr0}-WD{weight_decay}-DO{dropout}"
            

        mlflow.log_params({
                "lr0": lr0,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "lrf": lrf,            
                "momentum": momentum,  
                "warmup_epochs": warmup_epochs, 
                "epochs": epochs,
                "imgsz": imgsz,
                "dropout": dropout,
                "model": "yolov8s"
            })
            

        model = YOLO("yolov8s.pt")
            

        try:
            device = "cpu" if force_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
            mlflow.log_param("device", device)
                
            results = model.train(
                    data="data.yaml",  
                    epochs=epochs,
                    imgsz=imgsz,
                    lr0=lr0,
                    lrf=lrf,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs,
                    batch=batch_size,
                    workers=WORKERS,
                    cache=False,
                    rect=True,
                    project="runs/train",
                    name=run_name,
                    device=device
                )
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error! Trying again with CPU...")
            results = model.train(
                    data="data.yaml",
                    epochs=epochs,
                    imgsz=imgsz,
                    lr0=lr0,
                    lrf=lrf,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    warmup_epochs=warmup_epochs,
                    batch=batch_size,
                    workers=WORKERS,
                    cache=False,
                    rect=True,
                    project="runs/train",
                    name=run_name,
                    device="cpu"
                )
            mlflow.log_param("device", "cpu (fallback)")
            
        metrics = extract_metrics_from_results(results)
            

        mlflow.log_metrics(metrics)
                
        weights_path = f"runs/train/{run_name}/weights/best.pt"
        if os.path.exists(weights_path):
            mlflow.log_artifact(weights_path)
                
        return metrics["precision"]
            

def main():
    # Print usage information
    if len(sys.argv) > 1 and "--help" in sys.argv:
        print("Usage: python hparam_tune.py [--cpu] [--new-db] [--deploy]")
        print("  --cpu: Force using CPU even if CUDA is available")
        print("  --new-db: Delete existing database and start fresh")
        print("  --deploy: Deploy the best model with MLflow using serve_model.py")
        print("  --monitor: Plot the loss during training")
        print("\nAfter running, use 'python analyze_hparam_results.py' with the correct study name")
        print("A unique study name will be generated each run")
        return
    
    # Check if database file exists and optionally delete it
    db_path = "optuna.sqlite3"
    if os.path.exists(db_path) and "--new-db" in sys.argv:
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")
    
    timestamp = int(time.time())
    random_id = random.randint(1000, 9999)
    study_name = f"yolov8-tune-{timestamp}-{random_id}"
    
    # Create MLflow callback Logs the trial's parameters and Records the resulting metric
    mlflow_callback = MLflowCallback(
        tracking_uri="sqlite:///mlruns.db",
        metric_name="precision"
    )
    
    # Create study Track All Trials - expirements 
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna.sqlite3",
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    print(f"Created study: {study_name}")
    
    # Optimize with MLflow tracking
    study.optimize(
        objective, 
        n_trials=TUNNING_TRIALS, 
        callbacks=[mlflow_callback]
    )
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    run_number = 1
    while os.path.exists(f"runs/train/yolo_best_params{run_number}"):
        run_number += 1
    best_params_dir = f"yolo_best_params{run_number}"
    print(f"\nTraining final model with best parameters in directory: {best_params_dir}")
    
    # Train a final model with the best parameters
    with mlflow.start_run(run_name=f"FinalModel-YOLOv8s-BestParams-LR{best_trial.params['lr0']}-WD{best_trial.params['weight_decay']}-DO{best_trial.params.get('dropout', 0.0)}"):
        model = YOLO("yolov8s.pt")
        
        force_cpu = "--cpu" in sys.argv
        device = "cpu" if force_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)
        
        try:
            if "--monitor" in sys.argv:
                import threading
                threading.Thread(target=plot_loss, args=(run_number,), daemon=True).start()
                
            final_results = model.train(
                data="data.yaml",
                epochs=TRAINING_EPOCHS,
                imgsz=640,
                **best_trial.params,
                workers=WORKERS,
                cache=False,
                rect=True,
                project="runs/train",
                name=best_params_dir,
                device=device
            )
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error in final training! Falling back to CPU...")
            final_results = model.train(
                data="data.yaml",
                epochs=TRAINING_EPOCHS,
                imgsz=640,
                **best_trial.params,
                workers=WORKERS,
                cache=False,
                rect=True,
                project="runs/train",
                name=best_params_dir,
                device="cpu"
            )
            mlflow.log_param("device", "cpu (fallback)")
        
        mlflow.log_params(best_trial.params)
        
        final_metrics = extract_metrics_from_results(final_results)
        if final_metrics:
            mlflow.log_metrics(final_metrics)
            
        log_plots_as_artifacts(f"runs/train/{best_params_dir}")
            
        mlflow.end_run()
        
        weights_path = f"runs/train/{best_params_dir}/weights/best.pt"
        if os.path.exists(weights_path):
            print(f"Final model weights saved to {weights_path}")
        else:
            print(f"Warning: Final model weights not found at {weights_path}")
            
    print("\nTo view results in MLflow UI, run:")
    print("mlflow ui --backend-store-uri sqlite:///mlruns.db")
    
    if "--deploy" in sys.argv:
        print("\nDeploying best model with MLflow...")
        try:
            from serve_model import register_model
            model_path = f"runs/train/{best_params_dir}/weights/best.pt"
            if os.path.exists(model_path):
                model_uri = register_model(model_path)
                print(f"Model deployed as: {model_uri}")
                print("To serve the model, run: python serve_model.py --serve --port 8000")
            else:
                print(f"Error: Model not found at {model_path}")
        except Exception as e:
            print(f"Error deploying model: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Duplicate stdout and stderr to both console and a log file#
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open("hparam_tune.log", "w")
    class TeeStream:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = TeeStream(orig_stdout, log_file)
    sys.stderr = TeeStream(orig_stderr, log_file)
    ##############################################################
    
    main() 