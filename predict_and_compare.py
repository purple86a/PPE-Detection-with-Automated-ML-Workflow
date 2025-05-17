import os
import sys
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from collections import Counter, defaultdict
import cv2
import random
import mlflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
from tqdm import tqdm

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

def setup_mlflow(run_name=None):
    """
    Set up MLflow tracking.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "model-evaluation"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            import time
            new_experiment_name = f"{experiment_name}-{int(time.time())}"
            print(f"Experiment '{experiment_name}' is deleted. Creating new experiment '{new_experiment_name}'")
            mlflow.create_experiment(new_experiment_name)
            experiment_name = new_experiment_name
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        print("Using default experiment")
    
    return mlflow.start_run(run_name=run_name)

def read_yaml(yaml_file):
    """
    Read YAML file.
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_ground_truth(label_file, class_names):
    """
    Load ground truth labels from a YOLO format label file.
    """
    if not os.path.exists(label_file):
        if 'datasets/images' in label_file:
            alternative_path = label_file.replace('datasets/images', 'datasets/labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(alternative_path):
                label_file = alternative_path
            else:
                return []
        else:
            return []
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # class, x, y, w, h
            class_id = int(parts[0])
            if class_id < len(class_names):
                class_name = class_names[class_id]
                labels.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': 1.0,  # Ground truth has 100% confidence
                    'bbox': [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                })
    
    return labels

def get_prediction_vs_ground_truth_data(model, test_images, data_yaml, conf_threshold=0.25):
    """
    Get prediction vs ground truth data for visualization.
    """

    data_config = read_yaml(data_yaml)
    class_names = data_config['names']
    
    class_counts = {
        'true': Counter(),
        'pred': Counter()
    }
    
    y_true = []
    y_pred = []
    
    sample_images = []
    
    all_predictions = []
    all_ground_truths = []
    
    for img_path in tqdm(test_images, desc="Processing images"):
        if 'datasets/images' in img_path:
            label_file = img_path.replace('datasets/images', 'datasets/labels').rsplit('.', 1)[0] + '.txt'
        else:
            label_file = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        
        ground_truth = load_ground_truth(label_file, class_names)
        
        gt_vector = [0] * len(class_names)
        for gt in ground_truth:
            class_counts['true'][gt['class_name']] += 1
            y_true.append(gt['class_id'])
            
            gt_vector[gt['class_id']] = 1
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        pred_vector = [0] * len(class_names)
        
        predictions = []
        for i, det in enumerate(results[0].boxes):
            cls_id = int(det.cls.item())
            if cls_id < len(class_names):
                cls_name = class_names[cls_id]
                conf = det.conf.item()
                predictions.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': det.xywhn.tolist()[0]  # Normalized bbox
                })
                
                pred_vector[cls_id] = max(pred_vector[cls_id], conf)
                
                class_counts['pred'][cls_name] += 1
                y_pred.append(cls_id)
        
        if sum(gt_vector) > 0 or sum(pred_vector) > 0:
            all_ground_truths.append(gt_vector)
            all_predictions.append(pred_vector)
        
        if len(ground_truth) > 0 and len(predictions) > 0 and random.random() < 0.1:  # 10% chance
            sample_images.append({
                'image_path': img_path,
                'ground_truth': ground_truth,
                'predictions': predictions
            })
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    print(f"Collected {len(all_ground_truths)} ground truth vectors and {len(all_predictions)} prediction vectors for PR curves")
    
    return class_counts, y_true, y_pred, sample_images, class_names, all_predictions, all_ground_truths

def plot_class_distribution_comparison(class_counts, class_names, save_path):
    """
    Plot comparison of true vs predicted class distribution.
    """
    all_classes = sorted(set(list(class_counts['true'].keys()) + list(class_counts['pred'].keys())))
    
    true_counts = [class_counts['true'].get(cls, 0) for cls in all_classes]
    pred_counts = [class_counts['pred'].get(cls, 0) for cls in all_classes]
    
    sorted_indices = np.argsort(true_counts)[::-1]
    all_classes = [all_classes[i] for i in sorted_indices]
    true_counts = [true_counts[i] for i in sorted_indices]
    pred_counts = [pred_counts[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.4
    index = np.arange(len(all_classes))
    
    true_bars = ax.bar(index - bar_width/2, true_counts, bar_width, label='Ground Truth')
    pred_bars = ax.bar(index + bar_width/2, pred_counts, bar_width, label='Predictions')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Ground Truth vs Predicted Class Distribution')
    ax.set_xticks(index)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    
    def add_labels(bars, counts):
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}', ha='center', va='bottom')
    
    add_labels(true_bars, true_counts)
    add_labels(pred_bars, pred_counts)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution comparison saved to {save_path}")
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix for class predictions.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path)
    fig = plt.gcf()
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    return fig, cm, cm_norm

def visualize_sample_predictions(sample_images, class_names, output_dir):
    """
    Visualize sample predictions vs ground truth.
    """
    if not sample_images:
        print("No sample images to visualize")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = []
    
    for i, sample in enumerate(sample_images[:10]):  # Limit to 10 samples
        img_path = sample['image_path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        
        for gt in sample['ground_truth']:
            bbox = gt['bbox']
            x, y, width, height = bbox
            x1 = int((x - width/2) * w)
            y1 = int((y - height/2) * h)
            x2 = int((x + width/2) * w)
            y2 = int((y + height/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT: {gt['class_name']}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for pred in sample['predictions']:
            bbox = pred['bbox']
            x, y, width, height = bbox
            x1 = int((x - width/2) * w)
            y1 = int((y - height/2) * h)
            x2 = int((x + width/2) * w)
            y2 = int((y + height/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Pred: {pred['class_name']} ({pred['confidence']:.2f})", 
                       (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(f"Sample {i+1}: Ground Truth (Green) vs Predictions (Red)")
        plt.axis('off')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        
        visualization_paths.append(save_path)
    
    print(f"Sample visualizations saved to {output_dir}")
    return visualization_paths

def calculate_metrics(class_counts):
    """
    Calculate precision, recall and F1-score per class.
    """
    metrics = {}
    
    for class_name in set(list(class_counts['true'].keys()) + list(class_counts['pred'].keys())):
        tp = min(class_counts['true'].get(class_name, 0), class_counts['pred'].get(class_name, 0))
        fp = max(0, class_counts['pred'].get(class_name, 0) - tp)
        fn = max(0, class_counts['true'].get(class_name, 0) - tp)
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_count': class_counts['true'].get(class_name, 0),
            'pred_count': class_counts['pred'].get(class_name, 0)
        }
    
    return metrics

def plot_metrics_by_class(metrics, save_path):
    """
    Plot precision, recall and F1-score for each class.
    """
    sorted_classes = sorted(metrics.keys(), key=lambda x: metrics[x]['f1'], reverse=True)
    
    classes = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for class_name in sorted_classes:
        classes.append(class_name)
        precision_values.append(metrics[class_name]['precision'])
        recall_values.append(metrics[class_name]['recall'])
        f1_values.append(metrics[class_name]['f1'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.25
    index = np.arange(len(classes))
    
    ax.bar(index - bar_width, precision_values, bar_width, label='Precision')
    ax.bar(index, recall_values, bar_width, label='Recall')
    ax.bar(index + bar_width, f1_values, bar_width, label='F1-score')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall and F1-score by Class')
    ax.set_xticks(index)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics by class saved to {save_path}")
    
    return fig

def plot_pr_curves(all_predictions, all_ground_truths, class_names, save_dir):
    """
    Plot precision-recall curves for each class.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    y_true = np.array(all_ground_truths)
    y_scores = np.array(all_predictions)
    
    if len(y_true) == 0 or len(y_scores) == 0:
        print("No data for PR curves")
        return {}, {}
    
    if len(y_true) != len(y_scores):
        print(f"Warning: Inconsistent array sizes detected. y_true: {len(y_true)}, y_scores: {len(y_scores)}")
        min_size = min(len(y_true), len(y_scores))
        y_true = y_true[:min_size]
        y_scores = y_scores[:min_size]
        print(f"Arrays truncated to {min_size} samples for PR curve calculation")
    
    average_precisions = {}
    pr_curve_figs = {}
    
    for i, class_name in enumerate(class_names):
        if i < y_true.shape[1]:
            try:
                precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
                
                ap = average_precision_score(y_true[:, i], y_scores[:, i])
                average_precisions[class_name] = ap
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve: {class_name}')
                plt.legend(loc='best')
                plt.grid(True)
                
                save_path = os.path.join(save_dir, f"pr_curve_{class_name.replace(' ', '_')}.png")
                plt.savefig(save_path)
                pr_curve_figs[class_name] = plt.gcf()
                plt.close()
            except Exception as e:
                print(f"Error calculating PR curve for class {class_name}: {e}")
                continue
    
    if average_precisions:
        try:
            plt.figure(figsize=(8, 6))
            mean_ap = np.mean(list(average_precisions.values()))
            plt.title(f'Mean Precision-Recall Curve (mAP = {mean_ap:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            
            save_path = os.path.join(save_dir, "mean_pr_curve.png")
            plt.savefig(save_path)
            pr_curve_figs['mean'] = plt.gcf()
            plt.close()
            
            print(f"PR curves saved to {save_dir}")
        except Exception as e:
            print(f"Error calculating mean PR curve: {e}")
    else:
        print("No PR curves could be calculated")
    
    return average_precisions, pr_curve_figs

def main():
    parser = argparse.ArgumentParser(description="Predict with YOLOv8 model and compare with ground truth")
    parser.add_argument("--model", type=str, default=None, help="Path to the model weights file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model weights file (alternative to --model)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing the model (used for naming)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to the data YAML file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--output", type=str, default="artifacts/prediction_analysis", help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the MLflow run")
    args = parser.parse_args()
    
    # Use model-path if provided (for compatibility with run_mlflow_pipeline.py)
    if args.model_path is not None:
        args.model = args.model_path
    
    # Determine output subdirectory (original or augmented)
    data_type = "original" if args.data == "data.yaml" else "augmented"

    # Create a timestamped directory for this run
    import time
    timestamp = int(time.time())

    output_dir = os.path.join(args.output, f"{data_type}_run_{timestamp}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving prediction analysis to: {output_dir}")
    
    # Find the latest model if not specified
    if args.model is None:
        model_path, model_dir = find_latest_model()
        if model_path:
            args.model = model_path
            args.model_dir = model_dir
            print(f"Using latest model: {args.model}")
        else:
            print("No model found. Please specify a model path with --model")
            return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        return
    
    if args.run_name is None:
        if args.model_dir:
            model_name = args.model_dir
        else:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model)))
        data_type = "Original" if args.data == "data.yaml" else "Augmented"
        args.run_name = f"Evaluation-{data_type}-{model_name}-conf_{args.conf}"
    
    # Start MLflow run
    with setup_mlflow(run_name=args.run_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_path": args.model,
            "data_yaml": args.data,
            "confidence_threshold": args.conf,
            "output_directory": args.output
        })
        
        # Load the model
        print(f"Loading model from {args.model}")
        model = YOLO(args.model)
        
        # Log model to MLflow
        mlflow.log_artifact(args.model, "model")
        
        # Load data configuration
        data_config = read_yaml(args.data)
        
        # Get test directory from data config
        test_path = data_config.get('test', 'images/test')
        
        # Handle both absolute and relative paths
        if os.path.isabs(test_path):
            test_dir = test_path
        else:
            # If path starts with 'datasets/', use as is
            if test_path.startswith('datasets/'):
                test_dir = test_path
            # If path is just 'images/test' format, prepend with 'datasets/'
            elif test_path.startswith('images/'):
                test_dir = os.path.join('datasets', test_path)
            else:
                test_dir = test_path
                
        print(f"Using test directory: {test_dir}")
        
        # Get test images
        if os.path.exists(test_dir):
            test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            print(f"Warning: Test directory {test_dir} does not exist, trying fallback path")
            # Fallback to default test directory
            fallback_dir = os.path.join('datasets', 'images', 'test')
            if os.path.exists(fallback_dir):
                test_images = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
            else:
                test_images = []
        
        if not test_images:
            print(f"No test images found in {test_dir}")
            return
        
        print(f"Found {len(test_images)} test images")
        
        # Get prediction vs ground truth data
        print("Analyzing predictions vs ground truth...")
        class_counts, y_true, y_pred, sample_images, class_names, all_predictions, all_ground_truths = get_prediction_vs_ground_truth_data(
            model, test_images, args.data, args.conf
        )
        
        # Plot class distribution comparison
        class_dist_fig = plot_class_distribution_comparison(
            class_counts, 
            class_names,
            os.path.join(output_dir, "class_distribution_comparison.png")
        )
        mlflow.log_artifact(os.path.join(output_dir, "class_distribution_comparison.png"), "visualizations")
        
        # Plot confusion matrix
        conf_matrix_fig, cm, cm_norm = plot_confusion_matrix(
            y_true, 
            y_pred, 
            class_names,
            os.path.join(output_dir, "confusion_matrix.png")
        )
        mlflow.log_artifact(os.path.join(output_dir, "confusion_matrix.png"), "visualizations")
        
        # Save confusion matrix data
        np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
        np.save(os.path.join(output_dir, "confusion_matrix_normalized.npy"), cm_norm)
        mlflow.log_artifact(os.path.join(output_dir, "confusion_matrix.npy"), "data")
        mlflow.log_artifact(os.path.join(output_dir, "confusion_matrix_normalized.npy"), "data")
        
        # Visualize sample predictions
        visualization_paths = visualize_sample_predictions(
            sample_images,
            class_names,
            os.path.join(output_dir, "samples")
        )
        for path in visualization_paths:
            mlflow.log_artifact(path, "sample_predictions")
        
        # Calculate and plot metrics by class
        metrics = calculate_metrics(class_counts)
        metrics_fig = plot_metrics_by_class(
            metrics,
            os.path.join(output_dir, "metrics_by_class.png")
        )
        mlflow.log_artifact(os.path.join(output_dir, "metrics_by_class.png"), "visualizations")
        
        # Plot PR curves
        try:
            average_precisions, pr_curve_figs = plot_pr_curves(
                all_predictions,
                all_ground_truths,
                class_names,
                os.path.join(output_dir, "pr_curves")
            )
            
            # Log PR curves if they were generated
            if os.path.exists(os.path.join(output_dir, "pr_curves")):
                for curve_path in os.listdir(os.path.join(output_dir, "pr_curves")):
                    mlflow.log_artifact(
                        os.path.join(output_dir, "pr_curves", curve_path), 
                        "pr_curves"
                    )
        except Exception as e:
            print(f"Error generating PR curves: {e}")
            average_precisions = {}
        
        # Log metrics to MLflow
        overall_metrics = {
            "mean_precision": np.mean([m['precision'] for m in metrics.values()]),
            "mean_recall": np.mean([m['recall'] for m in metrics.values()]),
            "mean_f1": np.mean([m['f1'] for m in metrics.values()]),
            "mean_average_precision": np.mean(list(average_precisions.values()) if average_precisions else [0])
        }
        mlflow.log_metrics(overall_metrics)
        
        # Log per-class metrics
        for class_name, class_metrics in metrics.items():
            mlflow.log_metrics({
                f"{class_name}_precision": class_metrics['precision'],
                f"{class_name}_recall": class_metrics['recall'],
                f"{class_name}_f1": class_metrics['f1']
            })
        
        # Save metrics to JSON
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({
                "overall": overall_metrics,
                "per_class": metrics,
                "average_precision": average_precisions
            }, f, indent=4)
        mlflow.log_artifact(os.path.join(output_dir, "metrics.json"), "data")
        
        # Register model in MLflow
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "yolov8-object-detection"
        )
        
        print(f"All analysis results saved to {output_dir}")
        print(f"Results also logged to MLflow run: {run.info.run_id}")
        print(f"To view results in MLflow UI, run: mlflow ui --backend-store-uri sqlite:///mlruns.db")

if __name__ == "__main__":
    main() 