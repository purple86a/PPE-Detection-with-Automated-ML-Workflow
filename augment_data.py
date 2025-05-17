import os
import shutil
import random
import yaml
import cv2
import numpy as np
import mlflow
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
Data Preprocessing and Augmentation Script

This script processes a dataset by:
1. Applying noise filtering (median filter) to the training images only
2. Keeping validation and test sets unchanged
3. Applying augmentations to the training set
4. Saving the processed data to a destination directory with proper structure
5. Updating data.yaml and data_aug.yaml to point to the processed data

Usage:
    python augment_data.py --src orig_dataset --dst datasets --level medium

Parameters:
    --src: Source directory containing the original dataset (default: orig_dataset)
    --dst: Destination directory to save the processed data (default: datasets)
    --level: Augmentation level - light, medium, or heavy (default: medium)
    --noise: Enable noise filtering (default: True)
    --num-aug: Number of augmentations per image (default: 2)
"""

def setup_mlflow(run_name=None):
    """Set up MLflow tracking."""
    # Use environment variable for tracking URI if available
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "data-augmentation"
    
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
    
    return mlflow.start_run(run_name=run_name)

def read_yaml(yaml_file):
    """Read YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(data, yaml_file):
    """Save data to YAML file."""
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def create_directory_structure(base_dir):
    """Create directory structure for augmented data."""
    # Create main directories
    os.makedirs(os.path.join(base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'test'), exist_ok=True)
    
    # Create label directories
    os.makedirs(os.path.join(base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'test'), exist_ok=True)
    
    return base_dir

def apply_noise_filtering(image, filter_type="median", kernel_size=5):
    """Apply noise filtering to an image.
    
    Args:
        image: The input image.
        filter_type: Type of filter to apply ('median', 'gaussian', 'bilateral').
        kernel_size: Size of the filter kernel.
        
    Returns:
        The filtered image.
    """
    if filter_type == "median":
        # Median filtering is effective at removing salt-and-pepper noise
        # while preserving edges
        return cv2.medianBlur(image, kernel_size)
    
    elif filter_type == "gaussian":
        # Gaussian filtering is good for removing Gaussian noise
        # but can blur edges
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif filter_type == "bilateral":
        # Bilateral filtering preserves edges while removing noise
        # More computationally expensive
        d = kernel_size * 2
        return cv2.bilateralFilter(image, d, 75, 75)
    
    else:
        # Default: no filtering
        return image

def apply_augmentations(image, bboxes, class_ids, augmentation_level="light"):
    """Apply augmentations to image and bounding boxes."""
    # Define augmentation pipeline based on level
    if augmentation_level == "light":
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(p=0.3),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
    
    elif augmentation_level == "medium":
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.7),
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
    
    elif augmentation_level == "heavy":
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.7),
            A.RGBShift(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.7),
            A.RandomShadow(p=0.5),
            A.RandomFog(p=0.3),
            A.RandomRain(p=0.2),
            A.RandomSnow(p=0.2),
            A.CLAHE(p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
    
    else:
        # Default: no augmentation
        return image, bboxes, class_ids
    
    # Apply transformation
    transformed = transform(image=image, bboxes=bboxes, class_ids=class_ids)
    
    return transformed['image'], transformed['bboxes'], transformed['class_ids']

def augment_dataset(src_dir, dst_dir, augmentation_level="medium", noise_filter=True, num_augmentations=2):
    """Augment dataset with various transformations and apply noise filtering to training images."""
    # Check if source and destination are the same
    same_dir = os.path.abspath(src_dir) == os.path.abspath(dst_dir)
    if same_dir:
        print("Source and destination directories are the same. Skipping file copying and only performing augmentation.")
    
    # Get source directories
    src_images_dir = os.path.join(src_dir, 'images')
    src_labels_dir = os.path.join(src_dir, 'labels')
    
    # Get destination directories
    dst_images_dir = os.path.join(dst_dir, 'images')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    
    # Create destination directories if they don't exist
    create_directory_structure(dst_dir)
    
    # Get class names from data.yaml
    data_config = read_yaml(os.path.join(src_dir, 'data.yaml'))
    class_names = data_config['names']
    
    # Track augmentation statistics
    stats = {
        'original_images': 0,
        'augmented_images': 0,
        'noise_filtered_images': 0,
        'train': {'original': 0, 'augmented': 0},
        'val': {'original': 0, 'augmented': 0},
        'test': {'original': 0, 'augmented': 0},
        'class_distribution': {cls: {'original': 0, 'augmented': 0} for cls in class_names}
    }
    
    # Create timestamped directory for this run's examples
    timestamp = int(time.time())
    artifacts_dir = 'artifacts'
    example_type_dir = os.path.join(artifacts_dir, 'augmentation')
    example_dir = os.path.join(example_type_dir, f'examples_{timestamp}')
    
    # Create parent directories if they don't exist
    os.makedirs(example_type_dir, exist_ok=True)
    os.makedirs(example_dir, exist_ok=True)
    print(f"Saving examples to {example_dir}")
    
    # Counters to limit examples
    filter_example_count = 0
    augmentation_example_count = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Get source and destination paths
        src_split_images = os.path.join(src_images_dir, split)
        src_split_labels = os.path.join(src_labels_dir, split)
        
        dst_split_images = os.path.join(dst_images_dir, split)
        dst_split_labels = os.path.join(dst_labels_dir, split)
        
        # Check if source directories exist
        if not os.path.exists(src_split_images) or not os.path.exists(src_split_labels):
            print(f"Source directories for {split} not found. Skipping...")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(src_split_images) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Found {len(image_files)} images in {split} split")
        stats['original_images'] += len(image_files)
        stats[split]['original'] = len(image_files)
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            # Get image and label paths
            img_path = os.path.join(src_split_images, img_file)
            label_path = os.path.join(src_split_labels, os.path.splitext(img_file)[0] + '.txt')
            
            # Skip if label file doesn't exist
            if not os.path.exists(label_path):
                continue
            
            # Skip copying if source and destination are the same
            if same_dir:
                # Just count the image
                stats['original_images'] += 1
                stats[split]['original'] += 1
                
                # Count classes
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Try to parse as int first
                                class_id = int(parts[0])
                            except ValueError:
                                # If it's a float (like '0.0'), convert to float first, then int
                                class_id = int(float(parts[0]))
                            
                            if class_id < len(class_names):
                                stats['class_distribution'][class_names[class_id]]['original'] += 1
                continue
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply noise filtering only to training images if enabled
            if split == 'train' and noise_filter:
                # Apply median filtering to remove noise
                filtered_image = apply_noise_filtering(image, filter_type="median", kernel_size=5)
                stats['noise_filtered_images'] += 1
                
                # Save filtered image
                dst_img_path = os.path.join(dst_split_images, img_file)
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                cv2.imwrite(dst_img_path, cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))
                
                # Use filtered image for further processing
                image = filtered_image
                
                # Save example of filtering
                if filter_example_count < 2:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Draw original image
                    axes[0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    # Draw filtered image
                    axes[1].imshow(filtered_image)
                    axes[1].set_title('Noise Filtered (Median)')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'{example_dir}/filter_{split}_{os.path.splitext(img_file)[0]}.png')
                    plt.close()
                    filter_example_count += 1
            else:
                # For validation and test sets, just copy the original image without modification
                dst_img_path = os.path.join(dst_split_images, img_file)
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                shutil.copy(img_path, dst_img_path)
            
            # Copy label file
            dst_label_path = os.path.join(dst_split_labels, os.path.splitext(img_file)[0] + '.txt')
            os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
            shutil.copy(label_path, dst_label_path)
            
            # Read labels for statistics
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            bboxes = []
            class_ids = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # Try to parse as int first
                        class_id = int(parts[0])
                    except ValueError:
                        # If it's a float (like '0.0'), convert to float first, then int
                        class_id = int(float(parts[0]))
                    
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Add to statistics
                    if class_id < len(class_names):
                        stats['class_distribution'][class_names[class_id]]['original'] += 1
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_ids.append(class_id)
            
            # Apply augmentations only to training set
            if split == 'train':
                # Create augmented versions
                for i in range(num_augmentations):
                    # Apply augmentations
                    aug_image, aug_bboxes, aug_class_ids = apply_augmentations(
                        image, bboxes, class_ids, augmentation_level
                    )
                    
                    # Generate augmented file names
                    aug_img_file = f"{os.path.splitext(img_file)[0]}_aug{i+1}{os.path.splitext(img_file)[1]}"
                    aug_img_path = os.path.join(dst_split_images, aug_img_file)
                    aug_label_path = os.path.join(dst_split_labels, f"{os.path.splitext(img_file)[0]}_aug{i+1}.txt")
                    
                    # Save augmented image
                    cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    # Save augmented labels
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(aug_bboxes, aug_class_ids):
                            # Convert class_id to int to ensure it's not a float
                            class_id_int = int(class_id)
                            f.write(f"{class_id_int} {' '.join(map(str, bbox))}\n")
                            
                            # Add to statistics
                            if class_id_int < len(class_names):
                                stats['class_distribution'][class_names[class_id_int]]['augmented'] += 1
                    
                    stats['augmented_images'] += 1
                    stats[split]['augmented'] += 1
                    
                    # Save example of augmentation
                    if augmentation_example_count < 2:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        
                        # Draw original/filtered image with bboxes
                        axes[0].imshow(image)
                        for bbox, class_id in zip(bboxes, class_ids):
                            x, y, w, h = bbox
                            x1 = int((x - w/2) * image.shape[1])
                            y1 = int((y - h/2) * image.shape[0])
                            x2 = int((x + w/2) * image.shape[1])
                            y2 = int((y + h/2) * image.shape[0])
                            
                            axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                          fill=False, edgecolor='green', linewidth=2))
                            if int(class_id) < len(class_names):
                                axes[0].text(x1, y1-5, class_names[int(class_id)], 
                                            color='white', bbox=dict(facecolor='green', alpha=0.7))
                        
                        title = 'Filtered' if noise_filter else 'Original'
                        axes[0].set_title(title)
                        axes[0].axis('off')
                        
                        # Draw augmented image with bboxes
                        axes[1].imshow(aug_image)
                        for bbox, class_id in zip(aug_bboxes, aug_class_ids):
                            x, y, w, h = bbox
                            x1 = int((x - w/2) * aug_image.shape[1])
                            y1 = int((y - h/2) * aug_image.shape[0])
                            x2 = int((x + w/2) * aug_image.shape[1])
                            y2 = int((y + h/2) * aug_image.shape[0])
                            
                            axes[1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                          fill=False, edgecolor='blue', linewidth=2))
                            if int(class_id) < len(class_names):
                                axes[1].text(x1, y1-5, class_names[int(class_id)], 
                                            color='white', bbox=dict(facecolor='blue', alpha=0.7))
                        
                        axes[1].set_title(f'Augmented ({augmentation_level})')
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(f'{example_dir}/aug_{split}_{os.path.splitext(img_file)[0]}_{i+1}.png')
                        plt.close()
                        augmentation_example_count += 1
    
    # Create augmented data.yaml
    aug_data_config = data_config.copy()
    
    # Use absolute paths for compatibility with different tools
    dst_dir_abs = os.path.abspath(dst_dir)
    aug_data_config['train'] = os.path.join(dst_dir_abs, 'images', 'train')
    aug_data_config['val'] = os.path.join(dst_dir_abs, 'images', 'val')
    aug_data_config['test'] = os.path.join(dst_dir_abs, 'images', 'test')
    
    # Save to data_aug.yaml in the root directory
    save_yaml(aug_data_config, 'data_aug.yaml')
    
    # Also save to datasets/data.yaml for compatibility
    save_yaml(aug_data_config, os.path.join(dst_dir, 'data.yaml'))
    
    # Update the main data.yaml in the project root to point to the new data
    # This will ensure all tools use the processed data
    if os.path.exists('data.yaml'):
        main_data_config = read_yaml('data.yaml')
        main_data_config['train'] = aug_data_config['train']
        main_data_config['val'] = aug_data_config['val']
        main_data_config['test'] = aug_data_config['test']
        save_yaml(main_data_config, 'data.yaml')
        print(f"Updated main data.yaml to use processed data in {dst_dir_abs}")
    
    # Create visualization of class distribution
    plot_class_distribution(stats, os.path.join(example_dir, 'class_distribution.png'))
    
    return stats

def plot_class_distribution(stats, save_path):
    """Plot class distribution before and after augmentation."""
    class_names = list(stats['class_distribution'].keys())
    original_counts = [stats['class_distribution'][cls]['original'] for cls in class_names]
    augmented_counts = [stats['class_distribution'][cls]['augmented'] for cls in class_names]
    
    # Sort by original count
    sorted_indices = np.argsort(original_counts)[::-1]
    class_names = [class_names[i] for i in sorted_indices]
    original_counts = [original_counts[i] for i in sorted_indices]
    augmented_counts = [augmented_counts[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(class_names))
    
    # Plot bars
    bar1 = ax.bar(index - bar_width/2, original_counts, bar_width, label='Original')
    bar2 = ax.bar(index + bar_width/2, augmented_counts, bar_width, label='Augmented')
    
    # Add labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution Before and After Augmentation')
    ax.set_xticks(index)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    
    # Add count labels
    for i, v in enumerate(original_counts):
        ax.text(i - bar_width/2, v + 5, str(v), ha='center')
    
    for i, v in enumerate(augmented_counts):
        ax.text(i + bar_width/2, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Augment dataset with MLflow tracking")
    parser.add_argument("--src", type=str, default="orig_dataset", help="Source directory")
    parser.add_argument("--dst", type=str, default="datasets", help="Destination directory")
    parser.add_argument("--level", type=str, default="medium", choices=["light", "medium", "heavy"], help="Augmentation level")
    parser.add_argument("--noise", action="store_true", default=True, help="Apply noise filtering")
    parser.add_argument("--num-aug", type=int, default=2, help="Number of augmentations per image")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the MLflow run")
    args = parser.parse_args()
    
    # Create a meaningful run name
    run_name = args.run_name if args.run_name else f"Augmentation-{args.level}-filter_{args.noise}-aug_{args.num_aug}"
    
    # Start MLflow run
    with setup_mlflow(run_name=run_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "augmentation_level": args.level,
            "noise_filtering": args.noise,
            "num_augmentations": args.num_aug,
            "source_directory": args.src,
            "destination_directory": args.dst
        })
        
        # Perform augmentation
        print(f"Augmenting dataset from {args.src} to {args.dst}...")
        stats = augment_dataset(
            args.src, 
            args.dst, 
            augmentation_level=args.level,
            noise_filter=args.noise,
            num_augmentations=args.num_aug
        )
        
        # Log metrics
        mlflow.log_metrics({
            "original_images": stats['original_images'],
            "augmented_images": stats['augmented_images'],
            "noise_filtered_images": stats['noise_filtered_images'],
            "train_original": stats['train']['original'],
            "train_augmented": stats['train']['augmented'],
            "val_original": stats['val']['original'],
            "val_augmented": stats['val']['augmented'],
            "test_original": stats['test']['original'],
            "test_augmented": stats['test']['augmented']
        })
        
        # Log artifacts
        mlflow.log_artifacts(example_dir, "augmentation_examples")
        
        print(f"Augmentation complete. Stats:")
        print(f"  Original images: {stats['original_images']}")
        print(f"  Augmented images: {stats['augmented_images']}")
        print(f"  Noise filtered images: {stats['noise_filtered_images']}")
        print(f"  Train: {stats['train']['original']} original, {stats['train']['augmented']} augmented")
        print(f"  Val: {stats['val']['original']} original, {stats['val']['augmented']} augmented")
        print(f"  Test: {stats['test']['original']} original, {stats['test']['augmented']} augmented")
        
        print(f"Augmentation examples saved to '{example_dir}' directory")
        print(f"Processed data saved to '{args.dst}' directory")
        print(f"Data paths in data.yaml and data_aug.yaml have been updated to point to the processed data")
        print(f"You can now use this data for training with: --data data.yaml")

if __name__ == "__main__":
    main() 