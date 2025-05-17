import os
import shutil
import argparse
import yaml
import cv2
import numpy as np
import mlflow
from tqdm import tqdm
from pathlib import Path
import time

# Import functions from augment_data.py
from augment_data import (
    setup_mlflow, read_yaml, save_yaml, create_directory_structure,
    apply_noise_filtering, apply_augmentations, plot_class_distribution
)

def preprocess_dataset(src_dir, dst_dir, noise_level=0.02, run_name=None):
    """
    Preprocess dataset:
    1. Apply noise filtering to train and validation sets
    2. Create augmented test set with heavy augmentation
    3. Keep original test set intact
    """
    
    if run_name is None:
        run_name = f"Preprocessing-NoiseFiltering-{noise_level}"
    
    with setup_mlflow(run_name=run_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        mlflow.log_params({
            "source_directory": src_dir,
            "destination_directory": dst_dir,
            "noise_level": noise_level
        })
        
        src_images_dir = os.path.join(src_dir, 'images')
        src_labels_dir = os.path.join(src_dir, 'labels')
        
        dst_images_dir = os.path.join(dst_dir, 'images')
        dst_labels_dir = os.path.join(dst_dir, 'labels')
        
        create_directory_structure(dst_dir)
        
        data_config = read_yaml('data.yaml')
        class_names = data_config['names']
        
        stats = {
            'original_images': 0,
            'noise_filtered_images': 0,
            'augmented_test_images': 0,
            'train': {'original': 0, 'processed': 0},
            'val': {'original': 0, 'processed': 0},
            'test': {'original': 0},
            'test_aug': {'augmented': 0},
            'class_distribution': {cls: {'original': 0, 'processed': 0} for cls in class_names}
        }
        
        timestamp = int(time.time())
        artifacts_dir = 'artifacts'
        example_type_dir = os.path.join(artifacts_dir, 'preprocessing')
        example_dir = os.path.join(example_type_dir, f'examples_{timestamp}')
        
        os.makedirs(example_type_dir, exist_ok=True)
        os.makedirs(example_dir, exist_ok=True)
        print(f"Saving examples to {example_dir}")
        
        filter_example_count = 0
        test_aug_example_count = 0
        
        for split in ['train', 'val']:
            print(f"\nProcessing {split} split with noise filtering...")
            
            src_split_images = os.path.join(src_images_dir, split)
            src_split_labels = os.path.join(src_labels_dir, split)
            
            dst_split_images = os.path.join(dst_images_dir, split)
            dst_split_labels = os.path.join(dst_labels_dir, split)
            
            if not os.path.exists(src_split_images) or not os.path.exists(src_split_labels):
                print(f"Source directories for {split} not found. Skipping...")
                continue
            
            image_files = [f for f in os.listdir(src_split_images) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Found {len(image_files)} images in {split} split")
            stats['original_images'] += len(image_files)
            stats[split]['original'] = len(image_files)
            
            for img_file in tqdm(image_files, desc=f"Noise filtering {split}"):
                img_path = os.path.join(src_split_images, img_file)
                label_path = os.path.join(src_split_labels, os.path.splitext(img_file)[0] + '.txt')
                
                if not os.path.exists(label_path):
                    continue
                
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                filtered_image = apply_noise_filtering(image, filter_type="median", kernel_size=5)
                
                dst_img_path = os.path.join(dst_split_images, img_file)
                dst_label_path = os.path.join(dst_split_labels, os.path.splitext(img_file)[0] + '.txt')
                
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
                
                cv2.imwrite(dst_img_path, cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))
                
                shutil.copy(label_path, dst_label_path)
                
                stats['noise_filtered_images'] += 1
                stats[split]['processed'] += 1
                
                if filter_example_count < 2:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(image)
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    axes[1].imshow(filtered_image)
                    axes[1].set_title('Filtered (Median)')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'{example_dir}/noise_filter_{split}_{os.path.splitext(img_file)[0]}.png')
                    plt.close()
                    
                    mlflow.log_artifact(f'{example_dir}/noise_filter_{split}_{os.path.splitext(img_file)[0]}.png', 
                                       "noise_filtering_examples")
                    filter_example_count += 1
        
        print("\nCopying original test set...")
        src_test_images = os.path.join(src_images_dir, 'test')
        src_test_labels = os.path.join(src_labels_dir, 'test')
        
        dst_test_images = os.path.join(dst_images_dir, 'test')
        dst_test_labels = os.path.join(dst_labels_dir, 'test')
        
        if os.path.exists(src_test_images) and os.path.exists(src_test_labels):
            test_image_files = [f for f in os.listdir(src_test_images) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Found {len(test_image_files)} images in test split")
            stats['test']['original'] = len(test_image_files)
            
            for img_file in tqdm(test_image_files, desc="Copying test set"):
                img_path = os.path.join(src_test_images, img_file)
                label_path = os.path.join(src_test_labels, os.path.splitext(img_file)[0] + '.txt')
                
                if not os.path.exists(label_path):
                    continue
                
                dst_img_path = os.path.join(dst_test_images, img_file)
                dst_label_path = os.path.join(dst_test_labels, os.path.splitext(img_file)[0] + '.txt')
                
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
                
                shutil.copy(img_path, dst_img_path)
                shutil.copy(label_path, dst_label_path)
        
        print("\nCreating augmented test set...")
        
        src_test_images = os.path.join(dst_images_dir, 'test')
        src_test_labels = os.path.join(dst_labels_dir, 'test')
        
        dst_test_aug_images = os.path.join(dst_images_dir, 'test_aug')
        dst_test_aug_labels = os.path.join(dst_labels_dir, 'test_aug')
        
        os.makedirs(dst_test_aug_images, exist_ok=True)
        os.makedirs(dst_test_aug_labels, exist_ok=True)
        
        if os.path.exists(src_test_images) and os.path.exists(src_test_labels):
            test_image_files = [f for f in os.listdir(src_test_images) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in tqdm(test_image_files, desc="Creating test_aug"):
                img_path = os.path.join(src_test_images, img_file)
                label_path = os.path.join(src_test_labels, os.path.splitext(img_file)[0] + '.txt')
                
                if not os.path.exists(label_path):
                    continue
                
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                bboxes = []
                class_ids = []
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            class_id = int(float(parts[0]))
                        
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_ids.append(class_id)
                
                aug_image, aug_bboxes, aug_class_ids = apply_augmentations(
                    image, bboxes, class_ids, "medium"
                )
                
                dst_img_path = os.path.join(dst_test_aug_images, img_file)
                dst_label_path = os.path.join(dst_test_aug_labels, os.path.splitext(img_file)[0] + '.txt')
                
                cv2.imwrite(dst_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                with open(dst_label_path, 'w') as f:
                    for bbox, class_id in zip(aug_bboxes, aug_class_ids):
                        class_id_int = int(class_id)
                        f.write(f"{class_id_int} {' '.join(map(str, bbox))}\n")
                
                stats['augmented_test_images'] += 1
                stats['test_aug']['augmented'] += 1
                
                if test_aug_example_count < 2:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    axes[0].imshow(image)
                    axes[0].set_title('Original Test Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(aug_image)
                    axes[1].set_title('Augmented Test Image')
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'{example_dir}/test_aug_{os.path.splitext(img_file)[0]}.png')
                    plt.close()
                    
                    mlflow.log_artifact(f'{example_dir}/test_aug_{os.path.splitext(img_file)[0]}.png', 
                                       "test_augmentation_examples")
                    test_aug_example_count += 1
        
        aug_data_config = data_config.copy()
        aug_data_config['train'] = 'datasets/images/train'
        aug_data_config['val'] = 'datasets/images/val'
        aug_data_config['test'] = 'datasets/images/test_aug'
        
        save_yaml(aug_data_config, 'data_aug.yaml')
        
        mlflow.log_metrics({
            "original_images": stats['original_images'],
            "noise_filtered_images": stats['noise_filtered_images'],
            "augmented_test_images": stats['augmented_test_images'],
            "train_original": stats['train']['original'],
            "train_processed": stats['train']['processed'],
            "val_original": stats['val']['original'],
            "val_processed": stats['val']['processed'],
            "test_original": stats['test']['original'],
            "test_aug_augmented": stats['test_aug']['augmented']
        })
        
        mlflow.log_artifacts(example_dir, "preprocessing_examples")
        
        print(f"\nPreprocessing complete. Stats:")
        print(f"  Original images: {stats['original_images']}")
        print(f"  Noise filtered images: {stats['noise_filtered_images']}")
        print(f"  Augmented test images: {stats['augmented_test_images']}")
        print(f"  Train: {stats['train']['original']} original, {stats['train']['processed']} noise filtered")
        print(f"  Val: {stats['val']['original']} original, {stats['val']['processed']} noise filtered")
        print(f"  Test: {stats['test']['original']} original")
        print(f"  Test_aug: {stats['test_aug']['augmented']} augmented")
        
        print(f"\nPreprocessing examples saved to {example_dir}")
        print(f"Processed data saved to '{dst_dir}' directory")
        print(f"data_aug.yaml created for augmented test data")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset with MLflow tracking")
    parser.add_argument("--src", type=str, default=".", help="Source directory")
    parser.add_argument("--dst", type=str, default="datasets", help="Destination directory")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level for filtering")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the MLflow run")
    args = parser.parse_args()
    
    # Create a meaningful run name if not provided
    run_name = args.run_name if args.run_name else f"Preprocessing-NoiseFiltering-{args.noise}"
    
    # Pass the run_name to preprocess_dataset
    preprocess_dataset(args.src, args.dst, args.noise, run_name=run_name)

if __name__ == "__main__":
    main() 