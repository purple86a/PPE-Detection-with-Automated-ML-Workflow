import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import glob
from PIL import Image
from pathlib import Path
import random
import matplotlib.gridspec as gridspec

def read_yaml(yaml_file):
    """Read YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data

def count_classes_in_labels(labels_dir, class_count=13):
    """Count the number of instances of each class in the labels directory."""
    counts = [0] * class_count
    
    print(f"Looking for labels in: {labels_dir}")
    
    # Find all text files in the directory
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if 0 <= class_id < class_count:
                        counts[class_id] += 1
    
    return counts

def get_random_images(image_dir, num_images=4):
    """Get random images from the directory."""
    print(f"Looking for images in: {image_dir}")
    
    image_files = list(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
                  list(glob.glob(os.path.join(image_dir, '*.jpeg'))) + \
                  list(glob.glob(os.path.join(image_dir, '*.png')))
    
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        return []
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    return selected_images

def plot_sample_images(datasets, base_dir, save_path):
    """Plot sample images from each dataset and save the figure."""
    # Create figure for sample images
    fig = plt.figure(figsize=(15, 5))
    
    # Create a grid for images
    num_datasets = len(datasets)
    
    # Display random images from each set
    for i, dataset in enumerate(datasets):
        # Path should be datasets/images/train, datasets/images/val, etc.
        img_dir = os.path.join(base_dir, 'images', dataset)
        random_img = get_random_images(img_dir, 1)
        
        if random_img:
            ax = plt.subplot(1, num_datasets, i+1)
            img = Image.open(random_img[0])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{dataset.upper()} Set', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample images saved to {save_path}")

def plot_class_distribution(dataset, class_counts, class_names, save_path):
    """Plot class distribution for a single dataset and save the figure."""
    fig = plt.figure(figsize=(12, 6))
    
    # Sort classes by count for better visualization
    counts = class_counts[dataset]
    sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
    
    # Create sorted lists
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_names = [f"{i}: {class_names[i]}" for i in sorted_indices]
    
    # Plot bar chart
    bars = plt.bar(range(len(sorted_counts)), sorted_counts, color='skyblue')
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{sorted_counts[i]}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{dataset.upper()} Set - Class Distribution', fontsize=14)
    plt.xticks(range(len(sorted_counts)), sorted_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class distribution for {dataset} saved to {save_path}")

def main():
    # Create graphs directory if it doesn't exist
    os.makedirs('graphs', exist_ok=True)
    
    # Read data configuration
    data_config = read_yaml('data.yaml')
    class_names = data_config['names']
    
    # Get base directory - use the datasets directory as the base
    base_dir = Path('datasets')
    
    # Define datasets
    datasets = ['train', 'val', 'test']
    
    # Plot sample images
    plot_sample_images(datasets, base_dir, 'graphs/sample_images.png')
    
    # Count classes in each dataset
    class_counts = {}
    for dataset in datasets:
        # Path should be datasets/labels/train, datasets/labels/val, etc.
        labels_dir = os.path.join(base_dir, 'labels', dataset)
        class_counts[dataset] = count_classes_in_labels(labels_dir, len(class_names))
    
    # Plot individual class distributions for each dataset
    for dataset in datasets:
        plot_class_distribution(dataset, class_counts, class_names, f'graphs/{dataset}_class_distribution.png')
    
    # Create a combined distribution plot for comparison
    fig = plt.figure(figsize=(15, 8))
    x = np.arange(len(class_names))
    width = 0.25
    
    # Sort classes by total count across all datasets
    total_counts = [sum(class_counts[dataset][i] for dataset in datasets) for i in range(len(class_names))]
    sorted_indices = np.argsort(total_counts)[::-1]  # Sort in descending order
    
    # Reorder data according to sorted indices
    sorted_names = [f"{i}: {class_names[i]}" for i in sorted_indices]
    sorted_counts = {}
    for dataset in datasets:
        sorted_counts[dataset] = [class_counts[dataset][i] for i in sorted_indices]
    
    # Plot bars
    bar1 = plt.bar(x - width, sorted_counts['train'], width, label='Train')
    bar2 = plt.bar(x, sorted_counts['val'], width, label='Validation')
    bar3 = plt.bar(x + width, sorted_counts['test'], width, label='Test')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution Comparison Across Datasets', fontsize=14)
    plt.xticks(x, sorted_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/combined_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined class distribution visualization saved to graphs/combined_class_distribution.png")

if __name__ == "__main__":
    main() 