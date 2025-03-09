#!/usr/bin/env python3
"""
Script to analyze and visualize COCO detection results.
Run with:
    python3 analyze_results.py --metrics_file outputs/val_metrics_epoch_1.json
    or
    python3 analyze_results.py --predictions_file outputs/testdev_predictions_final.json --test_images coco_data/test2017
"""

import json
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# COCO class labels for visualization
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
    21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack',
    26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
    31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
    36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
    40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
    45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
    50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
    55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
    60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
    65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
    70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
    75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
    80: 'toothbrush'
}

def analyze_metrics(metrics_file):
    """Analyze and display metrics from a validation run"""
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return

    print("\n===== MODEL PERFORMANCE METRICS =====")
    print(f"AP (Average Precision): {metrics['AP']:.4f}")
    print(f"AP50 (AP at IoU=0.5): {metrics['AP50']:.4f}")
    print(f"AP75 (AP at IoU=0.75): {metrics['AP75']:.4f}")
    print(f"APs (AP for small objects): {metrics['APs']:.4f}")
    print(f"APm (AP for medium objects): {metrics['APm']:.4f}")
    print(f"APl (AP for large objects): {metrics['APl']:.4f}")

    # Create bar chart of key metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    values = [metrics[m] for m in metrics_to_plot]
    
    plt.bar(metrics_to_plot, values, color='skyblue')
    plt.ylim(0, 1.0)  # AP values range from 0 to 1
    plt.ylabel('Score')
    plt.title('COCO Evaluation Metrics')
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Save figure
    output_path = os.path.join(os.path.dirname(metrics_file), 'metrics_chart.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nMetrics chart saved to: {output_path}")
    
    # Return the metrics for use elsewhere
    return metrics

def visualize_predictions(predictions_file, test_images_dir, num_samples=5, score_threshold=0.5):
    """Visualize predictions on test images"""
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        return

    print(f"\n===== PREDICTION STATISTICS =====")
    print(f"Total predictions: {len(predictions)}")
    
    # Group predictions by image
    images_dict = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in images_dict:
            images_dict[img_id] = []
        images_dict[img_id].append(pred)
    
    print(f"Number of images with predictions: {len(images_dict)}")
    
    # Calculate average predictions per image
    preds_per_image = [len(preds) for preds in images_dict.values()]
    print(f"Average predictions per image: {np.mean(preds_per_image):.1f}")
    
    # Calculate confidence distribution
    confidences = [pred['score'] for pred in predictions]
    print(f"Mean confidence score: {np.mean(confidences):.4f}")
    print(f"Median confidence score: {np.median(confidences):.4f}")
    
    # Plot confidence histogram
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Confidence Scores')
    conf_hist_path = os.path.join(os.path.dirname(predictions_file), 'confidence_histogram.png')
    plt.savefig(conf_hist_path)
    plt.close()
    print(f"Confidence histogram saved to: {conf_hist_path}")
    
    # Plot class distribution (top 20 classes)
    class_counts = {}
    for pred in predictions:
        if pred['score'] >= score_threshold:
            class_id = pred['category_id']
            class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Sort and get top classes
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh([x[0] for x in top_classes], [x[1] for x in top_classes], color='green')
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.title(f'Top 20 Detected Classes (confidence ≥ {score_threshold})')
    plt.tight_layout()
    class_dist_path = os.path.join(os.path.dirname(predictions_file), 'class_distribution.png')
    plt.savefig(class_dist_path)
    plt.close()
    print(f"Class distribution chart saved to: {class_dist_path}")
    
    # Visualize sample images with predictions
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Sample random images for visualization
    sample_ids = random.sample(list(images_dict.keys()), min(num_samples, len(images_dict)))
    
    output_viz_dir = os.path.join(os.path.dirname(predictions_file), 'visualizations')
    os.makedirs(output_viz_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations for {len(sample_ids)} sample images...")
    
    for img_id in sample_ids:
        # Get image filename (COCO images are named with 12-digit zero-padded IDs)
        img_file = os.path.join(test_images_dir, f"{img_id:012d}.jpg")
        if not os.path.exists(img_file):
            continue
            
        img = Image.open(img_file)
        
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)
        
        # Get predictions for this image
        img_preds = images_dict[img_id]
        
        # Plot bounding boxes
        for pred in img_preds:
            if pred['score'] > score_threshold:
                bbox = pred['bbox']  # [x, y, width, height]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Get class name
                class_id = pred['category_id']
                class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
                
                # Add label
                ax.text(
                    bbox[0], bbox[1]-5, 
                    f"{class_name}: {pred['score']:.2f}", 
                    color='white', fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.8)
                )
        
        ax.set_title(f"Image ID: {img_id} - {len([p for p in img_preds if p['score'] > score_threshold])} detections")
        plt.axis('off')
        
        # Save figure
        viz_path = os.path.join(output_viz_dir, f"detection_{img_id}.png")
        plt.savefig(viz_path, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_viz_dir}")

def run_evaluation(model_path, val_images, val_annotations, output_dir="outputs"):
    """Run evaluation and generate metrics"""
    import subprocess
    import sys
    
    print("\n===== RUNNING MODEL EVALUATION =====")
    cmd = [
        sys.executable, "-m", "dino_detector.train",
        "--val_images", val_images,
        "--val_annotations", val_annotations,
        "--checkpoint", model_path,
        "--output_dir", output_dir,
        "--only_evaluate"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    # Look for generated metrics file
    metrics_files = [f for f in os.listdir(output_dir) if f.startswith("val_metrics_epoch_") and f.endswith(".json")]
    if metrics_files:
        latest_metrics = sorted(metrics_files)[-1]
        metrics_path = os.path.join(output_dir, latest_metrics)
        print(f"Found metrics file: {metrics_path}")
        return metrics_path
    else:
        print("No metrics file generated")
        return None

def print_metrics_table(metrics):
    """Print a nicely formatted table of metrics"""
    if not metrics:
        return
        
    print("\n" + "="*50)
    print("                EVALUATION METRICS SUMMARY                ")
    print("="*50)
    print(f"| {'Metric':<12} | {'Value':<10} | {'Description':<25} |")
    print("|" + "-"*14 + "|" + "-"*12 + "|" + "-"*27 + "|")
    
    metric_descriptions = {
        'AP': 'Average Precision (IoU=0.5:0.95)',
        'AP50': 'Average Precision (IoU=0.5)',
        'AP75': 'Average Precision (IoU=0.75)',
        'APs': 'AP for small objects',
        'APm': 'AP for medium objects',
        'APl': 'AP for large objects'
    }
    
    for metric, description in metric_descriptions.items():
        value = metrics.get(metric, 0)
        print(f"| {metric:<12} | {value:>8.4f}  | {description:<25} |")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze COCO detection results")
    parser.add_argument("--metrics_file", type=str, help="Path to validation metrics JSON file")
    parser.add_argument("--predictions_file", type=str, help="Path to test predictions JSON file")
    parser.add_argument("--test_images", type=str, help="Path to test image directory")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for visualizations")
    
    # Arguments for running evaluation
    parser.add_argument("--run_eval", action="store_true", help="Run evaluation first")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint for evaluation")
    parser.add_argument("--val_images", type=str, help="Path to validation images")
    parser.add_argument("--val_annotations", type=str, help="Path to validation annotations")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    metrics = None
    
    # Run evaluation if requested
    if args.run_eval:
        if not args.model_path or not args.val_images or not args.val_annotations:
            print("Error: model_path, val_images, and val_annotations are required for evaluation")
            return
        
        metrics_file = run_evaluation(
            args.model_path, 
            args.val_images, 
            args.val_annotations, 
            args.output_dir
        )
        
        if metrics_file:
            metrics = analyze_metrics(metrics_file)
    
    # Analyze metrics if file provided
    elif args.metrics_file:
        if not os.path.exists(args.metrics_file):
            print(f"Error: Metrics file not found: {args.metrics_file}")
            return
        
        metrics = analyze_metrics(args.metrics_file)
    
    # Visualize predictions if file provided
    if args.predictions_file and args.test_images:
        if not os.path.exists(args.predictions_file):
            print(f"Error: Predictions file not found: {args.predictions_file}")
            return
            
        if not os.path.exists(args.test_images):
            print(f"Error: Test images directory not found: {args.test_images}")
            return
            
        visualize_predictions(
            args.predictions_file,
            args.test_images,
            args.num_visualizations,
            args.confidence_threshold
        )
    
    # Print final metrics table
    if metrics:
        print_metrics_table(metrics)

if __name__ == "__main__":
    main()