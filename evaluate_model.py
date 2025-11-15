"""
Evaluate Unified Classifier Performance
Shows per-class accuracy and confusion matrix
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================

class Config:
    MNIST_TRAIN_DIR = 'mnist_competition/train'
    MNIST_TRAIN_CSV = 'mnist_competition/train_label.csv'
    SHAPES_DIR = 'Shapes_Classifier/dataset/output'
    MODEL_PATH = 'unified_model_19classes_best.pth'
    LABEL_MAPPING_PATH = 'label_mapping.json'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 19
    INPUT_SIZE = 128  # Updated to match new training size
    BATCH_SIZE = 128

# ============================================================================
# Dataset
# ============================================================================

class UnifiedDataset(Dataset):
    def __init__(self, mnist_df, shapes_df, mnist_dir, shapes_dir,
                 shape_to_id, transform=None, sample_fraction=1.0):
        self.data_list = []
        
        # Add MNIST data (labels 0-9)
        if mnist_df is not None:
            for idx, row in mnist_df.iterrows():
                self.data_list.append({
                    'path': os.path.join(mnist_dir, row['image_name']),
                    'label': int(row['label']),
                    'source': 'mnist'
                })
        
        # Add shapes data (labels 10-18) 
        if shapes_df is not None:
            shapes_df_sampled = shapes_df.sample(frac=sample_fraction, random_state=42)
            for idx, row in shapes_df_sampled.iterrows():
                self.data_list.append({
                    'path': os.path.join(shapes_dir, row['image_name']),
                    'label': shape_to_id[row['label']],
                    'source': 'shape',
                    'shape_name': row['label']
                })
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['path']).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

# ============================================================================
# Evaluation
# ============================================================================

def load_model(model_path, device):
    """Load trained model."""
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, Config.NUM_CLASSES)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, dataloader, device, id_to_label):
    """Evaluate model and return predictions."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved confusion_matrix.png")
    plt.close()
    
    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    plt.title(f'{title} (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved confusion_matrix_normalized.png")
    plt.close()

def analyze_class_performance(y_true, y_pred, y_probs, id_to_label):
    """Analyze per-class performance."""
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    results = []
    
    for class_id in range(Config.NUM_CLASSES):
        class_mask = y_true == class_id
        if class_mask.sum() == 0:
            continue
            
        class_preds = y_pred[class_mask]
        class_true = y_true[class_mask]
        class_probs = y_probs[class_mask, class_id]
        
        accuracy = (class_preds == class_true).mean()
        avg_confidence = class_probs.mean()
        
        # Find most common misclassification
        misclassified = class_preds[class_preds != class_true]
        if len(misclassified) > 0:
            unique, counts = np.unique(misclassified, return_counts=True)
            most_common_error = unique[counts.argmax()]
            error_rate = counts.max() / len(class_preds)
            error_label = id_to_label[str(most_common_error)]
        else:
            error_label = "N/A"
            error_rate = 0.0
        
        results.append({
            'Class': id_to_label[str(class_id)],
            'Total': class_mask.sum(),
            'Accuracy': f"{accuracy:.2%}",
            'Avg Confidence': f"{avg_confidence:.4f}",
            'Most Common Error': error_label,
            'Error Rate': f"{error_rate:.2%}"
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('per_class_performance.csv', index=False)
    print(f"\n✅ Saved per_class_performance.csv")
    
    return df

def main():
    print("="*70)
    print("UNIFIED MODEL EVALUATION")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_PATH}")
    print("="*70 + "\n")
    
    # Load label mapping
    with open(Config.LABEL_MAPPING_PATH, 'r') as f:
        id_to_label = json.load(f)
    
    print(f"✅ Loaded label mapping: {len(id_to_label)} classes\n")
    
    # Load data
    print("Loading test data...")
    
    # Load MNIST
    mnist_df = pd.read_csv(Config.MNIST_TRAIN_CSV)
    mnist_train, mnist_test = train_test_split(
        mnist_df, test_size=0.15, random_state=42, stratify=mnist_df['label']
    )
    print(f"✅ MNIST test: {len(mnist_test)} images")
    
    # Load Shapes  
    shape_files = [f for f in os.listdir(Config.SHAPES_DIR) if f.endswith('.png')]
    shape_labels = [f.split('_')[0] for f in shape_files]
    shapes_df = pd.DataFrame({'image_name': shape_files, 'label': shape_labels})
    
    shape_names = sorted(shapes_df['label'].unique())
    shape_to_id = {name: idx + 10 for idx, name in enumerate(shape_names)}
    
    shapes_train, shapes_test = train_test_split(
        shapes_df, test_size=0.15, random_state=42, stratify=shapes_df['label']
    )
    print(f"✅ Shapes test: {len(shapes_test)} images (will use 67%)")
    
    # Create test dataset
    test_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = UnifiedDataset(
        mnist_test, shapes_test,
        Config.MNIST_TRAIN_DIR, Config.SHAPES_DIR,
        shape_to_id, transform=test_transform,
        sample_fraction=0.67
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✅ Test dataset: {len(test_dataset)} images\n")
    
    # Load model
    print("Loading model...")
    model = load_model(Config.MODEL_PATH, Config.DEVICE)
    print(f"✅ Model loaded\n")
    
    # Evaluate
    print("Evaluating model...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, Config.DEVICE, id_to_label)
    
    overall_acc = (y_true == y_pred).mean()
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY: {overall_acc:.2%}")
    print(f"{'='*70}\n")
    
    # Per-class analysis
    df_results = analyze_class_performance(y_true, y_pred, y_probs, id_to_label)
    
    # Confusion matrix
    print("\nGenerating confusion matrices...")
    labels = [id_to_label[str(i)] for i in range(Config.NUM_CLASSES)]
    plot_confusion_matrix(y_true, y_pred, labels)
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Save detailed report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('classification_report.csv')
    print(f"✅ Saved classification_report.csv")
    
    # Find problematic pairs
    print("\n" + "="*70)
    print("TOP CONFUSION PAIRS")
    print("="*70)
    
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []
    
    for i in range(Config.NUM_CLASSES):
        for j in range(Config.NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'True Class': labels[i],
                    'Predicted As': labels[j],
                    'Count': cm[i, j],
                    'Percentage': f"{cm[i, j] / cm[i].sum():.2%}"
                })
    
    confusion_df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)
    print(confusion_df.head(20).to_string(index=False))
    confusion_df.to_csv('confusion_pairs.csv', index=False)
    print(f"\n✅ Saved confusion_pairs.csv")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - per_class_performance.csv")
    print("  - classification_report.csv")
    print("  - confusion_pairs.csv")
    print("="*70)

if __name__ == '__main__':
    main()

