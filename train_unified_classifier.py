"""
Train Unified Classifier for 19 classes (Digits + Shapes)
Usage: python train_unified_classifier.py --epochs 10 --batch-size 64
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    MNIST_TRAIN_DIR = 'mnist_competition/train'
    MNIST_TRAIN_CSV = 'mnist_competition/train_label.csv'
    SHAPES_DIR = 'Shapes_Classifier/dataset/output'
    
    # Training
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    NUM_CLASSES = 19
    INPUT_SIZE = 128  # Increased from 64 to better distinguish high-edge shapes
    
    # Output
    MODEL_PATH = 'unified_model_19classes_best.pth'
    LABEL_MAPPING_PATH = 'label_mapping.json'

# ============================================================================
# Dataset
# ============================================================================

class UnifiedDataset(Dataset):
    """Unified dataset for MNIST digits and geometric shapes."""
    
    def __init__(self, mnist_df, shapes_df, mnist_dir, shapes_dir,
                 shape_to_id, transform=None, sample_fraction=0.67):
        """
        Args:
            mnist_df: DataFrame with MNIST data
            shapes_df: DataFrame with shapes data
            mnist_dir: Directory with MNIST images
            shapes_dir: Directory with shape images
            shape_to_id: Mapping from shape name to class ID (10-18)
            transform: Image transforms
            sample_fraction: Fraction of shapes to use (balance with MNIST)
        """
        # Sample shapes to balance dataset
        shapes_df_sampled = shapes_df.sample(frac=sample_fraction, random_state=42)
        
        self.data_list = []
        
        # Add MNIST data (labels 0-9)
        for idx, row in mnist_df.iterrows():
            self.data_list.append({
                'path': os.path.join(mnist_dir, row['image_name']),
                'label': int(row['label']),
                'source': 'mnist'
            })
        
        # Add shapes data (labels 10-18)
        for idx, row in shapes_df_sampled.iterrows():
            self.data_list.append({
                'path': os.path.join(shapes_dir, row['image_name']),
                'label': shape_to_id[row['label']],
                'source': 'shape'
            })
        
        self.transform = transform
        
        print(f"✅ Dataset created: {len(self.data_list)} images")
        print(f"   - MNIST: {len(mnist_df)} images (classes 0-9)")
        print(f"   - Shapes: {len(shapes_df_sampled)} images (classes 10-18)")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['path']).convert('L')
        
        # Apply preprocessing (CLAHE for contrast enhancement)
        # Convert PIL to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Apply CLAHE for better contrast (especially for real-world images)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img_enhanced = clahe.apply(img_array)
        
        # Convert back to PIL
        image = Image.fromarray(img_enhanced)
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{running_loss/(pbar.n+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{running_loss/(pbar.n+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================================
# Main Training Loop
# ============================================================================

def main(args):
    print("="*60)
    print("UNIFIED CLASSIFIER TRAINING (19 CLASSES)")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60 + "\n")
    
    # ========================================
    # Load Data
    # ========================================
    print("Loading datasets...")
    
    # Load MNIST
    mnist_df = pd.read_csv(Config.MNIST_TRAIN_CSV)
    print(f"✅ MNIST: {len(mnist_df)} images")
    
    # Load Shapes
    shape_files = [f for f in os.listdir(Config.SHAPES_DIR) if f.endswith('.png')]
    shape_labels = [f.split('_')[0] for f in shape_files]
    shapes_df = pd.DataFrame({'image_name': shape_files, 'label': shape_labels})
    print(f"✅ Shapes: {len(shapes_df)} images")
    
    # Create label mapping
    shape_names = sorted(shapes_df['label'].unique())
    shape_to_id = {name: idx + 10 for idx, name in enumerate(shape_names)}
    id_to_label = {i: str(i) for i in range(10)}
    id_to_label.update({v: k for k, v in shape_to_id.items()})
    
    print(f"\n📋 Label Mapping:")
    for class_id, label_name in sorted(id_to_label.items()):
        print(f"   Class {class_id:2d}: {label_name}")
    
    # Save label mapping
    with open(Config.LABEL_MAPPING_PATH, 'w') as f:
        json.dump(id_to_label, f, indent=2)
    print(f"✅ Saved {Config.LABEL_MAPPING_PATH}\n")
    
    # ========================================
    # Data Splits & Transforms
    # ========================================
    
    # Transforms: Grayscale → RGB, Resize to 128x128 with balanced augmentation
    # Reduced perspective distortion to preserve shape details (especially for Nonagon/Octagon)
    train_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(30),  # Keep rotation for robustness
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Translation OK
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # REDUCED: was 0.2/0.5, now 0.1/0.3 to preserve shape edges
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # REDUCED: was 0.3, now 0.2 to preserve contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    mnist_train, mnist_val = train_test_split(
        mnist_df, test_size=0.15, random_state=42, stratify=mnist_df['label']
    )
    shapes_train, shapes_val = train_test_split(
        shapes_df, test_size=0.15, random_state=42, stratify=shapes_df['label']
    )
    
    print(f"📊 Data Split:")
    print(f"   Train: MNIST {len(mnist_train)} + Shapes ~{int(len(shapes_train)*0.67)}")
    print(f"   Val:   MNIST {len(mnist_val)} + Shapes ~{int(len(shapes_val)*0.67)}\n")
    
    # Create datasets
    train_dataset = UnifiedDataset(
        mnist_train, shapes_train,
        Config.MNIST_TRAIN_DIR, Config.SHAPES_DIR,
        shape_to_id, transform=train_transform,
        sample_fraction=0.67
    )
    
    val_dataset = UnifiedDataset(
        mnist_val, shapes_val,
        Config.MNIST_TRAIN_DIR, Config.SHAPES_DIR,
        shape_to_id, transform=val_transform,
        sample_fraction=0.67
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"✅ DataLoaders ready\n")
    
    # ========================================
    # Model
    # ========================================
    print("Loading EfficientNet-B0...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify classifier for 19 classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready: {total_params:,} parameters\n")
    
    # ========================================
    # Training Setup
    # ========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # ========================================
    # Training Loop
    # ========================================
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_mapping': id_to_label,
                'config': vars(args)
            }, Config.MODEL_PATH)
            print(f"✅ Saved best model: {Config.MODEL_PATH} (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {Config.MODEL_PATH}")
    print(f"{'='*60}")
    
    return history

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unified Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    history = main(args)
    
    print("\n✅ Training script completed successfully!")

