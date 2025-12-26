"""
Training and Evaluation Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    hamming_loss, jaccard_score
)
import numpy as np


def train_epoch(model, train_loader, criterion, optimizer, device, rank=0):
    """한 에포크 학습
    
    Returns:
        average_loss: 평균 손실
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    average_loss = running_loss / total_samples
    return average_loss


def validate(model, val_loader, criterion, device, rank=0, threshold=0.5):
    """검증 실행 (Multi-label 지원)
    
    Args:
        threshold: Multi-label 예측을 위한 threshold (기본값: 0.5)
    
    Returns:
        dict: {'loss', 'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
               'precision', 'recall', 'hamming_loss', 'jaccard_score', 'confusion_matrix'}
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Multi-label 예측: sigmoid + threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 지표 계산
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    average_loss = running_loss / len(all_labels)
    
    # Multi-label 지표 계산
    # Subset accuracy (exact match ratio)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Precision and Recall
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Hamming loss (낮을수록 좋음)
    hamming = hamming_loss(all_labels, all_preds)
    
    # Jaccard score (평균)
    jaccard = jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Confusion matrix는 각 클래스별로 계산 (첫 번째 클래스 기준)
    # Multi-label의 경우 모든 클래스에 대한 평균 confusion matrix는 의미가 없으므로
    # 첫 번째 클래스만 사용 (하위 호환성)
    cm = confusion_matrix(all_labels[:, 0], all_preds[:, 0])
    
    return {
        'loss': average_loss,
        'accuracy': accuracy,  # Subset accuracy
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'confusion_matrix': cm,
    }


def test(model, test_loader, criterion, device, rank=0, threshold=0.5):
    """테스트 실행 (Multi-label 지원)
    
    Args:
        threshold: Multi-label 예측을 위한 threshold (기본값: 0.5)
    
    Returns:
        dict: validate와 동일한 형식
    """
    return validate(model, test_loader, criterion, device, rank, threshold)

