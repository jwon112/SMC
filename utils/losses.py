"""
Custom Loss Functions for SMC Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 클래스별 가중치 (tensor 또는 list). None이면 balanced weight 자동 계산
        gamma: focusing parameter (기본값: 2.0)
        reduction: 'mean' 또는 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - 모델 출력 (logits)
            targets: (N,) - 정답 레이블
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE_loss)
        
        # 클래스별 가중치 적용
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLabelFocalLoss(nn.Module):
    """Multi-label Focal Loss for handling class imbalance in multi-label classification
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: 클래스별 가중치 (tensor 또는 list). None이면 balanced weight 자동 계산
        gamma: focusing parameter (기본값: 2.0)
        reduction: 'mean' 또는 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - 모델 출력 (logits)
            targets: (N, C) - multi-label binary targets (0 or 1)
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # p_t 계산 (sigmoid 확률)
        pt = torch.sigmoid(inputs)
        # positive class에 대한 p_t
        pt_positive = pt * targets + (1 - pt) * (1 - targets)
        
        # Focal Loss
        focal_loss = (1 - pt_positive) ** self.gamma * bce_loss
        
        # 클래스별 가중치 적용
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # alpha를 (C,)에서 (1, C)로 확장하여 브로드캐스팅
            alpha_t = self.alpha.unsqueeze(0).expand_as(targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_focal_loss_alpha(train_labels, num_classes=None):
    """Focal Loss용 클래스 가중치 계산
    
    Args:
        train_labels: 학습 데이터 레이블 리스트 또는 numpy array
            - Single-label: (N,) 형태의 클래스 인덱스
            - Multi-label: (N, C) 형태의 binary matrix
        num_classes: 클래스 개수 (None이면 자동 계산)
    
    Returns:
        alpha: 클래스별 가중치 (numpy array)
    """
    train_labels = np.array(train_labels)
    
    # Multi-label인지 확인
    if train_labels.ndim == 2:
        # Multi-label: 각 클래스별로 positive 샘플 수 계산
        if num_classes is None:
            num_classes = train_labels.shape[1]
        
        # 각 클래스별 positive 샘플 수
        class_counts = train_labels.sum(axis=0)
        total_samples = len(train_labels)
        
        # Balanced weight 계산: n_samples / (n_classes * np.bincount(y))
        # Multi-label의 경우: total_samples / (num_classes * class_counts)
        class_weights = total_samples / (num_classes * (class_counts + 1e-6))  # 0 방지
        
    else:
        # Single-label
        if num_classes is None:
            num_classes = len(np.unique(train_labels))
        
        # Balanced class weight 계산
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=train_labels
        )
    
    return class_weights

