"""
Evaluation Utilities
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def calculate_metrics(y_true, y_pred):
    """지표 계산
    
    Returns:
        dict: accuracy, f1_macro, f1_weighted, precision, recall, confusion_matrix
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
    }


def get_classification_report(y_true, y_pred, class_names=None):
    """분류 리포트 생성"""
    if class_names is None:
        class_names = ['BN', 'CN', 'EF', 'NI']
    
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

