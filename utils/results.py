"""
Results Saving Utilities
"""

import os
import pandas as pd
from datetime import datetime


def save_results_to_csv(results, csv_path, mode='append'):
    """결과를 CSV 파일로 저장
    
    Args:
        results: dict 또는 list of dict
        csv_path: CSV 파일 경로
        mode: 'append' 또는 'overwrite'
    """
    # DataFrame 생성
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame(results)
    
    # 파일 존재 여부 확인
    if os.path.exists(csv_path) and mode == 'append':
        # 기존 파일 읽기
        existing_df = pd.read_csv(csv_path)
        # 병합
        df = pd.concat([existing_df, df], ignore_index=True)
    
    # 저장
    df.to_csv(csv_path, index=False)


def create_result_dict(
    model_name,
    image_type,
    epoch=None,
    train_loss=None,
    val_loss=None,
    val_accuracy=None,
    val_f1_macro=None,
    val_f1_weighted=None,
    val_precision=None,
    val_recall=None,
    test_accuracy=None,
    test_f1_macro=None,
    test_f1_weighted=None,
    test_precision=None,
    test_recall=None,
    improvement_accuracy=None,
    improvement_f1=None,
):
    """결과 딕셔너리 생성"""
    result = {
        'model': model_name,
        'image_type': image_type,
    }
    
    if epoch is not None:
        result['epoch'] = epoch
    if train_loss is not None:
        result['train_loss'] = train_loss
    if val_loss is not None:
        result['val_loss'] = val_loss
    if val_accuracy is not None:
        result['val_accuracy'] = val_accuracy
    if val_f1_macro is not None:
        result['val_f1_macro'] = val_f1_macro
    if val_f1_weighted is not None:
        result['val_f1_weighted'] = val_f1_weighted
    if val_precision is not None:
        result['val_precision'] = val_precision
    if val_recall is not None:
        result['val_recall'] = val_recall
    if test_accuracy is not None:
        result['test_accuracy'] = test_accuracy
    if test_f1_macro is not None:
        result['test_f1_macro'] = test_f1_macro
    if test_f1_weighted is not None:
        result['test_f1_weighted'] = test_f1_weighted
    if test_precision is not None:
        result['test_precision'] = test_precision
    if test_recall is not None:
        result['test_recall'] = test_recall
    if improvement_accuracy is not None:
        result['improvement_accuracy'] = improvement_accuracy
    if improvement_f1 is not None:
        result['improvement_f1'] = improvement_f1
    
    return result


def calculate_improvement(original_metrics, processed_metrics):
    """성능 증감 계산
    
    Args:
        original_metrics: 원본 이미지 지표 dict
        processed_metrics: 전처리 이미지 지표 dict
    
    Returns:
        dict: improvement_accuracy, improvement_f1
    """
    improvement_accuracy = None
    improvement_f1 = None
    
    if 'accuracy' in original_metrics and 'accuracy' in processed_metrics:
        if original_metrics['accuracy'] > 0:
            improvement_accuracy = (
                (processed_metrics['accuracy'] - original_metrics['accuracy']) 
                / original_metrics['accuracy'] * 100
            )
    
    if 'f1_macro' in original_metrics and 'f1_macro' in processed_metrics:
        if original_metrics['f1_macro'] > 0:
            improvement_f1 = (
                (processed_metrics['f1_macro'] - original_metrics['f1_macro']) 
                / original_metrics['f1_macro'] * 100
            )
    
    return {
        'improvement_accuracy': improvement_accuracy,
        'improvement_f1': improvement_f1,
    }

