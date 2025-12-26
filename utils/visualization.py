"""
Visualization Utilities
"""

import os
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_comparison(results_df, save_path, metric='test_accuracy'):
    """모델별 원본 vs 전처리 성능 비교 그래프
    
    Args:
        results_df: 결과 DataFrame (model, image_type, test_accuracy, test_f1_macro 등 포함)
        save_path: 저장 경로
        metric: 비교할 지표 ('test_accuracy', 'test_f1_macro', 'accuracy', 'f1_macro' 등)
    """
    # 모델별로 그룹화
    models = results_df['model'].unique()
    
    original_metrics = []
    processed_metrics = []
    model_names = []
    
    for model in models:
        model_data = results_df[results_df['model'] == model]
        original = model_data[model_data['image_type'] == 'original']
        processed = model_data[model_data['image_type'] == 'processed']
        
        if len(original) > 0 and len(processed) > 0:
            original_metrics.append(original[metric].values[0])
            processed_metrics.append(processed[metric].values[0])
            model_names.append(model)
    
    # 그래프 생성
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, original_metrics, width, label='Original', alpha=0.8)
    bars2 = ax.bar(x + width/2, processed_metrics, width, label='Processed', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Model Comparison: Original vs Processed ({metric.capitalize()})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Confusion Matrix 시각화
    
    Args:
        cm: confusion matrix (numpy array)
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
        title: 그래프 제목
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_improvement(results_df, save_path):
    """성능 증감 폭 시각화
    
    Args:
        results_df: 결과 DataFrame (improvement_accuracy, improvement_f1 포함)
        save_path: 저장 경로
    """
    # 모델별 improvement 계산
    models = results_df['model'].unique()
    improvements_acc = []
    improvements_f1 = []
    model_names = []
    
    for model in models:
        model_data = results_df[results_df['model'] == model]
        processed = model_data[model_data['image_type'] == 'processed']
        
        if len(processed) > 0 and 'improvement_accuracy' in processed.columns:
            improvements_acc.append(processed['improvement_accuracy'].values[0])
            improvements_f1.append(processed['improvement_f1'].values[0])
            model_names.append(model)
    
    # 그래프 생성
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, improvements_acc, width, label='Accuracy Improvement (%)', alpha=0.8)
    bars2 = ax.bar(x + width/2, improvements_f1, width, label='F1-Score Improvement (%)', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvement: Processed vs Original', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_absolute_comparison(results_df, save_path, metric='test_accuracy', metric_name='Accuracy'):
    """절대 성능 값 비교 그래프 (원본 vs 전처리)
    
    Args:
        results_df: 결과 DataFrame (model, image_type, test_accuracy, test_f1_macro 등 포함)
        save_path: 저장 경로
        metric: 비교할 지표 컬럼명 ('test_accuracy', 'test_f1_macro')
        metric_name: 지표 이름 (그래프 제목용)
    """
    # 모델별로 그룹화
    models = results_df['model'].unique()
    
    original_metrics = []
    processed_metrics = []
    model_names = []
    
    for model in models:
        model_data = results_df[results_df['model'] == model]
        original = model_data[model_data['image_type'] == 'original']
        processed = model_data[model_data['image_type'] == 'processed']
        
        if len(original) > 0 and len(processed) > 0 and metric in original.columns and metric in processed.columns:
            original_metrics.append(original[metric].values[0])
            processed_metrics.append(processed[metric].values[0])
            model_names.append(model)
    
    if len(model_names) == 0:
        print(f"Warning: No data found for metric {metric}")
        return
    
    # 그래프 생성
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, original_metrics, width, label='Original', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, processed_metrics, width, label='Processed', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
    ax.set_title(f'Absolute Performance Comparison: {metric_name} (Original vs Processed)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(original_metrics), max(processed_metrics)) * 1.15])
    
    # 값 표시 (막대 위에)
    for i, (orig, proc) in enumerate(zip(original_metrics, processed_metrics)):
        # 원본 값
        ax.text(i - width/2, orig + max(max(original_metrics), max(processed_metrics)) * 0.01,
               f'{orig:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        # 전처리 값
        ax.text(i + width/2, proc + max(max(original_metrics), max(processed_metrics)) * 0.01,
               f'{proc:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        # 차이 표시
        diff = proc - orig
        diff_text = f'{diff:+.4f}' if abs(diff) >= 0.0001 else f'{diff:.2e}'
        ax.text(i, max(orig, proc) + max(max(original_metrics), max(processed_metrics)) * 0.05,
               diff_text,
               ha='center', va='bottom', fontsize=9,
               color='green' if diff > 0 else 'red' if diff < 0 else 'black',
               fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history_df, save_path, model_name, image_type):
    """학습 곡선 시각화
    
    Args:
        history_df: 에포크별 결과 DataFrame
        save_path: 저장 경로
        model_name: 모델 이름
        image_type: 이미지 타입
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss 곡선
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - {image_type}: Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy 곡선
    axes[1].plot(history_df['epoch'], history_df['val_accuracy'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - {image_type}: Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(results_dir, results_df, detailed_df=None):
    """모든 시각화 생성
    
    Args:
        results_dir: 결과 디렉토리
        results_df: 최종 결과 DataFrame
        detailed_df: 에포크별 상세 결과 DataFrame (선택)
    """
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 성능 비교 그래프 (각 지표별)
    # test_ 컬럼이 있으면 우선 사용, 없으면 일반 컬럼 사용
    test_metrics = ['test_accuracy', 'test_f1_macro', 'test_f1_weighted', 'test_precision', 'test_recall']
    general_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']
    
    for test_metric, general_metric in zip(test_metrics, general_metrics):
        if test_metric in results_df.columns:
            # test_ 컬럼 사용
            metric_name = general_metric.replace('_', ' ').title()
            save_path = os.path.join(vis_dir, f'comparison_{general_metric}.png')
            plot_comparison(results_df, save_path, metric=test_metric)
        elif general_metric in results_df.columns:
            # 일반 컬럼 사용
            save_path = os.path.join(vis_dir, f'comparison_{general_metric}.png')
            plot_comparison(results_df, save_path, metric=general_metric)
    
    # 절대 성능 값 비교 그래프 (Accuracy, F1-Score 별도)
    # Accuracy 절대값 비교
    if 'test_accuracy' in results_df.columns:
        save_path = os.path.join(vis_dir, 'absolute_comparison_accuracy.png')
        plot_absolute_comparison(results_df, save_path, metric='test_accuracy', metric_name='Accuracy')
    
    # F1-Score 절대값 비교
    if 'test_f1_macro' in results_df.columns:
        save_path = os.path.join(vis_dir, 'absolute_comparison_f1_score.png')
        plot_absolute_comparison(results_df, save_path, metric='test_f1_macro', metric_name='F1-Score (Macro)')
    
    # 성능 증감 폭 (기존 유지)
    if 'improvement_accuracy' in results_df.columns:
        save_path = os.path.join(vis_dir, 'improvement.png')
        plot_improvement(results_df, save_path)
    
    # 학습 곡선 (상세 결과가 있는 경우)
    if detailed_df is not None:
        models = detailed_df['model'].unique()
        image_types = detailed_df['image_type'].unique()
        
        for model in models:
            for img_type in image_types:
                model_data = detailed_df[
                    (detailed_df['model'] == model) & 
                    (detailed_df['image_type'] == img_type)
                ]
                if len(model_data) > 0:
                    save_path = os.path.join(vis_dir, f'{model}_{img_type}_curves.png')
                    plot_training_curves(model_data, save_path, model, img_type)

