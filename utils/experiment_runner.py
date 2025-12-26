"""
Experiment Runner for SMC Classification
"""

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import pandas as pd

from SMC.models import get_model
from SMC.dataloaders import get_smc_data_loaders
from SMC.utils.distributed import setup_distributed, cleanup_distributed, is_main_process
from SMC.utils.training import train_epoch, validate, test
from SMC.utils.results import (
    save_results_to_csv,
    create_result_dict,
    calculate_improvement,
)
from SMC.utils.visualization import create_all_visualizations, plot_confusion_matrix
from SMC.utils.split_utils import get_fold_dataframes
from SMC.utils.losses import MultiLabelFocalLoss, compute_focal_loss_alpha
import numpy as np
import numpy as np


def run_classification_experiment(
    data_path,
    models=['vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'densenet'],
    epochs=50,
    batch_size=32,
    lr=0.001,
    optimizer_name='adam',
    image_size=224,
    num_workers=4,
    seed=24,
    results_dir=None,
    use_5fold=False,
):
    """SMC Classification 실험 실행
    
    Args:
        data_path: SMC/data 디렉토리 경로
        models: 실험할 모델 리스트
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        lr: 학습률
        optimizer_name: 'adam' 또는 'sgd'
        image_size: 이미지 리사이즈 크기
        num_workers: DataLoader 워커 수
        seed: 랜덤 시드
        results_dir: 결과 저장 디렉토리 (None이면 자동 생성)
        use_5fold: 5-fold cross-validation 사용 여부
    """
    
    # 결과 디렉토리 생성
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"SMC/results/experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
    
    # DDP 설정
    distributed, rank, local_rank, world_size = setup_distributed()
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"\nUsing DDP with world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
    
    # 시드 설정
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 5-fold CV 모드 설정
    if use_5fold:
        fold_list = list(range(5))
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"5-Fold Cross-Validation Mode")
            print(f"{'='*60}")
    else:
        fold_list = [None]
    
    # 결과 저장용
    all_results = []
    all_detailed_results = []
    fold_results = {}  # 5-fold 결과 집계용
    
    # 데이터셋 캐싱 (fold별, image_type별로 재사용)
    dataset_cache = {}  # key: (fold_idx, image_type, image_size)
    
    # 이미지 타입별로 실험
    image_types = ['original', 'processed']
    
    # Fold별 루프 (5-fold인 경우)
    for fold_idx in fold_list:
        if use_5fold and is_main_process(rank):
            print(f"\n{'#'*60}")
            print(f"Fold {fold_idx + 1}/5")
            print(f"{'#'*60}")
        
        # 5-fold split 데이터 준비
        if use_5fold:
            train_df, val_df, test_df = get_fold_dataframes(data_path, fold_idx, seed)
        else:
            train_df = val_df = test_df = None
        
        # 모델별 루프
        for model_name in models:
                print(f"\n{'='*60}")
                print(f"Model: {model_name.upper()}")
                print(f"{'='*60}")
                
                model_results = {}
                
                # 원본/전처리 이미지 각각 실험
                for image_type in image_types:
                    print(f"\n--- {image_type.upper()} Images ---")
                    
                    # 데이터셋 캐싱 키
                    cache_key = (fold_idx, image_type, image_size)
                    
                    # 데이터셋이 캐시에 없으면 생성
                    if cache_key not in dataset_cache:
                        if is_main_process(rank):
                            print(f"  [Cache Miss] Creating dataset for {image_type} images (fold {fold_idx})...")
                        # 데이터셋만 생성 (DataLoader는 나중에)
                        train_loader_temp, val_loader_temp, test_loader_temp, _, _, _ = get_smc_data_loaders(
                            data_base_dir=data_path,
                            image_type=image_type,
                            batch_size=batch_size,
                            num_workers=0,  # 데이터셋만 생성
                            image_size=image_size,
                            distributed=False,  # 데이터셋만 생성
                            world_size=None,
                            rank=None,
                            seed=seed,
                            use_5fold=use_5fold,
                            fold_idx=fold_idx,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            train_dataset=None,  # 명시적으로 None 전달
                            val_dataset=None,
                            test_dataset=None,
                        )
                        # 데이터셋 추출하여 캐시에 저장
                        dataset_cache[cache_key] = (
                            train_loader_temp.dataset,
                            val_loader_temp.dataset,
                            test_loader_temp.dataset
                        )
                        if is_main_process(rank):
                            print(f"  [Cache] Dataset created and cached for {image_type} images (fold {fold_idx})")
                    else:
                        if is_main_process(rank):
                            print(f"  [Cache] Reusing cached dataset for {image_type} images (fold {fold_idx})")
                    
                    # DDP 동기화는 필요하지 않음
                    # 각 프로세스가 독립적으로 데이터셋을 생성하고 사용하므로
                    # barrier()는 제거 (경고 방지)
                    
                    # 캐시된 데이터셋 사용
                    train_dataset, val_dataset, test_dataset = dataset_cache[cache_key]
                    
                    # DataLoader 생성 (캐시된 데이터셋 사용)
                    if is_main_process(rank):
                        print(f"  Creating DataLoaders...")
                    try:
                        train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = get_smc_data_loaders(
                            data_base_dir=data_path,
                            image_type=image_type,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            image_size=image_size,
                            distributed=distributed,
                            world_size=world_size,
                            rank=rank,
                            seed=seed,
                            use_5fold=use_5fold,
                            fold_idx=fold_idx,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            train_dataset=train_dataset,  # 캐시된 데이터셋 전달
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                        )
                        if is_main_process(rank):
                            print(f"  DataLoaders created successfully")
                    except Exception as e:
                        if is_main_process(rank):
                            print(f"  ERROR creating DataLoaders: {e}")
                        raise
                    
                    # 모델 생성
                    model = get_model(model_name, num_classes=4, pretrained=True, image_size=image_size)
                    model = model.to(device)
                    
                    # DDP 래핑
                    if distributed:
                        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                    
                    # Focal Loss용 클래스 가중치 계산 (Multi-label 지원)
                    if is_main_process(rank):
                        print(f"  Computing class weights for Multi-label Focal Loss...")
                    train_labels = [label for _, label in train_dataset]
                    # Multi-label의 경우 FloatTensor를 numpy array로 변환
                    if isinstance(train_labels[0], torch.Tensor):
                        train_labels = np.vstack([label.numpy() for label in train_labels])
                    else:
                        train_labels = np.array(train_labels)
                    alpha = compute_focal_loss_alpha(train_labels, num_classes=4)
                    if is_main_process(rank):
                        print(f"  Class weights (alpha): {alpha}")
                    
                    # 손실 함수: Multi-label Focal Loss 사용
                    criterion = MultiLabelFocalLoss(alpha=alpha, gamma=2.0).to(device)
                    
                    # 옵티마이저
                    if optimizer_name.lower() == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    elif optimizer_name.lower() == 'sgd':
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                    else:
                        raise ValueError(f"Unknown optimizer: {optimizer_name}")
                    
                    # 학습 루프
                    best_val_acc = 0.0
                    best_model_state = None
                    epoch_results = []
                    
                    for epoch in range(1, epochs + 1):
                        # Sampler 설정 (DDP)
                        if train_sampler is not None:
                            train_sampler.set_epoch(epoch)
                        
                        # 학습
                        train_loss = train_epoch(
                            model.module if distributed else model,
                            train_loader,
                            criterion,
                            optimizer,
                            device,
                            rank
                        )
                        
                        # 검증
                        val_metrics = validate(
                            model.module if distributed else model,
                            val_loader,
                            criterion,
                            device,
                            rank
                        )
                        
                        # 메인 프로세스에서만 출력 및 저장
                        if is_main_process(rank):
                            print(f"Epoch {epoch}/{epochs} - "
                                  f"Train Loss: {train_loss:.4f}, "
                                  f"Val Loss: {val_metrics['loss']:.4f}, "
                                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                                  f"Val F1: {val_metrics['f1_macro']:.4f}")
                            
                            # 에포크별 결과 저장
                            epoch_result = create_result_dict(
                                model_name=model_name,
                                image_type=image_type,
                                epoch=epoch,
                                train_loss=train_loss,
                                val_loss=val_metrics['loss'],
                                val_accuracy=val_metrics['accuracy'],
                                val_f1_macro=val_metrics['f1_macro'],
                                val_f1_weighted=val_metrics['f1_weighted'],
                                val_precision=val_metrics['precision'],
                                val_recall=val_metrics['recall'],
                            )
                            if use_5fold:
                                epoch_result['fold'] = fold_idx
                            epoch_results.append(epoch_result)
                            all_detailed_results.append(epoch_result)
                            
                            # Best model 저장
                            if val_metrics['accuracy'] > best_val_acc:
                                best_val_acc = val_metrics['accuracy']
                                best_model_state = {
                                    'epoch': epoch,
                                    'model_state_dict': model.module.state_dict() if distributed else model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'val_accuracy': val_metrics['accuracy'],
                                    'val_f1_macro': val_metrics['f1_macro'],
                                }
                                # Checkpoint 디스크에 저장
                                checkpoint_name = f"{model_name}_{image_type}"
                                if use_5fold:
                                    checkpoint_name += f"_fold{fold_idx}"
                                checkpoint_path = os.path.join(results_dir, 'checkpoints', f"{checkpoint_name}_best.pth")
                                torch.save(best_model_state, checkpoint_path)
                                print(f"  Saved checkpoint: {checkpoint_path} (Val Acc: {val_metrics['accuracy']:.4f})")
                    
                    # Best model 로드 및 테스트
                    if is_main_process(rank) and best_model_state is not None:
                        # 모델 로드
                        if distributed:
                            model.module.load_state_dict(best_model_state['model_state_dict'])
                        else:
                            model.load_state_dict(best_model_state['model_state_dict'])
                        
                        # 테스트
                        test_metrics = test(
                            model.module if distributed else model,
                            test_loader,
                            criterion,
                            device,
                            rank
                        )
                        
                        print(f"\nTest Results ({image_type}):")
                        print(f"  Accuracy (Subset): {test_metrics['accuracy']:.4f}")
                        print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
                        print(f"  F1-Micro: {test_metrics.get('f1_micro', 0):.4f}")
                        print(f"  F1-Weighted: {test_metrics['f1_weighted']:.4f}")
                        print(f"  Precision: {test_metrics['precision']:.4f}")
                        print(f"  Recall: {test_metrics['recall']:.4f}")
                        print(f"  Hamming Loss: {test_metrics.get('hamming_loss', 0):.4f}")
                        print(f"  Jaccard Score: {test_metrics.get('jaccard_score', 0):.4f}")
                        
                        # 결과 저장
                        result = create_result_dict(
                            model_name=model_name,
                            image_type=image_type,
                            test_accuracy=test_metrics['accuracy'],
                            test_f1_macro=test_metrics['f1_macro'],
                            test_f1_weighted=test_metrics['f1_weighted'],
                            test_precision=test_metrics['precision'],
                            test_recall=test_metrics['recall'],
                        )
                        # 5-fold인 경우 fold 인덱스 추가
                        if use_5fold:
                            result['fold'] = fold_idx
                        
                        all_results.append(result)
                        model_results[image_type] = test_metrics
                        
                        # 5-fold 결과 집계용 저장
                        if use_5fold:
                            key = f"{model_name}_{image_type}"
                            if key not in fold_results:
                                fold_results[key] = {
                                    'accuracy': [],
                                    'f1_macro': [],
                                    'f1_weighted': [],
                                    'precision': [],
                                    'recall': [],
                                    'confusion_matrix': [],  # Confusion matrix도 저장
                                }
                            fold_results[key]['accuracy'].append(test_metrics['accuracy'])
                            fold_results[key]['f1_macro'].append(test_metrics['f1_macro'])
                            fold_results[key]['f1_weighted'].append(test_metrics['f1_weighted'])
                            fold_results[key]['precision'].append(test_metrics['precision'])
                            fold_results[key]['recall'].append(test_metrics['recall'])
                            fold_results[key]['confusion_matrix'].append(test_metrics['confusion_matrix'])
                        
                        # Confusion Matrix 저장 (5-fold인 경우는 평균만 저장, 일반 모드는 즉시 저장)
                        if not use_5fold:
                            # 일반 모드: 즉시 저장
                            cm_path = os.path.join(
                                results_dir,
                                'visualizations',
                                f"{model_name}_{image_type}_confusion.png"
                            )
                            plot_confusion_matrix(
                                test_metrics['confusion_matrix'],
                                ['BN', 'CN', 'EF', 'NI'],
                                cm_path,
                                title=f'{model_name.upper()} - {image_type.upper()}: Confusion Matrix'
                            )
                        # 5-fold 모드는 모든 fold가 끝난 후 평균만 저장 (아래에서 처리)
                
                # 성능 증감 계산 (메인 프로세스에서만, 일반 모드에서만)
                if not use_5fold and is_main_process(rank) and 'original' in model_results and 'processed' in model_results:
                    improvement = calculate_improvement(
                        model_results['original'],
                        model_results['processed']
                    )
                    
                    # Improvement를 결과에 추가
                    for result in all_results:
                        if result['model'] == model_name and result['image_type'] == 'processed':
                            result['improvement_accuracy'] = improvement['improvement_accuracy']
                            result['improvement_f1'] = improvement['improvement_f1']
    
    # 결과 저장 (메인 프로세스에서만)
    if is_main_process(rank):
        if use_5fold:
            # 5-fold 모드: 모든 fold 결과와 평균 결과 분리 저장
            
            # 1. 모든 5-fold 결과 저장 (각 fold별 상세 결과)
            results_5fold_all_csv = os.path.join(results_dir, 'results_5fold_all.csv')
            save_results_to_csv(all_results, results_5fold_all_csv, mode='overwrite')
            
            # 2. 에포크별 상세 결과 저장
            detailed_csv = os.path.join(results_dir, 'results_detailed.csv')
            save_results_to_csv(all_detailed_results, detailed_csv, mode='overwrite')
            
            # 3. 5-fold 평균 결과 계산 및 저장
            if fold_results:
                summary_results = []
                for key, metrics in fold_results.items():
                    model_name, image_type = key.split('_', 1)
                    summary_results.append({
                        'model': model_name,
                        'image_type': image_type,
                        'test_accuracy_mean': np.mean(metrics['accuracy']),
                        'test_accuracy_std': np.std(metrics['accuracy']),
                        'test_f1_macro_mean': np.mean(metrics['f1_macro']),
                        'test_f1_macro_std': np.std(metrics['f1_macro']),
                        'test_f1_weighted_mean': np.mean(metrics['f1_weighted']),
                        'test_f1_weighted_std': np.std(metrics['f1_weighted']),
                        'test_precision_mean': np.mean(metrics['precision']),
                        'test_precision_std': np.std(metrics['precision']),
                        'test_recall_mean': np.mean(metrics['recall']),
                        'test_recall_std': np.std(metrics['recall']),
                    })
                
                # Improvement 계산
                for summary in summary_results:
                    if summary['image_type'] == 'processed':
                        # 같은 모델의 original 찾기
                        original_summary = next(
                            (s for s in summary_results 
                             if s['model'] == summary['model'] and s['image_type'] == 'original'),
                            None
                        )
                        if original_summary:
                            if original_summary['test_accuracy_mean'] > 0:
                                summary['improvement_accuracy'] = (
                                    (summary['test_accuracy_mean'] - original_summary['test_accuracy_mean']) 
                                    / original_summary['test_accuracy_mean'] * 100
                                )
                            if original_summary['test_f1_macro_mean'] > 0:
                                summary['improvement_f1'] = (
                                    (summary['test_f1_macro_mean'] - original_summary['test_f1_macro_mean']) 
                                    / original_summary['test_f1_macro_mean'] * 100
                                )
                
                # 5-fold 평균 결과 저장 (평균값만)
                summary_csv = os.path.join(results_dir, 'results_5fold_mean.csv')
                save_results_to_csv(summary_results, summary_csv, mode='overwrite')
                
                print(f"\n{'='*60}")
                print(f"5-Fold Cross-Validation Summary")
                print(f"{'='*60}")
                summary_df = pd.DataFrame(summary_results)
                print(summary_df.to_string(index=False))
                
                # 평균 Confusion Matrix 생성 및 저장
                for key, metrics in fold_results.items():
                    if 'confusion_matrix' in metrics and len(metrics['confusion_matrix']) > 0:
                        model_name, image_type = key.split('_', 1)
                        # 모든 fold의 confusion matrix를 numpy array로 변환 후 평균 계산
                        cm_list = metrics['confusion_matrix']
                        # 각 confusion matrix를 numpy array로 변환하고 shape 확인
                        cm_arrays = []
                        for cm in cm_list:
                            if isinstance(cm, np.ndarray):
                                cm_arrays.append(cm)
                            else:
                                cm_arrays.append(np.array(cm))
                        
                        # 모든 confusion matrix가 같은 shape를 가지는지 확인
                        if len(cm_arrays) > 0:
                            first_shape = cm_arrays[0].shape
                            if all(cm.shape == first_shape for cm in cm_arrays):
                                # 모든 confusion matrix를 stack하고 평균 계산
                                cm_stack = np.stack(cm_arrays, axis=0)
                                avg_cm = np.mean(cm_stack, axis=0).astype(int)
                            else:
                                # Shape가 다른 경우 첫 번째 것을 사용하고 경고
                                if is_main_process(rank):
                                    print(f"Warning: Confusion matrices have different shapes for {key}, using first one")
                                avg_cm = cm_arrays[0].astype(int)
                        else:
                            continue
                        
                        cm_path = os.path.join(
                            results_dir,
                            'visualizations',
                            f"{model_name}_{image_type}_confusion_mean.png"
                        )
                        plot_confusion_matrix(
                            avg_cm,
                            ['BN', 'CN', 'EF', 'NI'],
                            cm_path,
                            title=f'{model_name.upper()} - {image_type.upper()}: Average Confusion Matrix (5-Fold CV)'
                        )
                
                # 시각화 생성 (평균값 기준)
                vis_df = summary_df.copy()
                vis_df['test_accuracy'] = vis_df['test_accuracy_mean']
                vis_df['test_f1_macro'] = vis_df['test_f1_macro_mean']
                vis_df['test_f1_weighted'] = vis_df['test_f1_weighted_mean']
                vis_df['test_precision'] = vis_df['test_precision_mean']
                vis_df['test_recall'] = vis_df['test_recall_mean']
                create_all_visualizations(results_dir, vis_df, None)
        else:
            # 일반 모드: 기존 방식대로 저장
            results_csv = os.path.join(results_dir, 'results.csv')
            save_results_to_csv(all_results, results_csv, mode='overwrite')
            
            detailed_csv = os.path.join(results_dir, 'results_detailed.csv')
            save_results_to_csv(all_detailed_results, detailed_csv, mode='overwrite')
            
            # 시각화 생성
            results_df = pd.DataFrame(all_results)
            detailed_df = pd.DataFrame(all_detailed_results) if all_detailed_results else None
            create_all_visualizations(results_dir, results_df, detailed_df)
        
        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Results saved in: {results_dir}")
        if use_5fold:
            print(f"  - All fold results: results_5fold_all.csv")
            print(f"  - Mean results: results_5fold_mean.csv")
            print(f"  - Detailed (epochs): results_detailed.csv")
        print(f"{'='*60}")
    
    # DDP 정리
    if distributed:
        cleanup_distributed()
    
    return results_dir

