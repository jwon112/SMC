"""
5-Fold Cross-Validation Split Utilities
"""

import os
import random
import pandas as pd
from typing import Tuple, List
from torch.utils.data import Subset


def split_smc_dataset_5fold(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str,
    fold_idx: int,
    seed: int = 24,
) -> Tuple[List[str], List[str], List[str]]:
    """SMC 데이터셋을 5-fold로 분할
    
    Args:
        train_csv_path: train.csv 경로
        val_csv_path: val.csv 경로
        test_csv_path: test.csv 경로
        fold_idx: fold 인덱스 (0-4)
        seed: 랜덤 시드
    
    Returns:
        (train_image_names, val_image_names, test_image_names): 각 split의 이미지 파일명 리스트
    """
    if fold_idx < 0 or fold_idx >= 5:
        raise ValueError("fold_idx must be in [0, 4]")
    
    random.seed(seed)
    
    # 모든 CSV 파일 로드
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # 모든 이미지 합치기 (train + val + test)
    all_images = pd.concat([
        train_df[['image']],
        val_df[['image']],
        test_df[['image']]
    ], ignore_index=True)
    
    # 중복 제거
    all_images = all_images.drop_duplicates().reset_index(drop=True)
    
    # Shuffle
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    
    # 5-fold로 분할
    total_len = len(all_images)
    fold_size = total_len // 5
    fold_sizes = [fold_size] * 5
    remainder = total_len % 5
    for i in range(remainder):
        fold_sizes[i] += 1
    
    fold_starts = [0]
    for size in fold_sizes:
        fold_starts.append(fold_starts[-1] + size)
    
    # Test fold
    test_start = fold_starts[fold_idx]
    test_end = fold_starts[fold_idx + 1]
    test_indices = indices[test_start:test_end]
    
    # Val fold (다음 fold)
    val_fold_idx = (fold_idx + 1) % 5
    val_start = fold_starts[val_fold_idx]
    val_end = fold_starts[val_fold_idx + 1]
    val_indices = indices[val_start:val_end]
    
    # Train folds (나머지 3개 fold)
    train_indices = []
    for i in range(5):
        if i in (fold_idx, val_fold_idx):
            continue
        train_start = fold_starts[i]
        train_end = fold_starts[i + 1]
        train_indices.extend(indices[train_start:train_end])
    
    # 이미지 파일명 추출
    train_images = all_images.iloc[train_indices]['image'].tolist()
    val_images = all_images.iloc[val_indices]['image'].tolist()
    test_images = all_images.iloc[test_indices]['image'].tolist()
    
    return train_images, val_images, test_images


def get_fold_dataframes(
    data_base_dir: str,
    fold_idx: int,
    seed: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """5-fold split에 따라 train/val/test DataFrame 생성
    
    Args:
        data_base_dir: SMC/data 디렉토리 경로
        fold_idx: fold 인덱스 (0-4)
        seed: 랜덤 시드
    
    Returns:
        (train_df, val_df, test_df): 각 split의 DataFrame
    """
    train_csv = os.path.join(data_base_dir, 'train', 'train.csv')
    val_csv = os.path.join(data_base_dir, 'val', 'val.csv')
    test_csv = os.path.join(data_base_dir, 'test', 'test.csv')
    
    # 5-fold split
    train_images, val_images, test_images = split_smc_dataset_5fold(
        train_csv, val_csv, test_csv, fold_idx, seed
    )
    
    # 원본 CSV 로드
    all_train_df = pd.read_csv(train_csv)
    all_val_df = pd.read_csv(val_csv)
    all_test_df = pd.read_csv(test_csv)
    
    # 모든 데이터 합치기
    all_df = pd.concat([all_train_df, all_val_df, all_test_df], ignore_index=True)
    
    # Fold별로 필터링
    train_df = all_df[all_df['image'].isin(train_images)].copy()
    val_df = all_df[all_df['image'].isin(val_images)].copy()
    test_df = all_df[all_df['image'].isin(test_images)].copy()
    
    return train_df, val_df, test_df

