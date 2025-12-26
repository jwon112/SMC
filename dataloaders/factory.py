"""
SMC DataLoader Factory
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

from .smc_dataset import SMCImageDataset, get_default_transforms


def get_smc_data_loaders(
    data_base_dir,
    image_type='original',
    batch_size=32,
    num_workers=4,
    image_size=224,
    distributed=False,
    world_size=None,
    rank=None,
    seed=None,
    use_5fold=False,
    fold_idx=None,
    train_df=None,
    val_df=None,
    test_df=None,
    train_dataset=None,  # 캐시된 데이터셋 (선택)
    val_dataset=None,
    test_dataset=None,
):
    """SMC 데이터 로더 생성
    
    Args:
        data_base_dir: SMC/data 디렉토리 경로
        image_type: 'original' 또는 'processed'
        batch_size: 배치 크기
        num_workers: DataLoader 워커 수
        image_size: 이미지 리사이즈 크기
        distributed: DDP 사용 여부
        world_size: DDP world size
        rank: DDP rank
        seed: 랜덤 시드
        use_5fold: 5-fold CV 사용 여부
        fold_idx: fold 인덱스 (0-4, use_5fold=True일 때만 사용)
        train_df: 5-fold용 train DataFrame (use_5fold=True일 때)
        val_df: 5-fold용 val DataFrame (use_5fold=True일 때)
        test_df: 5-fold용 test DataFrame (use_5fold=True일 때)
    """
    
    # 캐시된 데이터셋이 있으면 재사용, 없으면 새로 생성
    if train_dataset is None or val_dataset is None or test_dataset is None:
        # 데이터 디렉토리
        train_dir = os.path.join(data_base_dir, 'train')
        val_dir = os.path.join(data_base_dir, 'val')
        test_dir = os.path.join(data_base_dir, 'test')
        
        # 5-fold 모드인 경우
        if use_5fold and train_df is not None and val_df is not None and test_df is not None:
            # 5-fold용 DataFrame 사용
            train_dataset = SMCImageDataset(
                data_dir=train_dir,
                csv_file=train_df,
                image_type=image_type,
                transform=get_default_transforms('train', image_size)
            )
            
            val_dataset = SMCImageDataset(
                data_dir=val_dir,
                csv_file=val_df,
                image_type=image_type,
                transform=get_default_transforms('val', image_size)
            )
            
            test_dataset = SMCImageDataset(
                data_dir=test_dir,
                csv_file=test_df,
                image_type=image_type,
                transform=get_default_transforms('test', image_size)
            )
        else:
            # 일반 모드: 기존 CSV 파일 사용
            train_dataset = SMCImageDataset(
                data_dir=train_dir,
                csv_file='train.csv',
                image_type=image_type,
                transform=get_default_transforms('train', image_size)
            )
            
            val_dataset = SMCImageDataset(
                data_dir=val_dir,
                csv_file='val.csv',
                image_type=image_type,
                transform=get_default_transforms('val', image_size)
            )
            
            test_dataset = SMCImageDataset(
                data_dir=test_dir,
                csv_file='test.csv',
                image_type=image_type,
                transform=get_default_transforms('test', image_size)
            )
    # 캐시된 데이터셋이 있으면 그대로 사용 (train_dir 등은 필요 없음)
    
    # DistributedSampler 설정
    train_sampler = val_sampler = test_sampler = None
    if distributed and world_size is not None and rank is not None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler

