#!/usr/bin/env python3
"""
SMC Classification Training Script
원본 이미지와 전처리 이미지를 비교하는 classification 실험

Usage:
    # 단일 GPU
    python SMC/train_classification.py --data_path SMC/data --epochs 50 --batch_size 32
    
    # DDP (Multi-GPU)
    torchrun --nproc_per_node=4 SMC/train_classification.py --data_path SMC/data --epochs 50 --batch_size 32
"""

import os
import argparse
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SMC.utils.experiment_runner import run_classification_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description='SMC Classification Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 데이터 경로
    parser.add_argument(
        '--data_path',
        type=str,
        default='SMC/data',
        help='Path to SMC/data directory'
    )
    
    # 모델 선택
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        default=['vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'densenet'],
        choices=['vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'densenet'],
        help='Models to train'
    )
    
    # 하이퍼파라미터
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per GPU'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='Optimizer (adam or sgd)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='Image resize size'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=24,
        help='Random seed'
    )
    
    # 결과 저장
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Results directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--use_5fold',
        action='store_true',
        default=False,
        help='Use 5-fold cross-validation'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 데이터 경로 확인 (상대 경로인 경우 현재 스크립트 위치 기준으로 변환)
    if not os.path.isabs(args.data_path):
        # 상대 경로인 경우 스크립트 위치 기준으로 변환
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, args.data_path)
    else:
        data_path = args.data_path
    
    if not os.path.exists(data_path):
        # 원본 경로와 변환된 경로 모두 확인
        if not os.path.exists(args.data_path):
            raise ValueError(
                f"Data path does not exist: {args.data_path}\n"
                f"Also tried: {data_path if not os.path.isabs(args.data_path) else 'N/A'}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Script directory: {os.path.dirname(os.path.abspath(__file__))}"
            )
        else:
            data_path = args.data_path
    
    # 결과 디렉토리 확인
    if args.results_dir and not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*60)
    print("SMC Classification Experiment")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Image size: {args.image_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Seed: {args.seed}")
    print(f"Results dir: {args.results_dir or 'auto-generated'}")
    print(f"5-Fold CV: {'Enabled' if args.use_5fold else 'Disabled'}")
    print("="*60)
    
    # 실험 실행
    try:
        results_dir = run_classification_experiment(
            data_path=data_path,
            models=args.models,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            image_size=args.image_size,
            num_workers=args.num_workers,
            seed=args.seed,
            results_dir=args.results_dir,
            use_5fold=args.use_5fold,
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved in: {results_dir}")
        
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

