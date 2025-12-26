"""
SMC Image Dataset for Classification
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SMCImageDataset(Dataset):
    """SMC 이미지 분류 데이터셋
    
    Args:
        data_dir: 데이터 디렉토리 경로 (train/val/test 중 하나)
        csv_file: CSV 파일 경로 (예: train.csv) 또는 DataFrame
        image_type: 'original' 또는 'processed' (전처리 이미지 사용 여부)
        transform: 이미지 변환 (augmentation 등)
        image_list: 사용할 이미지 리스트 (5-fold용, None이면 CSV 전체 사용)
        cache_images: 이미지를 메모리에 캐싱할지 여부 (기본 False, 메모리 부족 시 False 권장)
    """
    
    def __init__(self, data_dir, csv_file, image_type='original', transform=None, image_list=None, cache_images=False):
        self.data_dir = data_dir
        self.image_type = image_type
        self.transform = transform
        
        # CSV 파일 로드 또는 DataFrame 직접 사용
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file.copy()
        else:
            csv_path = os.path.join(data_dir, csv_file)
            self.df = pd.read_csv(csv_path)
        
        # 5-fold용 이미지 필터링
        if image_list is not None:
            self.df = self.df[self.df['image'].isin(image_list)].copy()
        
        # 라벨 컬럼
        self.label_columns = ['BN', 'CN', 'EF', 'NI']
        
        # 이미지 파일명, 라벨, 이미지 데이터 추출
        self.image_files = []
        self.labels = []
        self.images = []  # 이미지 데이터 캐싱용
        
        print(f"Loading images into memory from {data_dir} (type: {image_type})...")
        loaded_count = 0  # 실제로 로드된 이미지 개수
        total_rows = len(self.df)
        
        for idx, row in self.df.iterrows():
            image_name = row['image']
            
            # 원본 또는 전처리 이미지 선택
            if image_type == 'processed':
                # 전처리 이미지 파일명 생성 (예: tile_0_12288_13.jpg -> tile_0_12288_13_thr210_r130.jpg)
                base_name = image_name.replace('.jpg', '')
                # train/val/test 디렉토리 모두 확인
                for subdir in ['train', 'val', 'test']:
                    subdir_path = os.path.join(os.path.dirname(data_dir), subdir)
                    image_path = os.path.join(subdir_path, f"{base_name}_thr210_r130.jpg")
                    if os.path.exists(image_path):
                        break
                else:
                    # 모든 디렉토리에서 찾지 못한 경우 data_dir에서 시도
                    image_path = os.path.join(data_dir, f"{base_name}_thr210_r130.jpg")
            else:
                # 원본 이미지 - train/val/test 디렉토리 모두 확인
                for subdir in ['train', 'val', 'test']:
                    subdir_path = os.path.join(os.path.dirname(data_dir), subdir)
                    image_path = os.path.join(subdir_path, image_name)
                    if os.path.exists(image_path):
                        break
                else:
                    # 모든 디렉토리에서 찾지 못한 경우 data_dir에서 시도
                    image_path = os.path.join(data_dir, image_name)
            
            # 이미지 파일 존재 확인
            if os.path.exists(image_path):
                self.image_files.append(image_path)
                
                # 라벨 추출 (multi-label 지원)
                label_vec = [int(row[col]) for col in self.label_columns]
                # Multi-label: FloatTensor로 저장 (예: [1, 0, 1, 0])
                self.labels.append(torch.FloatTensor(label_vec))
                
                # 이미지 데이터를 메모리에 로드 (캐싱)
                try:
                    image = Image.open(image_path).convert('RGB')
                    self.images.append(image)  # PIL Image 객체를 메모리에 저장
                    loaded_count += 1
                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}")
                    # 실패한 경우 None 저장 (나중에 에러 처리)
                    self.images.append(None)
            
            # 진행 상황 출력 (1000개마다, 실제 로드된 개수 기준)
            if loaded_count > 0 and loaded_count % 1000 == 0:
                print(f"  Loaded {loaded_count}/{total_rows} images...")
        
        print(f"Loaded {len(self.image_files)} images into memory from {data_dir} (type: {image_type})")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 캐시된 이미지 사용 (디스크 I/O 없음)
        image = self.images[idx]
        
        # 이미지 로드 실패한 경우 재시도
        if image is None:
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert('RGB')
            self.images[idx] = image  # 캐시 업데이트
        
        # 변환 적용 (augmentation 등)
        if self.transform:
            image = self.transform(image)
        
        # 라벨
        label = self.labels[idx]
        
        return image, label


def get_default_transforms(image_type='train', image_size=224):
    """기본 이미지 변환 반환
    
    Args:
        image_type: 'train' 또는 'val'/'test'
        image_size: 리사이즈 크기
    """
    if image_type == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

