"""
SMC 데이터셋 탐색 스크립트
- Polygon 주석 확인
- 원본 이미지와 전처리 이미지 비교
- Classification vs Segmentation 가능성 검토
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_json_annotation(json_path):
    """JSON 주석 파일 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)

def polygon_to_mask(points, width, height):
    """Polygon 좌표를 binary mask로 변환"""
    from PIL import Image, ImageDraw
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(p) for p in points], fill=1)
    return np.array(mask)

def analyze_annotation(json_path):
    """주석 파일 분석"""
    ann = load_json_annotation(json_path)
    
    print(f"\n=== {os.path.basename(json_path)} ===")
    print(f"이미지 크기: {ann['imageWidth']} x {ann['imageHeight']}")
    print(f"이미지 경로: {ann['imagePath']}")
    print(f"Polygon 개수: {len(ann['shapes'])}")
    
    total_mask_area = 0
    for i, shape in enumerate(ann['shapes']):
        label = shape['label']
        points = shape['points']
        mask = polygon_to_mask(points, ann['imageWidth'], ann['imageHeight'])
        mask_area = mask.sum()
        total_mask_area += mask_area
        
        print(f"\n  Polygon {i+1}:")
        print(f"    라벨: {label}")
        print(f"    점 개수: {len(points)}")
        print(f"    Mask 영역 (픽셀): {mask_area}")
        print(f"    전체 대비 비율: {mask_area / (ann['imageWidth'] * ann['imageHeight']) * 100:.2f}%")
    
    print(f"\n  전체 주석 영역: {total_mask_area} 픽셀 ({total_mask_area / (ann['imageWidth'] * ann['imageHeight']) * 100:.2f}%)")
    
    return ann

def check_image_files(base_name, data_dir):
    """원본 이미지와 전처리 이미지 확인"""
    base_path = Path(data_dir) / base_name
    
    files = {
        'original': base_path.with_suffix('.jpg'),
        'processed': base_path.parent / f"{base_path.stem}_thr210_r130.jpg"
    }
    
    print(f"\n=== 이미지 파일 확인 ===")
    for name, path in files.items():
        exists = path.exists()
        print(f"{name}: {path.name} - {'존재' if exists else '없음'}")
        if exists:
            img = Image.open(path)
            print(f"  크기: {img.size}")
            print(f"  모드: {img.mode}")
    
    return files

if __name__ == "__main__":
    data_dir = "SMC/data/test"
    
    # 샘플 파일들 확인
    sample_files = [
        "tile_0_12288_13.json",
        "tile_0_12288_11.json",
        "tile_10240_10240_0.json",  # EF 라벨
    ]
    
    for json_file in sample_files:
        json_path = os.path.join(data_dir, json_file)
        if os.path.exists(json_path):
            ann = analyze_annotation(json_path)
            
            # 이미지 파일 확인
            base_name = ann['imagePath'].replace('.jpg', '')
            check_image_files(base_name, data_dir)
            
            print("\n" + "="*60)

