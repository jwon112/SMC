# SMC Classification Experiment

심장 병리 이미지 분류 실험 프로젝트: 원본 이미지와 전처리 이미지의 성능 비교

## 프로젝트 개요

이 프로젝트는 심장 병리 이미지에서 원본 이미지와 전처리된 이미지(배경의 흰색 영역 감소)의 분류 성능을 비교하는 실험을 수행합니다.

### 주요 특징

- **Multi-label Classification**: 한 이미지에 여러 클래스가 동시에 존재할 수 있는 경우 지원
- **5-Fold Cross-Validation**: 신뢰성 있는 성능 평가
- **Distributed Data Parallel (DDP)**: Multi-GPU 학습 지원
- **클래스 불균형 처리**: Multi-label Focal Loss 사용
- **다양한 모델 지원**: VGG16, ResNet50, MobileNet, EfficientNet, DenseNet

## 프로젝트 구조

```
SMC/
├── data/                    # 데이터 디렉토리 (이미지 파일은 .gitignore로 제외)
│   ├── train/
│   ├── val/
│   └── test/
├── dataloaders/             # 데이터 로더
│   ├── smc_dataset.py
│   └── factory.py
├── models/                  # 모델 정의
│   └── __init__.py
├── utils/                   # 유틸리티 함수
│   ├── distributed.py      # DDP 설정
│   ├── training.py         # 학습/평가 함수
│   ├── losses.py           # Loss functions (Focal Loss)
│   ├── evaluation.py       # 평가 지표
│   ├── results.py          # 결과 저장
│   ├── visualization.py    # 시각화
│   └── split_utils.py      # 5-fold split
├── train_classification.py  # 메인 학습 스크립트
├── analyze_multilabel.py   # Multi-label 데이터 분석
└── README.md
```

## 데이터 구조

```
data/
├── train/
│   ├── train.csv          # 학습 데이터 라벨
│   ├── *.jpg              # 원본 이미지
│   └── *.json             # LabelMe 주석 파일
├── val/
│   ├── val.csv
│   ├── *.jpg
│   └── *.json
└── test/
    ├── test.csv
    ├── *.jpg
    └── *.json
```

### 라벨 형식

CSV 파일은 다음 형식입니다:
```csv
image,BN,CN,EF,NI
tile_0_12288_11.jpg,1,0,0,0
tile_10240_10240_0.jpg,0,0,1,0
tile_16384_22528_9.jpg,1,0,1,0  # Multi-label 예시
```

- **BN, CN, EF, NI**: 4개 클래스
- **Multi-label 지원**: 한 이미지에 여러 클래스가 동시에 존재 가능 (예: `[1, 0, 1, 0]`)

### 전처리 이미지

전처리된 이미지는 `_thr210_r130` suffix를 가집니다:
- 원본: `tile_0_12288_13.jpg`
- 전처리: `tile_0_12288_13_thr210_r130.jpg`

## 설치

### 요구사항

```bash
pip install torch torchvision
pip install pandas numpy
pip install scikit-learn
pip install matplotlib seaborn
pip install pillow
```

또는 `requirements.txt` 사용:
```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실행 (Single GPU)

```bash
python SMC/train_classification.py --data_path SMC/data --epochs 50 --batch_size 32
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 SMC/train_classification.py \
    --data_path SMC/data \
    --epochs 50 \
    --batch_size 32
```

### 5-Fold Cross-Validation

```bash
torchrun --nproc_per_node=7 SMC/train_classification.py \
    --data_path SMC/data \
    --epochs 50 \
    --batch_size 128 \
    --use_5fold
```

### 주요 인자

- `--data_path`: 데이터 디렉토리 경로 (기본값: `SMC/data`)
- `--models`: 학습할 모델 리스트 (기본값: 모든 모델)
- `--epochs`: 학습 에포크 수 (기본값: 50)
- `--batch_size`: GPU당 배치 크기 (기본값: 32)
- `--lr`: 학습률 (기본값: 0.001)
- `--optimizer`: 옵티마이저 (`adam` 또는 `sgd`, 기본값: `adam`)
- `--image_size`: 이미지 리사이즈 크기 (기본값: 224)
- `--use_5fold`: 5-fold cross-validation 활성화

## 결과

실험 결과는 `SMC/results/experiment_YYYYMMDD_HHMMSS/` 디렉토리에 저장됩니다:

```
results/
├── results_5fold_all.csv          # 모든 fold 결과 (5-fold 사용 시)
├── results_5fold_mean.csv         # 5-fold 평균 결과
├── results_detailed.csv            # 에포크별 상세 결과
├── checkpoints/                    # Best model checkpoints
│   ├── vgg16_original_best.pth
│   ├── vgg16_processed_best.pth
│   └── ...
└── visualizations/                # 시각화 결과
    ├── performance_comparison.png
    ├── absolute_comparison_accuracy.png
    ├── absolute_comparison_f1_score.png
    ├── improvement_percentage.png
    ├── confusion_matrix_*.png
    └── ...
```

## 모델 및 Loss Function

### 지원 모델

- VGG16
- ResNet50
- MobileNetV3
- EfficientNet-B0
- DenseNet121

모든 모델은 ImageNet pretrained weights를 사용합니다.

### Loss Function

**Multi-label Focal Loss**를 사용하여 클래스 불균형 문제를 처리합니다:

```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

- `α_t`: 클래스별 가중치 (자동 계산)
- `γ`: focusing parameter (기본값: 2.0)

## 평가 지표

- **Accuracy (Subset)**: 모든 라벨이 정확히 일치해야 맞음
- **F1-Score (Macro/Micro/Weighted)**: 클래스별 F1의 평균
- **Precision/Recall**: 클래스별 평균
- **Hamming Loss**: 평균 오분류 비율
- **Jaccard Score**: 예측과 실제의 교집합/합집합 비율

## 데이터 통계

- **전체 샘플**: 7,972개
- **Single-label**: 7,906개 (99.17%)
- **Multi-label**: 66개 (0.83%)
- **주요 조합**: CN+EF (50%), BN+EF (33.3%), BN+CN (12.1%), BN+NI (4.5%)

## 주의사항

1. **클래스 불균형**: CN(1개), NI(2개) 등 극단적으로 적은 클래스가 있어 성능이 낮을 수 있습니다.
2. **Multi-label 비율**: Multi-label 샘플이 0.83%로 매우 적지만, 정보 보존을 위해 Multi-label 학습을 유지합니다.
3. **메모리 사용**: 이미지 데이터를 RAM에 캐싱하므로 메모리 사용량이 높을 수 있습니다.

## 라이선스

이 프로젝트는 연구 목적으로 사용됩니다.

