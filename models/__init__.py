"""
SMC Classification Models
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_vgg16(num_classes=4, pretrained=True, image_size=224):
    """VGG16 모델
    
    Args:
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (224 권장, 다른 크기도 가능)
    """
    if pretrained:
        weights = models.VGG16_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.vgg16(weights=weights)
    # 마지막 분류 레이어 수정
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # VGG16은 입력 크기에 유연함 (adaptive pooling 없음, 하지만 다른 크기도 작동)
    if image_size != 224:
        print(f"Warning: VGG16 is typically used with 224x224. Using {image_size}x{image_size} may affect performance.")
    
    return model


def get_resnet50(num_classes=4, pretrained=True, image_size=224):
    """ResNet50 모델
    
    Args:
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (224 권장, 다른 크기도 가능)
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet50(weights=weights)
    # 마지막 분류 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # ResNet50은 입력 크기에 유연함 (adaptive pooling 사용)
    # 다른 크기도 잘 작동함
    
    return model


def get_mobilenet(num_classes=4, pretrained=True, image_size=224):
    """MobileNetV3 Large 모델
    
    Args:
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (224 권장, 다른 크기도 가능)
    """
    if pretrained:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.mobilenet_v3_large(weights=weights)
    # 마지막 분류 레이어 수정
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    # MobileNetV3는 입력 크기에 유연함
    
    return model


def get_efficientnet(num_classes=4, pretrained=True, image_size=224):
    """EfficientNet-B0 모델
    
    Args:
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (224 권장, 다른 크기도 가능하지만 성능 저하 가능)
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.efficientnet_b0(weights=weights)
    # 마지막 분류 레이어 수정
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # EfficientNet은 224에 최적화되어 있지만 다른 크기도 작동함
    if image_size != 224:
        print(f"Warning: EfficientNet-B0 is optimized for 224x224. Using {image_size}x{image_size} may affect performance.")
    
    return model


def get_densenet(num_classes=4, pretrained=True, image_size=224):
    """DenseNet121 모델
    
    Args:
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (224 권장, 다른 크기도 가능)
    """
    if pretrained:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.densenet121(weights=weights)
    # 마지막 분류 레이어 수정
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # DenseNet121은 입력 크기에 유연함 (adaptive pooling 사용)
    
    return model


def get_model(model_name, num_classes=4, pretrained=True, image_size=224):
    """모델 팩토리 함수
    
    Args:
        model_name: 'vgg16', 'resnet50', 'mobilenet', 'efficientnet', 'densenet'
        num_classes: 분류 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부
        image_size: 입력 이미지 크기 (기본 224, 원본 이미지는 512x512)
    """
    model_dict = {
        'vgg16': get_vgg16,
        'resnet50': get_resnet50,
        'mobilenet': get_mobilenet,
        'efficientnet': get_efficientnet,
        'densenet': get_densenet,
    }
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    return model_dict[model_name.lower()](num_classes=num_classes, pretrained=pretrained, image_size=image_size)


__all__ = [
    'get_vgg16',
    'get_resnet50',
    'get_mobilenet',
    'get_efficientnet',
    'get_densenet',
    'get_model',
]

