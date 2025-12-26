"""
Distributed Training Utilities
"""

import os
import torch
from datetime import timedelta


def setup_distributed():
    """DDP 설정
    
    Returns:
        (distributed, rank, local_rank, world_size)
    """
    import torch.distributed as dist
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # NCCL timeout 설정 (기본 10분 -> 30분으로 증가)
        timeout = os.environ.get('NCCL_TIMEOUT', '1800')  # 30분 (초 단위)
        # device를 먼저 설정
        torch.cuda.set_device(local_rank)
        # init_process_group에 device_id 명시 (경고 방지)
        # device_id는 정수여야 함 (local_rank 값)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=int(timeout)),
            device_id=local_rank  # device_id 명시 (정수)
        )
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_distributed():
    """DDP 정리"""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int):
    """메인 프로세스 여부 확인"""
    return rank == 0

