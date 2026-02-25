import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )


def cleanup_ddp():
    dist.destroy_process_group()
