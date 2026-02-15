"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist


# Change this to reflect your cluster layout.

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    

    if os.environ.get("LOCAL_RANK") is None: # DDP時は通らない
        print(0)
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
       
        
    os.environ["OMP_NUM_THREADS"] = "1"
    
    print(0)
    
    #world_size = th.cuda.device_count() if th.cuda.is_available() else 1
    dist.init_process_group(backend=backend, init_method="env://")#, world_size=world_size)

    if th.cuda.is_available():  # This clears remaining caches in GPU 0
       th.cuda.set_device(dev())
       th.cuda.empty_cache()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")#cuda:0 (CFG)
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
