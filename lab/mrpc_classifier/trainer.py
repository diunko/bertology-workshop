from typing import Optional, List, Union, Dict
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainerConfig:
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    gpus: Optional[List[int]] = None
    num_tpu_cores: Optional[int] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_pct: float = 0.0
    track_grad_norm: int = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: int = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    test_percent_check: float = 1.0
    val_check_interval: float = 1.0
    log_save_interval: int = 100
    row_log_interval: int = 10
    distributed_backend: Optional[str] = None
    precision: int = 32
    weights_summary: Optional[str] = "full"
    weights_save_path: Optional[str] = None
    amp_level: str = "O1"
    num_sanity_val_steps: int = 5
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    # profiler: Optional[BaseProfiler] = None #TODO: support complex types
    benchmark: bool = False
    reload_dataloaders_every_epoch: bool = False


default_config = OmegaConf.structured(TrainerConfig)
