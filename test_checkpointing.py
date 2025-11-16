import torch.nn.functional as F
import os
import gc
import random
import imageio.v3 as iio
import torch
import tyro
from einops import repeat
from seva.eval import transform_img_and_K, get_value_dict, save_output, seed_everything, decode_output
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.utils import load_model
from seva.sampling import (
    MultiviewCFG
)
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seva.sampling import DiscreteDenoiser, append_dims, to_d
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import LoraConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler
logger = get_logger(__name__)
# Constants

T = 21

device = 'cuda'

# Model, ae, condition
teacher = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
teacher.module.to(dtype=torch.bfloat16)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_v"],  # layers to inject LoRA
    bias="none"
)
teacher.module.add_adapter(config)
print(teacher.module._supports_gradient_checkpointing)

teacher.module.enable_gradient_checkpointing()
