import os
import imageio.v3 as iio
import torch
import tyro
from einops import repeat
from seva.eval import transform_img_and_K, get_value_dict, save_output, seed_everything, decode_output
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.utils import load_model
from seva.sampling import (
    MultiviewCFG
)
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seva.sampling import DiscreteDenoiser, append_dims, to_d
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler
logger = get_logger(__name__)
# Constants
WORK_DIR = "work_dirs/demo_cli"

T = 21

def get_target_c2ws_and_Ks_from_preset(
    preprocessed: dict, preset_traj: str, num_frames: int, zoom_factor: float
):
    img_wh = preprocessed["input_wh"]
    start_c2w = preprocessed["input_c2ws"][0]
    start_w2c = torch.linalg.inv(start_c2w)
    look_at = torch.tensor([0, 0, 10])
    start_fov = DEFAULT_FOV_RAD

    target_c2ws, target_fovs = get_preset_pose_fov(
        preset_traj,
        num_frames,
        start_w2c,
        look_at,
        -start_c2w[:3, 1],
        start_fov,
        spiral_radii=[1.0, 1.0, 0.5],
        zoom_factor=zoom_factor,
    )
    target_c2ws = torch.as_tensor(target_c2ws)
    target_fovs = torch.as_tensor(target_fovs)
    target_Ks = get_default_intrinsics(
        target_fovs, aspect_ratio=img_wh[0] / img_wh[1]
    )
    return target_c2ws, target_Ks

class Image_Camera(Dataset):
    def __init__(self, root_dir, preset_traj, num_frames, zoom_factor, camera_scale):
        super().__init__()
        self.root_dir = root_dir
        self.img_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.preset_traj = preset_traj
        self.num_frames = num_frames
        self.zoom_factor = zoom_factor
        self.camera_scale = camera_scale
        self.img_files.sort()
    def preprocess(self, input_img_path: str, preset_traj, num_frames, zoom_factor, camera_scale):
        shorter = 576  # must divided by 64

        input_imgs = torch.as_tensor(
            iio.imread(input_img_path) / 255.0, dtype=torch.float32
        )[None, ..., :3]
        input_imgs = transform_img_and_K(
            input_imgs.permute(0, 3, 1, 2), shorter, K=None, size_stride=64
        )[0].permute(0, 2, 3, 1)

        input_Ks = get_default_intrinsics(
            aspect_ratio=input_imgs.shape[2] / input_imgs.shape[1]
        )
        input_c2ws = torch.eye(4)[None].to(dtype=torch.float32)

        preprocessed = {
            "input_imgs": input_imgs,
            "input_Ks": input_Ks,
            "input_c2ws": input_c2ws,
            "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
        }
        W, H = preprocessed["input_wh"]
        
        target_c2ws, target_Ks = get_target_c2ws_and_Ks_from_preset(
            preprocessed, preset_traj, num_frames, zoom_factor
        )
        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1])[:, None]

        traj_prior_c2ws = all_c2ws[1:]

        # --- Normalize input images only ---
        input_imgs = input_imgs[0].permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        input_imgs = input_imgs * 2.0 - 1.0  # already in [0,1], scale to [-1,1]

        # --- Get corresponding poses ---
        input_Ks = all_Ks[[0]]

        curr_input_sels = [0]  # input always at index 0
        curr_imgs = torch.cat(
            [input_imgs, input_imgs.new_zeros(len(traj_prior_c2ws), *input_imgs.shape[1:])],
            dim=0,
        )
        curr_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)
        curr_Ks = torch.cat([input_Ks, all_Ks[1:]], dim=0)

        # --- Build value_dict ---
        value_dict = get_value_dict(
            curr_imgs.to("cuda"),
            curr_input_sels,
            curr_c2ws,
            curr_Ks,
            list(range(T)),
            all_c2ws=all_c2ws,
            camera_scale=camera_scale,
        )
        return value_dict
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        sample = self.preprocess(img_path, self.preset_traj, self.num_frames, self.zoom_factor, self.camera_scale)
        return sample

import os
import torch

SAVE_FEATURE_DIR = "cached_features"

def main(
    eval_imgs: str = "./sana_gen/data",
    seed: int = 36,
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
    output_dir: str = 'cfg',
    cfg_min = 1.2,
    num_steps_eval=4,
):
    H, W = 576, 576
    torch.cuda.empty_cache()
    device = 'cuda'
    seed_everything(seed)

    # Dataset
    test_dataset = Image_Camera(root_dir=eval_imgs, preset_traj=preset_traj,
                                num_frames=num_frames, zoom_factor=zoom_factor,
                                camera_scale=camera_scale)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model, AE, Conditioner
    ae = AutoEncoder(chunk_size=1).to(device).eval()
    conditioner = CLIPConditioner().to(device).eval()

    os.makedirs(SAVE_FEATURE_DIR, exist_ok=True)

    for batch_idx, value_dict in enumerate(testloader):
        img_name = os.path.splitext(os.path.basename(test_dataset.img_files[batch_idx]))[0]

        imgs = value_dict["cond_frames"][0].to(device)
        input_masks = value_dict["cond_frames_mask"][0].to(device)
        pluckers = value_dict["plucker_coordinate"][0].to(device)

        feature_path = os.path.join(SAVE_FEATURE_DIR, f"{img_name}.pt")
        if os.path.exists(feature_path):
            print(f"✅ Skipping {img_name}, cached features already exist.")
            continue

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = ae.encode(imgs[input_masks], 1).cpu()          # move to CPU to save VRAM
            c_crossattn = conditioner(imgs[input_masks]).mean(0).cpu()

        # Save to disk
        torch.save({
            "encoded": encoded,
            "c_crossattn": c_crossattn,
        }, feature_path)

        print(f"💾 Saved cached features for {img_name} → {feature_path}")


                    
if __name__ == "__main__":
    tyro.cli(main)

