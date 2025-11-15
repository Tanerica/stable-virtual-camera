import os.path as osp
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
import random
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seva.sampling import DiscreteDenoiser, append_dims, to_d
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# Constants
WORK_DIR = "work_dirs/demo_cli"

T = 7



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
        shorter = 512  # must divided by 64

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
def main(
    input_imgs: str,
    seed: int = 23,
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
    num_steps = 2,
    cfg_min = 1.2,
    learning_rate = 1e-6,
    learning_rate_lora = 1e-5,
    epochs=2,
    s_tmin=0.0, 
    s_tmax=999.0
):
    H, W = 512, 512
    
    
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16'
    )
    device = accelerator.device
    seed_everything(seed)
    # Dataset
    dataset = Image_Camera(root_dir=input_imgs, preset_traj=preset_traj, num_frames=num_frames, zoom_factor=zoom_factor, camera_scale=camera_scale)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Model, ae, condition
    teacher = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    teacher.module.to(dtype=torch.bfloat16)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q"],  # layers to inject LoRA
        bias="none"
    )
    teacher = get_peft_model(teacher, config)
    teacher.print_trainable_parameters()

    #teacher = torch.compile(teacher, mode="reduce-overhead")
    ae = AutoEncoder(chunk_size=1).to(device).eval()
    conditioner = CLIPConditioner().to(device).eval()
    ##
    ae.requires_grad_(False)
    conditioner.requires_grad_(False)
    ##
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)
    guider=MultiviewCFG(cfg_min)
    ##
    optimizer_lora = AdamW(teacher.parameters(), lr = learning_rate_lora)
    ##
    lr_scheduler_lora = CosineAnnealingLR(optimizer_lora, T_max=100)
    ##
    teacher, dataloader = accelerator.prepare(teacher, dataloader)
    optimizer_lora, lr_scheduler_lora = accelerator.prepare(optimizer_lora, lr_scheduler_lora)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for value_dict in dataloader:
            torch.cuda.reset_peak_memory_stats()
            # Clear gradients

            # Extract batch data
            imgs = value_dict["cond_frames"][0].to(device)
            input_masks = value_dict["cond_frames_mask"][0].to(device)
            pluckers = value_dict["plucker_coordinate"][0].to(device)
            
            
            # --- Encode ---
            ae.to(device)
            conditioner.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latents = torch.nn.functional.pad(
                    ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0
                )

                c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
            ae.to('cpu')
            conditioner.to('cpu')
            uc_crossattn = torch.zeros_like(c_crossattn)

            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)

            c_concat = torch.cat(
                [
                    repeat(
                        input_masks,
                        "n -> n 1 h w",
                        h=pluckers.shape[2],
                        w=pluckers.shape[3],
                    ),
                    pluckers,
                ],
                1,
            )
            uc_concat = torch.cat(
                [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
            )

            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector

            cond = {
                "crossattn": c_crossattn,
                "replace": c_replace,
                "concat": c_concat,
                "dense_vector": c_dense_vector,
            }
            uc = {
                "crossattn": uc_crossattn,
                "replace": uc_replace,
                "concat": uc_concat,
                "dense_vector": uc_dense_vector,
            }
            guider_kwargs = {
                "c2w": value_dict["c2w"][0].to(device),
                "K": value_dict["K"][0].to(device),
                "input_frame_mask": value_dict["cond_frames_mask"][0].to(device),
            }
            del imgs, input_masks, pluckers
            # Sample initial noise
            latent = torch.randn((T, 4, H // 8, W // 8), dtype=torch.bfloat16, device=device)
            
            sigmas = denoiser.discretization(num_steps, device=device)
            
                
            # random step
            # first_step = sigmas[0] = 84.9163
            second_step = (75.2586 - 0.1634) * torch.rand(1, device=device) + 0.1634
            student_sigmas = torch.cat([sigmas[0:1], second_step])
            student_sigmas = torch.sort(student_sigmas, descending=True).values
            # student sigma will have first step with big noise, and second is random noise level
            
            s_in = latent.new_ones([latent.shape[0]])
            s_churn=1.0 # 1.0 for student, 0.0 for teacher
            
            next_sigma = s_in * 0.0 # next_sigma = 0 for student predict x_0
            guidance_scale = random.uniform(1.2, 4.0)
            for step_idx, sigma_step in enumerate(student_sigmas):
                gamma = min(s_churn / num_steps, 2**0.5 - 1) if s_tmin <= sigma_step <= s_tmax else 0.0
                sigma = s_in * sigma_step
                
                sigma_hat = sigma * (gamma + 1.0) + 1e-6
                x = latent * torch.sqrt(1.0 + sigma_step ** 2.0)
                
                lora_denoised = denoiser(teacher, *guider.prepare_inputs(x, sigma_hat, cond, uc), num_frames=T)
                # breakpoint()
                lora_pred = guider(lora_denoised, sigma_hat, guidance_scale, **guider_kwargs)
                num_params = 0
                for name, param in teacher.named_parameters():
                    if param.requires_grad:
                        num_params += param.numel()
                print("Num params:", num_params)

                loss_lora = F.mse_loss(lora_pred.float(), x.detach().float(), reduction='mean')
                
                with accelerator.accumulate(teacher): 
                    accelerator.backward(loss_lora)
                    optimizer_lora.step()
                    lr_scheduler_lora.step()
                    optimizer_lora.zero_grad()
                print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                print(f"Peak memory reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                free, total = torch.cuda.mem_get_info()
                print(f"Free: {free/1024**3:.2f} GB, Total: {total/1024**3:.2f} GB")
        
if __name__ == "__main__":
    tyro.cli(main)
check my code, lora use so much vram ?

