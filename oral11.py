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
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seva.sampling import DiscreteDenoiser, append_dims, to_d
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

def main(
    input_imgs: str,
    eval_imgs: str,
    seed: int = 36,
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
    num_steps = 2,
    cfg_min = 1.2,
    learning_rate = 1e-5,
    num_train_epochs=2,
    train_batch_size=1,
    eval_step = 2,
    save_ckp_step=4,
    num_steps_eval=4,
    gradient_accumulation_steps=8,
    output_dir='output_sds'
):
    H, W = 576, 576
    
    
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='bf16'
    )
    device = accelerator.device
    seed_everything(seed)
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    # Dataset
    dataset = Image_Camera(root_dir=input_imgs, preset_traj=preset_traj, num_frames=num_frames, zoom_factor=zoom_factor, camera_scale=camera_scale)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_dataset = Image_Camera(root_dir=eval_imgs, preset_traj=preset_traj, num_frames=num_frames, zoom_factor=zoom_factor, camera_scale=camera_scale)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Model, ae, condition
    teacher = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    #teacher.module.to(dtype=torch.bfloat16)

    student = SGMWrapper(load_model(device="cpu", verbose=True)).to(dtype=torch.bfloat16)
    student.to(device)
    
    student.train()
    student.module.enable_gradient_checkpointing()
    ae = AutoEncoder(chunk_size=1).to(device).eval()
    conditioner = CLIPConditioner().to(device).eval()
    ##
    ae.requires_grad_(False)
    conditioner.requires_grad_(False)
    teacher.requires_grad_(False)
    ##
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)
    guider=MultiviewCFG(cfg_min)
    ##
    optimizer_student = AdamW(student.parameters(), lr = learning_rate)

    ##
    lr_scheduler = CosineAnnealingLR(optimizer_student, T_max=100)

    ##
    student, dataloader = accelerator.prepare(student, dataloader)
    optimizer_student, lr_scheduler = accelerator.prepare(optimizer_student, lr_scheduler)
    ###
    global_step = 0
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_training_steps = num_update_steps_per_epoch * num_train_epochs * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Number of total training steps = {total_training_steps}")
    
    progress_bar = tqdm(
        range(0, total_training_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    ####
    for epoch in range(num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")
        for batch_idx, value_dict in enumerate(dataloader):
            torch.cuda.reset_peak_memory_stats()
            # Extract batch data
            imgs = value_dict["cond_frames"][0].to(device)
            input_masks = value_dict["cond_frames_mask"][0].to(device)
            pluckers = value_dict["plucker_coordinate"][0].to(device)
            
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latents = torch.nn.functional.pad(
                    ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0
                )

                c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)

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
            
            # first_step = sigmas[0] = 84.9163
            second_step = (75.2586 - 0.1634) * torch.rand(1, device=device) + 0.1634
            student_sigmas = torch.cat([sigmas[0:1], second_step])
            student_sigmas = torch.sort(student_sigmas, descending=True).values
            # student sigma will have first step with big noise, and second is random noise level
            
            s_in = latent.new_ones([latent.shape[0]])
            
            guidance_scale = random.uniform(3.0, 4.0)
            for step_idx, sigma_step in enumerate(student_sigmas):
                sigma = s_in * sigma_step
                if step_idx == 0:
                    x = latent * torch.sqrt(1.0 + sigma_step ** 2.0)
                else:
                    noise = torch.randn_like(x)
                    x = latent + noise * append_dims(sigma, latent.ndim)
                 
                student_original = denoiser(student, x, sigma, cond, num_frames=T)
         
                ############################################################################################################
                # SDS LOSS COMPUTATION
                ############################################################################################################
                
                # Sample noise for teacher evaluation
                noise = torch.randn_like(student_original)
                
                # Sample teacher sigma (different from student sigma for SDS)
                teacher_sigma_val = (84.9163 - 1) * torch.rand(1, device=device) + 1
                teacher_sigma = s_in * teacher_sigma_val
                teacher_sigma_expanded = append_dims(teacher_sigma, student_original.ndim)
                
                # Add noise to student's prediction for teacher evaluation
                noisy_samples = student_original + noise * teacher_sigma_expanded
                     
                # Teacher (frozen) prediction
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_denoised = denoiser(teacher, *guider.prepare_inputs(noisy_samples, teacher_sigma, cond, uc), num_frames=T)
                    teacher_denoised = guider(teacher_denoised, teacher_sigma, guidance_scale, **guider_kwargs)
                    teacher_noise_pred = to_d(noisy_samples, teacher_sigma, teacher_denoised)
                
                # Compute VSD gradient in score space
                
                score_gradient = (teacher_noise_pred - noise) / teacher_sigma_expanded
                
                # VSD loss: ||student_x0 - (student_x0 - score_gradient)||²
                target = (student_original - score_gradient).detach()
                loss_sds = 0.5 * F.mse_loss(student_original.float(), target.float(), reduction="mean")
               
                # Backward pass
                with accelerator.accumulate(student):
                    accelerator.backward(loss_sds)

                    if accelerator.sync_gradients:
                        optimizer_student.step()
                        lr_scheduler.step()
                        optimizer_student.zero_grad()
                    
                latent = student_original.detach()
                
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss_sds": loss_sds}, step=global_step
                                )
                progress_bar.update(1)
                if accelerator.is_main_process:   
                    if (global_step + 1) % eval_step == 0:
                        for batch_idx, value_dict in enumerate(testloader):
                            # Extract batch data
                            imgs = value_dict["cond_frames"][0].to(device)
                            input_masks = value_dict["cond_frames_mask"][0].to(device)
                            pluckers = value_dict["plucker_coordinate"][0].to(device)
                            
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                latents = torch.nn.functional.pad(
                                    ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0
                                )

                                c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)

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
                            x = torch.randn((T, 4, H // 8, W // 8)).to(device)
                            sigmas = denoiser.discretization(num_steps_eval, device=device)
                            x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
                            num_sigmas = len(sigmas)
                            s_in = x.new_ones([x.shape[0]])
                            
                            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                                for i in tqdm(range(num_sigmas - 1), total=num_sigmas - 1, desc="Sampling", leave=False):
                                    sigma = s_in * sigmas[i]
                                    next_sigma = s_in * sigmas[i + 1]
                                    denoised = denoiser(student, x, sigma, cond, num_frames=T)
                                    d = to_d(x, sigma, denoised)
                                    dt = append_dims(next_sigma - sigma, x.ndim)
                                    x = x + dt * d
                                samples = ae.decode(x, 1)

                                samples = decode_output(samples[1:], T)
                                # Save outputs
                                
                                save_output(
                                    samples,
                                    save_path=os.path.join(f'./log_sds/{global_step+1}', f"first-pass-{batch_idx}"),
                                    video_save_fps=5,
                                )
                    
                    if (global_step + 1) % save_ckp_step == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step+1}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")       
            logs = {
                "step_loss_sds": loss_sds.detach().item()
            }
            progress_bar.set_postfix(**logs)
if __name__ == "__main__":
    tyro.cli(main)
 

