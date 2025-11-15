import copy
import os.path as osp
from datetime import datetime
import os
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from einops import repeat
from seva.eval import transform_img_and_K, get_value_dict, save_output, seed_everything, decode_output
from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics, get_preset_pose_fov
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model
from seva.sampling import (
    EulerEDMSampler,
    MultiviewCFG,
)
device = "cuda:0"

# Constants
WORK_DIR = "work_dirs/demo_cli"
model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
ae = AutoEncoder(chunk_size=1).to(device)
conditioner = CLIPConditioner().to(device)
denoiser = DiscreteDenoiser(num_idx=1000, device=device)

T = 21
def preprocess(input_img_path: str):
    shorter = 576
    shorter = round(shorter / 64) * 64

    input_imgs = torch.as_tensor(
        iio.imread(input_img_path) / 255.0, dtype=torch.float32
    )[None, ..., :3]
    input_imgs = transform_img_and_K(
        input_imgs.permute(0, 3, 1, 2), shorter, K=None, size_stride=64
    )[0].permute(0, 2, 3, 1)

    input_Ks = get_default_intrinsics(
        aspect_ratio=input_imgs.shape[2] / input_imgs.shape[1]
    )
    input_c2ws = torch.eye(4)[None]

    return {
        "input_imgs": input_imgs,
        "input_Ks": input_Ks,
        "input_c2ws": input_c2ws,
        "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
    }
class SevaRenderer:
    def get_target_c2ws_and_Ks_from_preset(
        self, preprocessed: dict, preset_traj: str, num_frames: int, zoom_factor: float
    ):
        img_wh = preprocessed["input_wh"]
        start_c2w = preprocessed["input_c2ws"][0]
        start_w2c = torch.linalg.inv(start_c2w)
        look_at = torch.tensor([0, 0, 10])
        start_fov = DEFAULT_FOV_RAD #DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default

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

    def render(
        self,
        preprocessed: dict,
        seed: int,
        chunk_strategy: str,
        cfg: float,
        preset_traj: str,
        num_frames: int,
        zoom_factor: float,
        camera_scale: float,
    ):
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = osp.join(WORK_DIR, render_name)

        input_imgs, input_Ks, input_c2ws, (W, H) = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
        )

        num_inputs = len(input_imgs)
        input_c2ws = torch.eye(4)[None].to(dtype=input_c2ws.dtype)
        target_c2ws, target_Ks = self.get_target_c2ws_and_Ks_from_preset(
            preprocessed, preset_traj, num_frames, zoom_factor
        )
        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1])[
            :, None
        ]
        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))

        # Get anchor cameras
        num_anchors = T - num_inputs
     
        anchor_indices = np.linspace(
            num_inputs, num_inputs + num_targets - 1, num_anchors
        ).tolist()
        traj_prior_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]] # anchor_c2ws
        traj_prior_Ks = all_Ks[[round(ind) for ind in anchor_indices]] # anchor_Ks

        # Conditioning
        all_imgs_np = (
            F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
            * 255.0
        ).astype(np.uint8)
        image_cond = {
            "img": all_imgs_np,
            "input_indices": input_indices,
        }
        camera_cond = {
            "c2w": all_c2ws,
            "K": all_Ks,
            "input_indices": list(range(num_inputs + num_targets)),
        }

        # Options
        num_steps = 50
        options = {}
        options["camera_scale"] = camera_scale
        options["num_steps"] = num_steps
        options["cfg_min"] = 1.2

        print("Rendering started...")
        
        # --- Normalize input images only ---
        input_imgs = torch.as_tensor(image_cond["img"][0]).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        input_imgs = input_imgs / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        K = camera_cond["K"][0]
        # normalize intrinsics
        K[0] /= W
        K[1] /= H
        camera_cond["K"][0] = K
        # --- Get corresponding poses ---
        input_c2ws = camera_cond["c2w"][image_cond["input_indices"]]
        input_Ks   = camera_cond["K"][image_cond["input_indices"]]

        # --- Handle trajectory priors ---
        traj_prior_c2ws = torch.as_tensor(
            traj_prior_c2ws,
            device=input_c2ws.device,
            dtype=input_c2ws.dtype,
        )
        traj_prior_imgs = input_imgs.new_zeros(traj_prior_c2ws.shape[0], *input_imgs.shape[1:])

        if traj_prior_Ks is not None:
            for i, prior_k in enumerate(traj_prior_Ks):
                prior_k = prior_k.clone()
                prior_k[0] /= W
                prior_k[1] /= H
                traj_prior_Ks[i] = prior_k

        # --- Save inputs (optional) ---
        if options.get("save_input", True):
            print(f"Saving input images to {save_path} ...")
            save_output(
                {"/image": input_imgs},
                save_path=os.path.join(save_path, "input"),
                video_save_fps=2,
            )

        torch.cuda.empty_cache()
        seed_everything(seed)

        # --- Handle T split ---
        T_first_pass  = T

        # --- Simplified case: directly concat input + priors ---
        print("Two passes (first) - simplified 1-input + N-tests case")

        curr_input_sels = [0]  # input always at index 0
        curr_imgs = torch.cat([input_imgs, traj_prior_imgs[:T_first_pass-1]], dim=0)
        curr_c2ws = torch.cat([input_c2ws, traj_prior_c2ws[:T_first_pass-1]], dim=0)
        curr_Ks   = torch.cat([input_Ks,   traj_prior_Ks[:T_first_pass-1]], dim=0)

        # --- Build value_dict ---
        value_dict = get_value_dict(
            curr_imgs.to("cuda"),
            curr_input_sels,
            curr_c2ws,
            curr_Ks,
            list(range(T_first_pass)),
            all_c2ws=camera_cond["c2w"],
            camera_scale=options.get("camera_scale", 2.0),
        )

        sampler = EulerEDMSampler(
            discretization=denoiser.discretization,
            guider=MultiviewCFG(options["cfg_min"]),
            num_steps=options["num_steps"],
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999.0,
            s_noise=1.0,
            verbose=True,
            device=device,
        )

        imgs = value_dict["cond_frames"].to("cuda")
        input_masks = value_dict["cond_frames_mask"].to("cuda")
        pluckers = value_dict["plucker_coordinate"].to("cuda")

        # --- Encode ---
        #model.module.to(dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = torch.nn.functional.pad(
                ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0
            )

            c_crossattn = repeat(conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T_first_pass)
            uc_crossattn = torch.zeros_like(c_crossattn)

            c_replace = latents.new_zeros(T_first_pass, *latents.shape[1:])
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
                [pluckers.new_zeros(T_first_pass, 1, *pluckers.shape[-2:]), pluckers], 1
            )

            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector

            c = {
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

            additional_model_inputs = {"num_frames": T_first_pass}
            additional_sampler_inputs = {
                "c2w": value_dict["c2w"].to("cuda"),
                "K": value_dict["K"].to("cuda"),
                "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
            }

            randn = torch.randn((T_first_pass, 4, H // 8, W // 8)).to("cuda")
           
            samples_z = sampler(
                lambda input, sigma, c: denoiser(
                    model,
                    input,
                    sigma,
                    c,
                    **additional_model_inputs,
                ),
                randn,
                scale=float(cfg),
                cond=c,
                uc=uc,
                verbose=True,
                **additional_sampler_inputs,
            )
            samples = ae.decode(samples_z, 1)

            samples = decode_output(samples[1:], T_first_pass)

            # Save outputs if required
            if options.get("save_first_pass", True):
                save_output(
                    samples,
                    save_path=os.path.join(save_path, "first-pass"),
                    video_save_fps=5,
                )
                video_path_0 = os.path.join(save_path, "first-pass", "samples-rgb.mp4")
                print(f"Generated video : {video_path_0}")


def main(
    input_img: str,
    seed: int = 23,
    chunk_strategy: str = "interp",
    cfg: float = 4.0,
    preset_traj: str = "orbit",
    num_frames: int = 20,
    zoom_factor: float = 1.0,
    camera_scale: float = 2.0,
):
    torch.cuda.reset_peak_memory_stats()
    renderer = SevaRenderer()
    preprocessed = preprocess(input_img)
    renderer.render(
        preprocessed,
        seed,
        chunk_strategy,
        cfg,
        preset_traj,
        num_frames,
        zoom_factor,
        camera_scale,
    )
    print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Peak memory reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    print(torch.cuda.mem_get_info())
    free, total = torch.cuda.mem_get_info()
    print(f"Free: {free/1024**3:.2f} GB, Total: {total/1024**3:.2f} GB")


if __name__ == "__main__":
    tyro.cli(main)


