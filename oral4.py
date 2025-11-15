above code is for DDPM scheduler, But i want make a version of EulerEDMSampler
class EulerEDMSampler():
    def __init__(
        self,
        discretization: Discretization,
        guider: VanillaCFG | MultiviewCFG | MultiviewTemporalCFG,
        num_steps: int | None = None,
        verbose: bool = False,
        device: str | torch.device = "cuda",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        **kwargs,
    ):
        self.num_steps = num_steps
        self.discretization = discretization
        self.guider = guider
        self.verbose = verbose
        self.device = device

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def prepare_sampling_loop(
        self, x: torch.Tensor, cond: dict, uc: dict, num_steps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict, dict]:
        num_steps = num_steps or self.num_steps
        assert num_steps is not None, "num_steps must be specified"
        sigmas = self.discretization(num_steps, device=self.device)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def get_sigma_gen(self, num_sigmas: int, verbose: bool = True) -> range | tqdm:
        sigma_generator = range(num_sigmas - 1)
        if self.verbose and verbose:
            sigma_generator = tqdm(
                range(num_sigmas - 1),
                total=num_sigmas - 1,
                desc="Sampling",
                leave=False,
            )
        return sigma_generator

    def sampler_step(
        self,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict,
        gamma: float = 0.0,
        **guider_kwargs,
    ) -> torch.Tensor:
        sigma_hat = sigma * (gamma + 1.0) + 1e-6

        eps = torch.randn_like(x) * self.s_noise
        x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = denoiser(*self.guider.prepare_inputs(x, sigma_hat, cond, uc))
        denoised = self.guider(denoised, sigma_hat, scale, **guider_kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        return x + dt * d

    def __call__(
        self,
        denoiser,
        x: torch.Tensor,
        scale: float | torch.Tensor,
        cond: dict,
        uc: dict | None = None,
        num_steps: int | None = None,
        verbose: bool = True,
        global_pbar: gr.Progress | None = None,
        **guider_kwargs,
    ) -> torch.Tensor:
        
        uc = cond if uc is None else uc
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x,
            cond,
            uc,
            num_steps,
        )
        for i in self.get_sigma_gen(num_sigmas, verbose=verbose):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                scale,
                cond,
                uc,
                gamma,
                **guider_kwargs,
            )
        return x
this is my infer code:
for i, epoch in enumerate(range(epochs)):
        for value_dict in dataloader:
            imgs = value_dict["cond_frames"][0].to("cuda")
            input_masks = value_dict["cond_frames_mask"][0].to("cuda")
            pluckers = value_dict["plucker_coordinate"][0].to("cuda")
            
            
            # --- Encode ---
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
                    "c2w": value_dict["c2w"][0].to("cuda"),
                    "K": value_dict["K"][0].to("cuda"),
                    "input_frame_mask": value_dict["cond_frames_mask"][0].to("cuda"),
                }
          
                x = torch.randn((T, 4, H // 8, W // 8)).to("cuda")

                denoise_teacher = lambda input, sigma, c: denoiser(
                        teacher,
                        input,
                        sigma,
                        c,
                        num_frames=T,
                    )
                
                uc = cond if uc is None else uc
                sigmas = denoiser.discretization(num_steps, device=device)
                x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
                num_sigmas = len(sigmas)
                s_in = x.new_ones([x.shape[0]])
                s_churn=0.0
                s_tmin=0.0
                s_tmax=999.0
                test = True
                for i in tqdm(range(num_sigmas - 1), total=num_sigmas - 1, desc="Sampling", leave=False):
                    gamma = (
                        min(s_churn / (num_sigmas - 1), 2**0.5 - 1)
                        if s_tmin <= sigmas[i] <= s_tmax
                        else 0.0
                    )
                    
                    sigma = s_in * sigmas[i]
                    next_sigma = s_in * sigmas[i + 1]
                    sigma_hat = sigma * (gamma + 1.0) + 1e-6

                    eps = torch.randn_like(x) 
                    x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
                    denoised = denoise_teacher(*guider.prepare_inputs(x, sigma_hat, cond, uc))
                    denoised = guider(denoised, sigma_hat, cfg, **guider_kwargs)
            
                    d = to_d(x, sigma_hat, denoised)
                    dt = append_dims(next_sigma - sigma_hat, x.ndim)
                    x = x + dt * d
                samples = ae.decode(x, 1)

                samples = decode_output(samples[1:], T)
