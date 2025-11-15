        x = torch.randn((T, 4, H // 8, W // 8)).to("cuda") 
       
        sigmas = denoiser.discretization(num_steps, device=device)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        
        for i in tqdm(range(num_sigmas - 1), total=num_sigmas - 1, desc="Sampling", leave=False):
            
            sigma = s_in * sigmas[i]
            next_sigma = s_in * sigmas[i + 1]
            
            sigma2 = denoiser.idx_to_sigma(denoiser.sigma_to_idx(sigma))
            sigma_shape = sigma2.shape
            sigma2 = append_dims(sigma2, x.ndim)
            c_in = 1 / (sigma2**2 + 1.0) ** 0.5
            c_noise = sigma2.clone()
            
            c_noise = denoiser.sigma_to_idx(c_noise.reshape(sigma_shape))
            
            xi, mask = c_replace.split((x.shape[1], 1), dim=1)
            input_c = x * (1 - mask) + xi * mask
            
            xi, mask = uc_replace.split((x.shape[1], 1), dim=1)
            input_uc = x * (1 - mask) + xi * mask
            
            x_c = model(input_c * c_in, c_noise, cond, num_frames=T) * -sigma2 + input_c
            x_u = model(input_uc * c_in, c_noise, uc, num_frames=T) * -sigma2 + input_uc

            scale = guider.scale_rule(cfg, **guider_kwargs)
            denoised = guider.guidance(x_u, x_c, guider._expand_scale(sigma, scale))
            
            d = to_d(x, sigma, denoised)
            dt = append_dims(next_sigma - sigma, x.ndim)
            x = x + dt * d
        samples = ae.decode(x, 1)
