def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        # input blocks
        for i, module in enumerate(self.input_blocks):
            if self.is_gradient_checkpointing and self.training and i > 1:
                # use the injected checkpoint function from ModelMixin
                h = self._gradient_checkpointing_func(module, h, t_emb, y, dense_y, num_frames)
            else:
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
            hs.append(h)

        # middle block
        # if self.is_gradient_checkpointing and self.training:
        #     h = self._gradient_checkpointing_func(self.middle_block, h, t_emb, y, dense_y, num_frames)
        # else:
        h = self.middle_block(
            h, emb=t_emb, context=y, dense_emb=dense_y, num_frames=num_frames
        )

        # output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if self.is_gradient_checkpointing and self.training:
                h = self._gradient_checkpointing_func(module, h, t_emb, y, dense_y, num_frames)
            else:
                h = module(
                    h,
                    emb=t_emb,
                    context=y,
                    dense_emb=dense_y,
                    num_frames=num_frames,
                )
        return self.out(h)
