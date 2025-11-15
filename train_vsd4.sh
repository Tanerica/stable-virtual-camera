torchrun --nproc_per_node=2 train_vsd4.py \
  --master_port=29502 \
  --input_imgs ./data \
  --eval_imgs ./eval_data \
  --num_train_epochs 100 \
  --train_batch_size 1 \
  --eval_step 10 \
  --save_ckp_step 10 \
  --num_steps_eval 4 \
  --gradient_accumulation_steps 8 \
  --output_dir output_vsd \
  --cfg 4 \
  --num-frames 20 \
  --camera-scale 2 \
  --preset-traj orbit


