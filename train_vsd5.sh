torchrun --nproc_per_node=2 --master_port=29502 train_vsd5.py \
  --input_imgs ./data100 \
  --eval_imgs ./eval_data \
  --num_train_epochs 300 \
  --train_batch_size 1 \
  --eval_step 100 \
  --save_ckp_step 500 \
  --num_steps_eval 4 \
  --gradient_accumulation_steps 8 \
  --output_dir output_vsd \
  --cfg 4 \
  --num-frames 20 \
  --camera-scale 2 \
  --preset-traj orbit


