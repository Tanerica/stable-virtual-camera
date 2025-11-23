torchrun --nproc_per_node=2 --master_port=29502 oral25.py \
  --input_imgs ./sana_gen/data \
  --eval_imgs ./sana_gen/eval_data \
  --num_train_epochs 100 \
  --train_batch_size 1 \
  --eval_step 50 \
  --save_ckp_step 1000 \
  --num_steps_eval 4 \
  --gradient_accumulation_steps 8 \
  --output_dir output_sds \
  --cfg 4 \
  --num-frames 20 \
  --camera-scale 2 \
  --preset-traj orbit


