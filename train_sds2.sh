torchrun --nproc_per_node=1 --master_port=29502 train_sds2.py \
  --input_imgs ./sana_gen/eval_data1 \
  --eval_imgs ./sana_gen/eval_data1 \
  --num_train_epochs 10000 \
  --train_batch_size 1 \
  --eval_step 50 \
  --save_ckp_step 1000 \
  --num_steps_eval 2 \
  --gradient_accumulation_steps 1 \
  --output_dir output_sds2 \
  --cfg 4 \
  --num-frames 20 \
  --camera-scale 2 \
  --preset-traj orbit \
  --log_dir log_sds2


