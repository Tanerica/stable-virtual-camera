export HF_HOME=../cache/.cache/huggingface
python oral15.py \
  --input-img assets/basic/blue-car.jpg \
  --cfg 4 \
  --num-frames 20 \
  --camera-scale 2 \
  --preset-traj orbit
