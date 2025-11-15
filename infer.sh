export HF_HOME=../cache/.cache/huggingface
python oral.py \
  --input-img assets/basic/blue-car.jpg \
  --cfg 4 \
  --num-frames 80 \
  --camera-scale 2 \
  --preset-traj orbit
