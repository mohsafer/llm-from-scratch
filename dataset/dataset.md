Download the quick-reproduction data into ./dataset/
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset \
  --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" \
  --local_dir ./dataset
