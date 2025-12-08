nohup python train.py \
  --method_type RoME \
  --problem_type IS IP SC \
  --gnn_type moe \
  --device cuda:0 \
  --num_dedicate_experts 5 \
  --num_shared_experts 1 \
  --top_k 2 \
  --lr 0.0001 \
  --num_epochs 150 \
  > logs/rome_train.log 2>&1 &
