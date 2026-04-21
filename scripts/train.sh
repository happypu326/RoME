nohup python train.py \
  --method_type RoME \
  --problem_type BP CA CFLP GISP IP LB MIS MVC SC \
  --gnn_type moe \
  --device cuda:2 \
  --num_dedicate_experts 16 \
  --num_shared_experts 2 \
  --top_k 8 \
  --eps_wasserstein 0.2 \
  --lr 0.0001 \
  --num_epochs 80 \
  --patience  20 \
  --min_delta 0.0001 \
  > logs/rome_train.log 2>&1 &